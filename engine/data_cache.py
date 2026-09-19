#!/usr/bin/env python3
"""
data_cache.py - Cache local de datos de Binance en parquet.

Descarga una vez, reutiliza en todas las fases del pipeline.
Soporta ambos formatos: raw list (lab lite) y DataFrame (lab historico/validador).
"""

import os
import time
import ccxt
import pandas as pd
import numpy as np

CACHE_DIR = "data_cache"
_exchange = None
_frozen = False  # Tras prefetch, solo lectura (no descargar)
_trim_tail = 0   # Recortar ultimas N velas (para --wf-offset)


def _get_exchange():
    global _exchange
    if _exchange is None:
        _exchange = ccxt.binance({'enableRateLimit': True})
    return _exchange


def _cache_path(symbol, timeframe):
    """Retorna path del archivo parquet para un simbolo/timeframe."""
    sc = symbol.replace("/", "")
    return os.path.join(CACHE_DIR, f"{sc}_{timeframe}.parquet")


def _download_candles(symbol, timeframe, total_candles, max_retries=3, existing_df=None):
    """Descarga velas de Binance. Si existing_df, extiende hacia atras.
    Retorna DataFrame combinado o None si falla."""
    exchange = _get_exchange()
    tf_ms = {'1h': 3600000, '4h': 14400000, '1d': 86400000}
    interval_ms = tf_ms.get(timeframe, 3600000)

    if existing_df is not None and len(existing_df) > 0:
        # Calcular cuantas velas faltan y descargar solo esas (hacia atras)
        missing = total_candles - len(existing_df) + 200  # margen
        earliest_ts = existing_df['timestamp_ms'].min()
        current_since = earliest_ts - missing * interval_ms
        end_before = earliest_ts  # no descargar lo que ya tenemos
        print(f"   [CACHE] Extendiendo cache: {len(existing_df)} existentes, descargando ~{missing} adicionales...")
    else:
        missing = total_candles + 200
        current_since = int(time.time() * 1000) - missing * interval_ms
        end_before = None
        print(f"   [CACHE] Descargando {total_candles} velas de {symbol} ({timeframe})...")

    all_candles = []
    requests_made = 0
    consecutive_errors = 0

    while len(all_candles) < missing:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
            if not ohlcv:
                break
            # Si extendemos, no pasar del timestamp existente
            if end_before is not None:
                ohlcv = [c for c in ohlcv if c[0] < end_before]
                if not ohlcv:
                    break
            all_candles.extend(ohlcv)
            current_since = ohlcv[-1][0] + interval_ms
            requests_made += 1
            consecutive_errors = 0
            time.sleep(0.1)
        except Exception as e:
            consecutive_errors += 1
            if consecutive_errors >= max_retries:
                print(f"   [CACHE] Error tras {max_retries} reintentos: {e}")
                break
            print(f"   [CACHE] Reintento {consecutive_errors}/{max_retries}: {e}")
            time.sleep(2 * consecutive_errors)

    # Combinar con datos existentes
    if all_candles:
        new_df = pd.DataFrame(all_candles, columns=['timestamp_ms', 'open', 'high', 'low', 'close', 'volume'])
        if existing_df is not None and len(existing_df) > 0:
            combined = pd.concat([new_df, existing_df], ignore_index=True)
        else:
            combined = new_df
        combined = combined.drop_duplicates(subset=['timestamp_ms']).sort_values('timestamp_ms').reset_index(drop=True)
        print(f"   [CACHE] {len(combined)} velas totales ({requests_made} requests)")
        return combined
    elif existing_df is not None and len(existing_df) > 0:
        # Descarga falló pero tenemos datos parciales - preservarlos
        print(f"   [CACHE] Descarga falló, preservando {len(existing_df)} velas en cache")
        return existing_df

    print(f"   [CACHE] Sin datos descargados")
    return None


def ensure_cached(symbol, timeframe, total_candles, max_retries=3):
    """Asegura que los datos estan en cache. Extiende si faltan velas.
    En modo frozen (tras prefetch), solo lee del cache sin descargar."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(symbol, timeframe)
    existing_df = None

    if os.path.exists(path):
        existing_df = pd.read_parquet(path)
        if len(existing_df) >= total_candles:
            print(f"   [CACHE] Hit: {len(existing_df)} velas en cache para {symbol}")
            return True
        if _frozen:
            print(f"   [CACHE] FROZEN: cache tiene {len(existing_df)} velas (necesita {total_candles}), usando disponibles")
            return True  # Usar lo que hay, no descargar
        print(f"   [CACHE] Cache parcial ({len(existing_df)} < {total_candles}), extendiendo...")
    elif _frozen:
        print(f"   [CACHE] FROZEN: sin cache para {symbol}, no se puede descargar")
        return False

    df = _download_candles(symbol, timeframe, total_candles, max_retries, existing_df)
    if df is None:
        return False

    df.to_parquet(path, index=False)
    print(f"   [CACHE] Guardado: {path} ({len(df)} velas)")
    return len(df) >= total_candles


def get_as_list(symbol, timeframe, total_candles):
    """Retorna datos como lista de listas OHLCV (formato lab lite).
    Cada elemento: [timestamp_ms, open, high, low, close, volume]
    Si _trim_tail > 0, recorta las ultimas N velas (modo wf-offset).
    """
    path = _cache_path(symbol, timeframe)
    if not os.path.exists(path):
        if not ensure_cached(symbol, timeframe, total_candles):
            return []

    df = pd.read_parquet(path)
    if _trim_tail > 0:
        df = df.iloc[:-_trim_tail].reset_index(drop=True)
    if len(df) > total_candles:
        df = df.tail(total_candles).reset_index(drop=True)

    return df[['timestamp_ms', 'open', 'high', 'low', 'close', 'volume']].values.tolist()


def get_as_dataframe(symbol, timeframe, total_candles):
    """Retorna datos como DataFrame (formato lab historico/validador).
    Columnas: timestamp (datetime), open, high, low, close, volume
    Si _trim_tail > 0, recorta las ultimas N velas (modo wf-offset).
    """
    path = _cache_path(symbol, timeframe)
    if not os.path.exists(path):
        if not ensure_cached(symbol, timeframe, total_candles):
            return None

    df = pd.read_parquet(path)
    if _trim_tail > 0:
        df = df.iloc[:-_trim_tail].reset_index(drop=True)
    if len(df) > total_candles:
        df = df.tail(total_candles).reset_index(drop=True)

    result = df.copy()
    result['timestamp'] = pd.to_datetime(result['timestamp_ms'], unit='ms')
    result = result[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    return result


def get_full_dataframe(symbol, timeframe, total_candles):
    """Retorna DataFrame COMPLETO sin recorte de _trim_tail.
    Usado por la fase forward del wf-offset para acceder a las velas recortadas.
    """
    path = _cache_path(symbol, timeframe)
    if not os.path.exists(path):
        return None

    df = pd.read_parquet(path)
    if len(df) > total_candles:
        df = df.tail(total_candles).reset_index(drop=True)

    result = df.copy()
    result['timestamp'] = pd.to_datetime(result['timestamp_ms'], unit='ms')
    result = result[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    return result


def set_trim_tail(n):
    """Activa recorte de las ultimas N velas en todas las lecturas."""
    global _trim_tail
    _trim_tail = n
    if n > 0:
        print(f"[CACHE] Trim tail activado: ultimas {n} velas invisibles para el pipeline")


def patch_modules(lite_mod, lab_mod, total_candles_lite, total_candles_lab, frozen=True):
    """Monkey-patch fetch_all_candles en ambos modulos para usar cache.
    frozen=True asegura que workers no intenten descargar."""
    global _frozen
    if frozen:
        _frozen = True

    def lite_fetch(symbol, timeframe, total_candles, max_retries=3):
        ensure_cached(symbol, timeframe, total_candles, max_retries)
        return get_as_list(symbol, timeframe, total_candles)

    def lab_fetch(symbol, timeframe, total_candles, max_retries=3):
        ensure_cached(symbol, timeframe, total_candles, max_retries)
        return get_as_dataframe(symbol, timeframe, total_candles)

    if lite_mod is not None:
        lite_mod.fetch_all_candles = lite_fetch

    if lab_mod is not None:
        lab_mod.fetch_all_candles = lab_fetch


def freeze():
    """Activa modo frozen: solo lectura, no mas descargas."""
    global _frozen
    _frozen = True
    print("[CACHE] Modo FROZEN activado: no se descargaran mas datos")


def prefetch_symbols(symbols, timeframe, total_candles):
    """Pre-descarga datos para multiples simbolos y congela el cache."""
    global _frozen
    _frozen = False  # Asegurar que podemos descargar durante prefetch
    print(f"\n[CACHE] Pre-descargando {len(symbols)} simbolos ({total_candles} velas c/u)...")
    ok = 0
    fail = []
    for i, symbol in enumerate(symbols, 1):
        print(f"  [{i}/{len(symbols)}] {symbol}")
        if ensure_cached(symbol, timeframe, total_candles):
            ok += 1
        else:
            fail.append(symbol)
    print(f"[CACHE] {ok}/{len(symbols)} simbolos en cache")
    if fail:
        print(f"[CACHE] ATENCION: {len(fail)} simbolos sin datos suficientes: {', '.join(fail)}")
    freeze()
    return ok
