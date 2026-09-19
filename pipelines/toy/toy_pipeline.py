# -*- coding: utf-8 -*-
"""
toy_pipeline.py — Entrypoint autónomo de la tubería TOY (patrón MR:
mean_reversion_walk_forward.py es entrypoint propio, no orquestado por master).

Uso (desde la raíz del toolkit):
  # (a) sobre la muestra sintética (generada por examples/data/make_synthetic.py):
  python pipelines/toy/toy_pipeline.py --data examples/data/TOYGBM1USDT_1h.parquet

  # (b) sobre datos reales de un símbolo (descarga si falta, ~1 min):
  python pipelines/toy/toy_pipeline.py --symbol BTC/USDT --download

Salida: tabla top-N legible por consola + CSV completo de la matriz de
resultados (n_configs × 7 + parámetros decodificados).
Sin walk-forward todavía (Sesión 3): esto es el paso "simulación del universo".
"""
import os
import sys
import time
import argparse

# Fix Windows cp1252 encoding (mismo patrón que engine/master.py:30-32)
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith('cp'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))          # raíz del toolkit
sys.path.insert(0, _HERE)                                 # toy_features/toy_kernel
sys.path.insert(0, os.path.join(_ROOT, 'engine'))         # descargador

import numpy as np
import pandas as pd

from toy_kernel import (generate_toy_configs, decode_toy_config, run_on_slice,
                        COMMISSION_ROUND_TRIP)
from toy_features import WARMUP_BARS


def _load_data(args):
    if args.data:
        path = args.data
        if not os.path.exists(path):
            sys.exit(f"ERROR: no existe {path}. Genera la muestra con "
                     f"examples/data/make_synthetic.py o pasa --symbol.")
        return pd.read_parquet(path), path
    sc = args.symbol.replace('/', '')
    path = os.path.join('data_cache', f"{sc}_1h.parquet")
    if not os.path.exists(path):
        if not args.download:
            sys.exit(f"ERROR: no existe {path}. Añade --download para bajarlo "
                     f"de Binance (solo ese símbolo) o pasa --data.")
        import download_full_history as dl
        import ccxt
        os.makedirs(dl.CACHE_DIR, exist_ok=True)
        exchange = ccxt.binance({'enableRateLimit': True})
        dl.download_full(args.symbol, exchange)
    return pd.read_parquet(path), path


def main():
    ap = argparse.ArgumentParser(description="Tubería TOY: cruce de dos EMAs")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--data', help="Parquet formato caché (timestamp_ms,open,high,low,close,volume)")
    src.add_argument('--symbol', help="Símbolo real, ej. BTC/USDT (lee data_cache/)")
    ap.add_argument('--download', action='store_true',
                    help="Con --symbol: descarga el histórico si no está en data_cache/")
    ap.add_argument('--top', type=int, default=10, help="Filas de la tabla (default 10)")
    ap.add_argument('--out', default=None, help="CSV de salida (default: toy_results_<fuente>.csv)")
    args = ap.parse_args()

    t_wall = time.perf_counter()
    df, src_path = _load_data(args)
    t_load = time.perf_counter() - t_wall
    print(f"[TOY] Datos: {src_path} — {len(df)} barras "
          f"({pd.to_datetime(df['timestamp_ms'].min(), unit='ms')} -> "
          f"{pd.to_datetime(df['timestamp_ms'].max(), unit='ms')})")

    configs = generate_toy_configs()
    print(f"[TOY] Universo: {len(configs)} configs válidas (bit-packed 6 bits, "
          f"filtro fast<slow) | warmup={WARMUP_BARS} | comisión={COMMISSION_ROUND_TRIP}% RT")

    t0 = time.perf_counter()
    results, meta = run_on_slice(configs, df)
    t_sim = time.perf_counter() - t0

    # Tabla legible
    rows = []
    for i, cid in enumerate(configs):
        d = decode_toy_config(cid)
        pnl, trades, wins, cancels, max_dd, gp, gl = results[i]
        pf = gp / gl if gl > 0 else (np.inf if gp > 0 else 0.0)
        winrate = 100.0 * wins / trades if trades > 0 else 0.0
        rows.append({
            'config_id': d['config_id'], 'fast': d['fast_period'],
            'slow': d['slow_period'], 'pnl_pct': round(pnl, 2),
            'trades': int(trades), 'wins': int(wins),
            'winrate_pct': round(winrate, 1), 'pf': round(pf, 3),
            'max_dd_pct': round(max_dd, 2), 'gross_profit': round(gp, 2),
            'gross_loss': round(gl, 2),
        })
    table = pd.DataFrame(rows).sort_values('pnl_pct', ascending=False).reset_index(drop=True)

    out = args.out or f"toy_results_{os.path.splitext(os.path.basename(src_path))[0]}.csv"
    table.to_csv(out, index=False)

    print(f"\n[TOY] Top {args.top} por PnL neto % (de {meta['n_configs']} configs, "
          f"{meta['n_bars']} barras, accounting desde barra {meta['accounting_start']}):\n")
    print(table.head(args.top).to_string(index=False))
    print(f"\n[TOY] Peor config: pnl={table['pnl_pct'].iloc[-1]}% "
          f"(cfg {table['config_id'].iloc[-1]}, {table['fast'].iloc[-1]}/{table['slow'].iloc[-1]})")
    print(f"[TOY] Resultados completos -> {out}")
    print(f"[TOY] Tiempos: carga {t_load:.2f}s | features+kernel {t_sim:.2f}s "
          f"(incluye compilación JIT en la primera ejecución) | "
          f"total {time.perf_counter()-t_wall:.2f}s")
    print("\n[TOY] Recordatorio del Arnés: un PnL alto AQUÍ no es edge — es el "
          "máximo de 61 intentos sobre una serie. Compara SIEMPRE contra el "
          "placebo GBM (misma tubería, ruido puro) antes de creer nada.")


if __name__ == "__main__":
    main()
