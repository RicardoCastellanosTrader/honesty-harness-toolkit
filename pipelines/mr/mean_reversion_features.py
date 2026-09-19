#!/usr/bin/env python
"""
mean_reversion_features.py -- Precalculo de datos para el preset Mean-Reversion

Slow line: HA diaria repintable (forming) / fija (resolved)
Fast line: Tenkan-sen (len=9) sobre velas 1h
Zona invertida: zone_bull = fast < slow

Genera data_cache/SYMBOL_mean_reversion.npz por simbolo.

Uso:
    python mean_reversion_features.py                     # Todos los 45 simbolos
    python mean_reversion_features.py --symbols BTC/USDT  # Solo uno
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

from lab_historico_numba_v8_3 import (
    calc_heikin_ashi,
    calc_ha_trend,
    precalculate_divergences_pine_faithful,
    calc_rsi,
    calc_macd,
    calc_stochastic,
    calc_vwmacd,
    calc_cmf,
    calc_cci,
    calc_momentum,
)
from master import CONFIG

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_CACHE_DIR = CONFIG['data_cache_dir']
SYMBOLS = CONFIG['symbols']

# Divergence parameters (same as lab v8.3)
DIV_RSI_LEN = 14
DIV_MACD_FAST = 12
DIV_MACD_SLOW = 26
DIV_MACD_SIGNAL = 9
DIV_STOCH_K = 14
DIV_STOCH_D = 3
DIV_STOCH_SMOOTH = 3
DIV_VWMACD_FAST = 12
DIV_VWMACD_SLOW = 26
DIV_CMF_LEN = 21
DIV_CCI_LEN = 10
DIV_MOM_LEN = 10
DIV_PIVOT_PERIOD = 5
DIV_MAX_PIVOTS = 10
DIV_MAX_BARS = 100


# ============================================
# 1. calc_heikin_ashi_hlc3
# ============================================

def calc_heikin_ashi_hlc3(df_daily):
    """
    Calcula HA candles recursivas sobre velas diarias.
    Retorna Series con hlc3 = (ha_high + ha_low + ha_close) / 3.

    Formula HA (Pine-faithful):
      ha_open[0] = (O + C) / 2
      ha_close[i] = (O + H + L + C) / 4
      ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
      ha_high[i] = max(H, ha_open, ha_close)
      ha_low[i]  = min(L, ha_open, ha_close)
    """
    ha = calc_heikin_ashi(df_daily)
    return (ha['high'] + ha['low'] + ha['close']) / 3


# ============================================
# 2. calc_slow_line_resolved (no repinta)
# ============================================

def calc_slow_line_resolved(df_1h, use_smoothing=False, smoothing_len=10):
    """
    Para cada barra 1h, slow line usando TODAS las velas diarias cerradas
    hasta ese momento. Shift(1) asegura que solo se usan dias completados.
    Alineada al indice 1h con forward-fill.
    """
    df_indexed = df_1h.set_index('timestamp')
    df_daily = df_indexed.resample('1D', closed='left', label='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()

    hlc3 = calc_heikin_ashi_hlc3(df_daily)

    if use_smoothing and smoothing_len > 1:
        hlc3 = hlc3.rolling(window=smoothing_len, min_periods=1).mean()

    # shift(1): solo usar dias cerrados; reindex+ffill al indice 1h
    resolved = hlc3.shift(1).reindex(df_indexed.index).ffill().values
    return resolved


# ============================================
# 3. calc_slow_line_forming (repinta intra-dia)
# ============================================

def calc_slow_line_forming(df_1h, use_smoothing=False, smoothing_len=10):
    """
    Para CADA barra 1h del dia actual, recalcula la HA diaria incluyendo
    la vela diaria parcial (forming). Captura el repintado intra-dia.
    Al cierre del dia, forming == resolved del dia siguiente.

    Optimizacion: aprovecha la recurrencia HA para O(n) en vez de
    recalcular toda la cadena HA por cada barra.
    """
    n = len(df_1h)
    result = np.full(n, np.nan)

    timestamps = pd.to_datetime(df_1h['timestamp'])
    o = df_1h['open'].values.astype(np.float64)
    h = df_1h['high'].values.astype(np.float64)
    l = df_1h['low'].values.astype(np.float64)
    c = df_1h['close'].values.astype(np.float64)

    # Pre-compute resolved daily HA for the base HA state per day
    df_indexed = df_1h.set_index('timestamp')
    df_daily = df_indexed.resample('1D', closed='left', label='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()
    ha_daily = calc_heikin_ashi(df_daily)

    ha_open_d = ha_daily['open'].values
    ha_close_d = ha_daily['close'].values
    ha_hlc3_d = ((ha_daily['high'] + ha_daily['low'] + ha_daily['close']) / 3).values
    daily_ts = ha_daily.index.values.astype('datetime64[D]')

    date_to_idx = {daily_ts[k]: k for k in range(len(daily_ts))}

    # Floor each 1h bar to its day (numpy datetime64[D])
    bar_days = timestamps.values.astype('datetime64[D]')

    current_day = None
    day_o = day_h = day_l = 0.0
    prev_ha_open = np.nan
    prev_ha_close = np.nan
    smoothing_buf = []  # resolved daily hlc3 for smoothing window

    for i in range(n):
        bd = bar_days[i]

        if bd != current_day:
            # Day transition: retrieve completed day's HA state
            if current_day is not None and current_day in date_to_idx:
                didx = date_to_idx[current_day]
                prev_ha_open = ha_open_d[didx]
                prev_ha_close = ha_close_d[didx]
                if use_smoothing:
                    smoothing_buf.append(ha_hlc3_d[didx])

            current_day = bd
            day_o = o[i]
            day_h = h[i]
            day_l = l[i]
        else:
            if h[i] > day_h:
                day_h = h[i]
            if l[i] < day_l:
                day_l = l[i]

        day_c = c[i]

        # Compute forming HA bar for the current partial day
        fha_close = (day_o + day_h + day_l + day_c) / 4.0
        if np.isnan(prev_ha_open):
            # First day: HA initialization (same as calc_heikin_ashi bar 0)
            fha_open = (day_o + day_c) / 2.0
        else:
            fha_open = (prev_ha_open + prev_ha_close) / 2.0

        fha_high = max(day_h, fha_open, fha_close)
        fha_low = min(day_l, fha_open, fha_close)
        forming_hlc3 = (fha_high + fha_low + fha_close) / 3.0

        if use_smoothing and smoothing_len > 1:
            tail = smoothing_buf[-(smoothing_len - 1):]
            result[i] = np.mean(tail + [forming_hlc3])
        else:
            result[i] = forming_hlc3

    return result


# ============================================
# 4. calc_fast_line (Tenkan-sen 1h)
# ============================================

def calc_fast_line(df_1h, tenkan_len=9):
    """
    Tenkan-sen: (highest(high, len) + lowest(low, len)) / 2
    Sobre velas 1h directamente.
    """
    h_series = pd.Series(df_1h['high'].values, dtype=np.float64)
    l_series = pd.Series(df_1h['low'].values, dtype=np.float64)
    highest = h_series.rolling(window=tenkan_len, min_periods=tenkan_len).max()
    lowest = l_series.rolling(window=tenkan_len, min_periods=tenkan_len).min()
    return ((highest + lowest) / 2.0).values


# ============================================
# 5. calc_zones (invertidas para mean-reversion)
# ============================================

# ---------------------------------------------------------------------------
# A05 S3 (2026-04-23): helpers compartidos para la semántica MR zone_bull/bear.
# Antes: brain_engine recomputaba `fast < slow` inline en 6 sitios distintos
# mientras `calc_zones` computaba arrays precomputados. Riesgo: drift
# silencioso si calc_zones cambiaba la fórmula sin actualizar brain. Fix:
# funciones puras module-level aquí, importadas por brain_engine y
# consumidas por calc_zones internamente. Single source of truth para la
# semántica MR invertida (§0.5).
# ---------------------------------------------------------------------------

def zone_bull_mr(fast, slow):
    """Mean-reversion zone_bull: fast < slow.

    Semántica invertida respecto al TF (donde zone_bull = fast > slow).
    Precio bajo la media → reversion al alza esperada (§0.5).

    Escalar-friendly (usado en brain_engine bar-by-bar) y array-friendly
    (usado por calc_zones vectorizado). NaN en cualquier input → False.

    Args:
        fast: valor escalar float o np.ndarray de la fast line (Tenkan 1h).
        slow: valor escalar float o np.ndarray de la slow line (HA daily
              smoothed).

    Returns:
        bool si inputs escalares, np.ndarray si arrays.
    """
    if np.isscalar(fast) and np.isscalar(slow):
        if np.isnan(fast) or np.isnan(slow):
            return False
        return bool(fast < slow)
    f = np.asarray(fast)
    s = np.asarray(slow)
    valid = ~(np.isnan(f) | np.isnan(s))
    result = np.zeros_like(f, dtype=np.bool_)
    result[valid] = f[valid] < s[valid]
    return result


def zone_bear_mr(fast, slow):
    """Mean-reversion zone_bear: fast > slow. Ver zone_bull_mr para semántica."""
    if np.isscalar(fast) and np.isscalar(slow):
        if np.isnan(fast) or np.isnan(slow):
            return False
        return bool(fast > slow)
    f = np.asarray(fast)
    s = np.asarray(slow)
    valid = ~(np.isnan(f) | np.isnan(s))
    result = np.zeros_like(f, dtype=np.bool_)
    result[valid] = f[valid] > s[valid]
    return result


def calc_zones(fast_line, slow_line_forming, slow_line_resolved):
    """
    Zona invertida para mean-reversion:
      zone_bull = fast < slow (precio debajo de la media -> reversion al alza)
      zone_bear = fast > slow

    A05 S3 refactor 2026-04-23: delega a zone_bull_mr/zone_bear_mr
    (single source of truth). Mantiene firma + shape de retorno idéntica
    para compatibilidad con consumers existentes.
    """
    return {
        'zone_bull_forming': zone_bull_mr(fast_line, slow_line_forming),
        'zone_bear_forming': zone_bear_mr(fast_line, slow_line_forming),
        'zone_bull_resolved': zone_bull_mr(fast_line, slow_line_resolved),
        'zone_bear_resolved': zone_bear_mr(fast_line, slow_line_resolved),
    }


# ============================================
# TF filters (forming + resolved)
# ============================================

def _compute_tf_forming_incremental(timestamps, o, h, l, c, rule):
    """
    Computa HA trend forming para un TF superior (4h o 1D) incrementalmente.
    Equivale a resamplear hasta la barra actual y tomar el ultimo ha_trend,
    pero en O(n) en vez de O(n^2).
    """
    n = len(o)
    result = np.ones(n, dtype=np.bool_)

    period_floor = timestamps.dt.floor(rule).values

    current_period = None
    p_open = p_high = p_low = p_close = 0.0
    prev_close = np.nan

    for i in range(n):
        pf = period_floor[i]

        if pf != current_period:
            if current_period is not None:
                prev_close = p_close
            current_period = pf
            p_open = o[i]
            p_high = h[i]
            p_low = l[i]
        else:
            if h[i] > p_high:
                p_high = h[i]
            if l[i] < p_low:
                p_low = l[i]

        p_close = c[i]

        # ha_trend: avg_price = (O+H+L+C)/4; avg_oc = (O + prev_close)/2
        avg_price = (p_open + p_high + p_low + p_close) / 4.0
        if np.isnan(prev_close):
            result[i] = True
        else:
            avg_oc = (p_open + prev_close) / 2.0
            result[i] = avg_price > avg_oc

    return result


def _compute_tf_filters(df_1h):
    """
    Filtros TF1 (1h HA), TF2 (4h HA), TF3 (1D HA)
    forming y resolved. Packed bits compatibles con el lab.

    Bit layout:
      bit 0: TF1 (1h HA trend)
      bit 1: TF2 (4h HA trend)
      bit 2: TF3 (1D HA trend)
    """
    n = len(df_1h)
    timestamps = pd.to_datetime(df_1h['timestamp'])
    o = df_1h['open'].values.astype(np.float64)
    h = df_1h['high'].values.astype(np.float64)
    l = df_1h['low'].values.astype(np.float64)
    c = df_1h['close'].values.astype(np.float64)

    df_indexed = df_1h.set_index('timestamp')

    # --- TF1: 1h HA trend (vectorized, same for forming & resolved) ---
    avg_price_1h = (o + h + l + c) / 4.0
    prev_c_1h = np.empty(n)
    prev_c_1h[0] = np.nan
    prev_c_1h[1:] = c[:-1]
    avg_oc_1h = (o + prev_c_1h) / 2.0
    tf1_bull = avg_price_1h > avg_oc_1h
    tf1_bull[0] = True  # default for first bar (NaN comparison)

    # --- TF2 resolved: 4h HA trend, shift(1), ffill ---
    df_4h = df_indexed.resample('4h', closed='left', label='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()
    tf2_resolved = (
        calc_ha_trend(df_4h)
        .shift(1)
        .reindex(df_indexed.index)
        .ffill()
        .fillna(True)
        .values.astype(bool)
    )

    # --- TF3 resolved: 1D HA trend, shift(1), ffill ---
    df_1d = df_indexed.resample('1D', closed='left', label='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()
    tf3_resolved = (
        calc_ha_trend(df_1d)
        .shift(1)
        .reindex(df_indexed.index)
        .ffill()
        .fillna(True)
        .values.astype(bool)
    )

    # --- TF2 forming: incremental 4h HA trend ---
    tf2_forming = _compute_tf_forming_incremental(timestamps, o, h, l, c, '4h')

    # --- TF3 forming: incremental 1D HA trend ---
    tf3_forming = _compute_tf_forming_incremental(timestamps, o, h, l, c, '1D')

    # Pack bits (vectorized)
    filters_forming = (
        tf1_bull.astype(np.uint32)
        | (tf2_forming.astype(np.uint32) << 1)
        | (tf3_forming.astype(np.uint32) << 2)
    )
    filters_resolved = (
        tf1_bull.astype(np.uint32)
        | (tf2_resolved.astype(np.uint32) << 1)
        | (tf3_resolved.astype(np.uint32) << 2)
    )

    return filters_forming, filters_resolved


# ============================================
# Divergences
# ============================================

def _compute_divergence_indicators(df_1h):
    """Calcula los 8 indicadores de divergencia sobre 1h (mismos que el lab)."""
    close = df_1h['close']
    high = df_1h['high']
    low = df_1h['low']
    volume = df_1h['volume']

    rsi = calc_rsi(close, DIV_RSI_LEN).values
    macd_line, macd_hist = calc_macd(close, DIV_MACD_FAST, DIV_MACD_SLOW, DIV_MACD_SIGNAL)
    stoch = calc_stochastic(high, low, close, DIV_STOCH_K, DIV_STOCH_D, DIV_STOCH_SMOOTH).values
    vwmacd = calc_vwmacd(close, volume, DIV_VWMACD_FAST, DIV_VWMACD_SLOW).values
    cmf = calc_cmf(high, low, close, volume, DIV_CMF_LEN).values
    cci = calc_cci(high, low, close, DIV_CCI_LEN).values
    mom = calc_momentum(close, DIV_MOM_LEN).values

    return {
        'rsi': rsi,
        'macd_hist': macd_hist.values,
        'macd_line': macd_line.values,
        'stoch': stoch,
        'vwmacd': vwmacd,
        'cmf': cmf,
        'cci': cci,
        'mom': mom,
    }


# ============================================
# 6. precalculate_mean_reversion (principal)
# ============================================

def precalculate_mean_reversion(symbol, df_1h):
    """
    Funcion principal: calcula todos los arrays para simulacion mean-reversion.
    Retorna dict con todos los arrays y guarda en data_cache/SYMBOL_mean_reversion.npz.
    """
    t0 = time.time()
    n = len(df_1h)
    sym = symbol.replace("/", "")
    print(f"[{sym}] Precalculando mean-reversion ({n} barras 1h)...")

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_1h['timestamp']):
        df_1h = df_1h.copy()
        df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'])

    # --- Fase 1: Fast line (Tenkan 9 sobre 1h) ---
    print(f"  [1/5] Fast line (Tenkan 9)...")
    fast_line = calc_fast_line(df_1h, tenkan_len=9)

    # --- Fase 2: Slow line resolved (HA diaria, solo dias cerrados) ---
    print(f"  [2/5] Slow line resolved (HA diaria cerrada)...")
    slow_resolved = calc_slow_line_resolved(df_1h)

    # --- Fase 3: Slow line forming (HA diaria parcial, repinta intra-dia) ---
    print(f"  [3/5] Slow line forming (HA diaria forming)...")
    slow_forming = calc_slow_line_forming(df_1h)

    # --- Fase 4: Zones invertidas + TF filters ---
    print(f"  [4/5] Zones + TF filters...")
    zones = calc_zones(fast_line, slow_forming, slow_resolved)
    filters_forming, filters_resolved = _compute_tf_filters(df_1h)

    # --- Fase 5: Divergences ---
    print(f"  [5/5] Divergencias (Pine-faithful)...")
    close_arr = df_1h['close'].values.astype(np.float64)
    high_arr = df_1h['high'].values.astype(np.float64)
    low_arr = df_1h['low'].values.astype(np.float64)

    indicators = _compute_divergence_indicators(df_1h)
    div_bits = precalculate_divergences_pine_faithful(
        close_arr, high_arr, low_arr, indicators, n,
        DIV_PIVOT_PERIOD, DIV_MAX_PIVOTS, DIV_MAX_BARS
    )

    # Fix hidden divergence bug: the lab precalc swaps hidden bull/bear.
    # In the raw output:
    #   bit 1 (bull_hid) = actually hidden BEAR  (found at pivot highs)
    #   bit 3 (bear_hid) = actually hidden BULL  (found at pivot lows)
    # Swap bits 1 and 3 so the MR pipeline has correct semantics:
    #   bit 0 = Regular Bull,  bit 1 = Hidden Bull (corrected)
    #   bit 2 = Regular Bear,  bit 3 = Hidden Bear (corrected)
    bit1 = (div_bits >> 1) & np.uint8(1)
    bit3 = (div_bits >> 3) & np.uint8(1)
    div_bits = (div_bits & np.uint8(0x05)) | (bit3 << 1).astype(np.uint8) | (bit1 << 3).astype(np.uint8)

    # Print divergence summary (now with corrected labels)
    ind_names = ["RSI", "MACD_H", "MACD_L", "STOCH", "VWMACD", "CMF", "CCI", "MOM"]
    for ii, name in enumerate(ind_names):
        br = int(np.sum((div_bits[100:, ii] & 1) > 0))
        bh = int(np.sum((div_bits[100:, ii] & 2) > 0))
        ber = int(np.sum((div_bits[100:, ii] & 4) > 0))
        beh = int(np.sum((div_bits[100:, ii] & 8) > 0))
        if ii == 0:
            print(f"      {'Ind':<8} {'BullR':>6} {'BullH':>6} {'BearR':>6} {'BearH':>6}")
        print(f"      {name:<8} {br:>6} {bh:>6} {ber:>6} {beh:>6}")

    # Build output dict
    data = {
        'close': close_arr,
        'high': high_arr,
        'low': low_arr,
        'timestamps': df_1h['timestamp'].values,
        'fast_line': fast_line,
        'slow_line_forming': slow_forming,
        'slow_line_resolved': slow_resolved,
        'zone_bull_forming': zones['zone_bull_forming'],
        'zone_bear_forming': zones['zone_bear_forming'],
        'zone_bull_resolved': zones['zone_bull_resolved'],
        'zone_bear_resolved': zones['zone_bear_resolved'],
        'filters_forming': filters_forming,
        'filters_resolved': filters_resolved,
        'div_bits': div_bits,
    }

    # Save .npz
    out_path = os.path.join(DATA_CACHE_DIR, f"{sym}_mean_reversion.npz")
    np.savez_compressed(out_path, **data)

    elapsed = time.time() - t0
    print(f"  Guardado -> {out_path} ({elapsed:.1f}s)")

    return data


# ============================================
# Verification
# ============================================

def _verify_btc(data):
    """Tests de verificacion para BTC/USDT."""
    print("\n--- Verificacion BTC/USDT ---")

    slow_f = data['slow_line_forming']
    slow_r = data['slow_line_resolved']
    fast = data['fast_line']
    ts = pd.to_datetime(data['timestamps'])

    # 1. Primeras 10 barras: forming vs resolved
    print("\nPrimeras 10 barras slow_line:")
    print(f"  {'Bar':>4} {'Timestamp':<20} {'Forming':>12} {'Resolved':>12} {'Diff':>10}")
    for i in range(min(10, len(slow_f))):
        f_val = slow_f[i]
        r_val = slow_r[i]
        if np.isnan(f_val) and np.isnan(r_val):
            print(f"  {i:>4} {str(ts[i]):<20} {'NaN':>12} {'NaN':>12} {'---':>10}")
        elif np.isnan(r_val):
            print(f"  {i:>4} {str(ts[i]):<20} {f_val:>12.2f} {'NaN':>12} {'---':>10}")
        elif np.isnan(f_val):
            print(f"  {i:>4} {str(ts[i]):<20} {'NaN':>12} {r_val:>12.2f} {'---':>10}")
        else:
            diff = f_val - r_val
            print(f"  {i:>4} {str(ts[i]):<20} {f_val:>12.2f} {r_val:>12.2f} {diff:>10.2f}")

    # 2. Verificar zona invertida: zone_bull_forming = (fast < slow_forming)
    zb = data['zone_bull_forming']
    valid = ~np.isnan(fast) & ~np.isnan(slow_f)
    if np.sum(valid) > 0:
        check_inv = np.all(zb[valid] == (fast[valid] < slow_f[valid]))
        n_bull = int(np.sum(zb[valid]))
        n_total = int(np.sum(valid))
        print(f"\nZona invertida: zone_bull = (fast < slow) -> {'OK' if check_inv else 'FALLO'}")
        print(f"  Bull bars: {n_bull}/{n_total} ({100.0 * n_bull / n_total:.1f}%)")

    # 3. Diferencias intra-dia forming vs resolved
    valid_both = ~np.isnan(slow_f) & ~np.isnan(slow_r)
    if np.sum(valid_both) > 0:
        diffs = np.abs(slow_f[valid_both] - slow_r[valid_both])
        n_diff = int(np.sum(diffs > 0.01))
        n_valid = int(np.sum(valid_both))
        print(f"\nForming != Resolved (>$0.01): {n_diff}/{n_valid} barras ({100.0 * n_diff / n_valid:.1f}%)")

    # 4. Boundary check: forming[last bar of day N] ~ resolved[first bar of day N+1]
    hours = ts.hour
    day_ends = np.where(hours == 23)[0]
    day_starts = day_ends + 1
    mask = (day_starts < len(slow_f)) & valid_both[day_ends] & valid_both[day_starts]
    if np.sum(mask) > 0:
        forming_at_end = slow_f[day_ends[mask]]
        resolved_at_start = slow_r[day_starts[mask]]
        boundary_diffs = np.abs(forming_at_end - resolved_at_start)
        max_bdiff = float(np.max(boundary_diffs))
        n_match = int(np.sum(boundary_diffs < 0.01))
        n_pairs = int(np.sum(mask))
        print(f"  Boundary forming[23:00] vs resolved[00:00+1]: "
              f"{n_match}/{n_pairs} match (max diff: {max_bdiff:.4f})")

    print("--- Fin verificacion ---\n")


# ============================================
# CLI
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description='Precalculo features mean-reversion (HA diaria / Tenkan)'
    )
    parser.add_argument(
        '--symbols', nargs='+', default=None,
        help='Simbolos a procesar (ej: BTC/USDT ETH/USDT). Default: todos.'
    )
    args = parser.parse_args()

    symbols = args.symbols if args.symbols else SYMBOLS

    print("=== Mean-Reversion Features ===")
    print(f"Simbolos: {len(symbols)}")
    print()

    t_total = time.time()
    ok = 0
    fail = 0

    for symbol in symbols:
        sym = symbol.replace("/", "")
        path = os.path.join(DATA_CACHE_DIR, f"{sym}_1h.parquet")
        if not os.path.exists(path):
            print(f"[{sym}] SKIP - no existe {path}")
            fail += 1
            continue

        df = pd.read_parquet(path)
        # Parquet stores timestamp_ms; convert to datetime column
        if 'timestamp' not in df.columns and 'timestamp_ms' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
        elif not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        data = precalculate_mean_reversion(symbol, df)

        if symbol == 'BTC/USDT':
            _verify_btc(data)

        ok += 1

    elapsed = time.time() - t_total
    print(f"\n=== Completado: {ok} OK, {fail} skip, {elapsed:.1f}s total ===")


if __name__ == '__main__':
    main()
