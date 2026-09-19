# -*- coding: utf-8 -*-
"""
toy_kernel.py — Kernel TOY: cruce de dos EMAs, stop-and-reverse, 2 parámetros.

Réplica pedagógica EXACTA del contrato del motor (INVENTARIO_TOOLKIT.md §2.2):
  - Universo de configs como enteros bit-packed + decode_toy_config
    (patrón decode_config, lab_historico_numba_v8_3.py:1207; y
    decode_mean_reversion_config, mean_reversion_kernel.py:717).
  - generate_toy_configs() filtra el universo a combos VÁLIDAS
    (patrón generate_valid_configs, lab_historico_numba_v8_3.py:1153).
  - Kernel @jit(nopython=True, parallel=True) que itera configs con prange
    y devuelve la matriz n_configs × 7 con las MISMAS 7 métricas del motor:
    [pnl, trades, wins, cancels, max_dd, gross_profit, gross_loss]
    (patrón run_simulation_numba :1302 / run_mean_reversion_numba :81).
  - Wrapper run_on_slice que gestiona warmup/accounting_start
    (patrón lab :1940 / MR :767).

Estrategia: long en cruce alcista EMA_fast×EMA_slow, short en cruce bajista
(stop-and-reverse). Sin stops (didáctico); comisión 0.10% round-trip idéntica
al motor (COMMISSION_ROUND_TRIP, lab_historico_numba_v8_3.py:423-431).

Extensiones Sesión 3 (código toy propio, el motor NO se toca):
  - hyst_frac: histéresis del cruce (eje de "preset" de la toy, espejo del
    bucle preset × histéresis [0.0, 0.5] del motor, regime_walk_forward.py:610-616).
    Con hyst_frac=0.0 el comportamiento es EXACTAMENTE el de la Sesión 2.
  - accounting_end: ventana de accounting acotada [start, end) para el
    walk-forward por anclajes (toy_walk_forward.py).
  - run_ema_cross_pairs_numba: mismo simulador sobre pares (fila_fast,
    fila_slow) explícitos — lo usa la escalera de tamaño de búsqueda del
    placebo (universo denso ~1000 configs, examples/toy/placebo_verdict.py).

Layout de bits del config (6 bits, universo bruto 64, válidas 61):
  bits 0-2: índice período rápido (TOY_FAST_PERIODS, 8 valores)
  bits 3-5: índice período lento  (TOY_SLOW_PERIODS, 8 valores)
  Validez: fast_period < slow_period (se filtra en generate_toy_configs).
"""
import numpy as np
from numba import jit, prange

from toy_features import (TOY_FAST_PERIODS, TOY_SLOW_PERIODS, N_FAST,
                          WARMUP_BARS, precalculate_toy_features)

COMMISSION_ROUND_TRIP = 0.10   # % por trade (entrada+salida), igual que el motor
N_METRICS = 7                  # [pnl, trades, wins, cancels, max_dd, gp, gl]


def generate_toy_configs():
    """Universo VÁLIDO de config_ids (fast_period < slow_period)."""
    configs = []
    for fi in range(len(TOY_FAST_PERIODS)):
        for si in range(len(TOY_SLOW_PERIODS)):
            if TOY_FAST_PERIODS[fi] < TOY_SLOW_PERIODS[si]:
                configs.append(fi | (si << 3))
    return np.array(configs, dtype=np.int64)


def decode_toy_config(config_id):
    """Decodifica un config_id toy a parámetros humanos (patrón decode_config)."""
    fi = config_id & 0x7
    si = (config_id >> 3) & 0x7
    return {
        'config_id': int(config_id),
        'fast_idx': int(fi),
        'slow_idx': int(si),
        'fast_period': int(TOY_FAST_PERIODS[fi]),
        'slow_period': int(TOY_SLOW_PERIODS[si]),
    }


@jit(nopython=True, cache=True)
def _simulate_rows(f_row, s_row, close_arr, ema_all, commission_pct,
                   accounting_start, accounting_end, hyst_frac, out, c):
    """Núcleo común: simula UN par (fila_fast, fila_slow) en [start, end).

    Con hyst_frac=0.0 la condición de cruce es idéntica a la Sesión 2.
    Con h>0: el cruce solo dispara si la rápida rebasa la banda ±h de la lenta
    (preset laxo/medio/duro del walk-forward toy).
    """
    position = 0        # 0 flat, +1 long, -1 short
    entry_price = 0.0
    pnl_acc = 0.0
    peak = 0.0
    max_dd = 0.0
    trades = 0.0
    wins = 0.0
    gp = 0.0
    gl = 0.0

    t0 = accounting_start
    if t0 < 1:
        t0 = 1
    for t in range(t0, accounting_end):
        ef_prev = ema_all[f_row, t - 1]
        es_prev = ema_all[s_row, t - 1]
        ef = ema_all[f_row, t]
        es = ema_all[s_row, t]
        up_band = es * (1.0 + hyst_frac)
        dn_band = es * (1.0 - hyst_frac)
        up_band_prev = es_prev * (1.0 + hyst_frac)
        dn_band_prev = es_prev * (1.0 - hyst_frac)
        cross_up = (ef_prev <= up_band_prev) and (ef > up_band)
        cross_dn = (ef_prev >= dn_band_prev) and (ef < dn_band)

        if cross_up or cross_dn:
            px = close_arr[t]
            if position != 0:
                trade_pnl = position * (px / entry_price - 1.0) * 100.0 - commission_pct
                pnl_acc += trade_pnl
                trades += 1.0
                if trade_pnl > 0.0:
                    wins += 1.0
                    gp += trade_pnl
                else:
                    gl += -trade_pnl
                if pnl_acc > peak:
                    peak = pnl_acc
                dd = peak - pnl_acc
                if dd > max_dd:
                    max_dd = dd
            position = 1 if cross_up else -1
            entry_price = px

    # Cierre contable de la posición abierta en la última barra de la ventana
    if position != 0:
        px = close_arr[accounting_end - 1]
        trade_pnl = position * (px / entry_price - 1.0) * 100.0 - commission_pct
        pnl_acc += trade_pnl
        trades += 1.0
        if trade_pnl > 0.0:
            wins += 1.0
            gp += trade_pnl
        else:
            gl += -trade_pnl
        if pnl_acc > peak:
            peak = pnl_acc
        dd = peak - pnl_acc
        if dd > max_dd:
            max_dd = dd

    out[c, 0] = pnl_acc
    out[c, 1] = trades
    out[c, 2] = wins
    out[c, 3] = 0.0          # cancels: la toy no tiene cancelaciones
    out[c, 4] = max_dd
    out[c, 5] = gp
    out[c, 6] = gl


@jit(nopython=True, parallel=True, cache=True)
def run_toy_simulation_numba(configs, close_arr, ema_all, commission_pct,
                             accounting_start, accounting_end, hyst_frac):
    """Kernel toy bit-packed. Devuelve results (n_configs, 7) como el motor."""
    n_configs = len(configs)
    results = np.zeros((n_configs, N_METRICS), dtype=np.float64)
    for c in prange(n_configs):
        cfg = configs[c]
        f_row = cfg & 0x7
        s_row = N_FAST + ((cfg >> 3) & 0x7)
        _simulate_rows(f_row, s_row, close_arr, ema_all, commission_pct,
                       accounting_start, accounting_end, hyst_frac, results, c)
    return results


@jit(nopython=True, parallel=True, cache=True)
def run_ema_cross_pairs_numba(pairs, close_arr, ema_all, commission_pct,
                              accounting_start, accounting_end, hyst_frac):
    """Mismo simulador sobre pares explícitos (fila_fast, fila_slow) en ema_all.

    pairs: array (n_configs, 2) int64 de índices de fila. Permite universos
    arbitrarios (p. ej. el denso ~1000 configs de la escalera del placebo)
    sin tocar el bit-packing canónico de 6 bits.
    """
    n_configs = pairs.shape[0]
    results = np.zeros((n_configs, N_METRICS), dtype=np.float64)
    for c in prange(n_configs):
        _simulate_rows(pairs[c, 0], pairs[c, 1], close_arr, ema_all,
                       commission_pct, accounting_start, accounting_end,
                       hyst_frac, results, c)
    return results


def run_on_slice(configs, df, accounting_start=None, accounting_end=None,
                 commission_pct=COMMISSION_ROUND_TRIP, hyst_frac=0.0):
    """Wrapper de despacho (patrón run_on_slice del motor): warmup + kernel.

    df: DataFrame formato caché de producción (timestamp_ms, open, high, low,
    close, volume). Ventana de accounting [accounting_start, accounting_end)
    — por defecto [WARMUP_BARS, n). Devuelve (results, meta).
    """
    close = np.ascontiguousarray(df['close'].values, dtype=np.float64)
    n = len(close)
    if accounting_start is None:
        accounting_start = WARMUP_BARS
    if accounting_end is None:
        accounting_end = n
    if not (0 < accounting_start < accounting_end <= n):
        raise ValueError(f"Ventana inválida: [{accounting_start}, {accounting_end}) sobre {n} barras")
    if n <= WARMUP_BARS + 10:
        raise ValueError(f"Serie demasiado corta: {n} barras <= warmup {WARMUP_BARS}+10")

    ema_all = precalculate_toy_features(close)
    results = run_toy_simulation_numba(
        np.asarray(configs, dtype=np.int64), close, ema_all,
        float(commission_pct), int(accounting_start), int(accounting_end),
        float(hyst_frac))
    meta = {
        'n_bars': int(n),
        'accounting_start': int(accounting_start),
        'accounting_end': int(accounting_end),
        'n_configs': int(len(configs)),
        'commission_pct': float(commission_pct),
        'hyst_frac': float(hyst_frac),
    }
    return results, meta
