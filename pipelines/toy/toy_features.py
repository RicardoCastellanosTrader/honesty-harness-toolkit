# -*- coding: utf-8 -*-
"""
toy_features.py — Precálculo de features de la tubería TOY (cruce de dos EMAs).

Contrato del motor (INVENTARIO_TOOLKIT.md §2.2 / §2.4): el kernel NUNCA calcula
indicadores; recibe arrays numpy precalculados por-bar. Aquí la "feature" es una
matriz de EMAs: una fila por período candidato (rápidos + lentos).

Equivalente en el motor real: lab_lite_zonas_v5e / precalculate_all_data
(lab_historico_numba_v8_3.py:999) producen las zonas/filtros que consume el
kernel TF; mean_reversion_features.py produce el .npz que consume el kernel MR.
"""
import numpy as np
from numba import njit

# Universo de períodos (el "preset" implícito de la toy — fijo, como en MR).
TOY_FAST_PERIODS = np.array([5, 8, 12, 16, 21, 26, 34, 42], dtype=np.int64)
TOY_SLOW_PERIODS = np.array([30, 40, 55, 75, 100, 140, 190, 250], dtype=np.int64)
N_FAST = len(TOY_FAST_PERIODS)
N_SLOW = len(TOY_SLOW_PERIODS)
# Filas de la matriz de features: [0..N_FAST) = EMAs rápidas, [N_FAST..N_FAST+N_SLOW) = lentas.
WARMUP_BARS = int(TOY_SLOW_PERIODS.max())  # 250: el período más lento domina el warmup


@njit(cache=True)
def _ema(close, period):
    """EMA clásica alpha=2/(period+1), sembrada con el primer close."""
    n = len(close)
    out = np.empty(n, dtype=np.float64)
    alpha = 2.0 / (period + 1.0)
    out[0] = close[0]
    for t in range(1, n):
        out[t] = alpha * close[t] + (1.0 - alpha) * out[t - 1]
    return out


def precalculate_toy_features(close):
    """Devuelve la matriz de EMAs (N_FAST+N_SLOW, n_bars) float64.

    Análogo toy de precalculate_all_data: todo lo path-dependent se computa
    UNA vez aquí; el kernel solo indexa filas según el config decodificado.
    """
    close = np.ascontiguousarray(close, dtype=np.float64)
    n = len(close)
    ema_all = np.empty((N_FAST + N_SLOW, n), dtype=np.float64)
    for i in range(N_FAST):
        ema_all[i, :] = _ema(close, TOY_FAST_PERIODS[i])
    for j in range(N_SLOW):
        ema_all[N_FAST + j, :] = _ema(close, TOY_SLOW_PERIODS[j])
    return ema_all
