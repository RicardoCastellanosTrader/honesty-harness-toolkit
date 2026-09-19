#!/usr/bin/env python3
"""
regime_features.py - Módulo compartido para cálculo de features de régimen.

Usado por:
  - train_regime_model.py (entrenamiento offline)
  - lab_lite_zonas_v5e.py (aplicación de modelos pre-entrenados)

Features ESTACIONARIAS (por barra, ventana rolling):
  1. Hurst exponent (R/S method) — carácter de persistencia [0-1]
  2. Z_ATR — z-score de volatilidad (ATR corto vs largo) [~N(0,1)]
  3. ER — Kaufman Efficiency Ratio [0-1]
  4. (opcional) Z_Vol — z-score de volumen relativo [~N(0,1)]
"""

import numpy as np
from numba import jit

# Configuración
LOOKBACK_SHORT = 100     # ventana para Hurst, ER, y media corta de ATR
LOOKBACK_LONG = 1000     # ventana para media/std larga del Z-Score
ATR_PERIOD = 14          # período del ATR base
USE_4TH_FEATURE = False  # activar Z-Score de volumen relativo

N_FEATURES = 4 if USE_4TH_FEATURE else 3
FEATURE_NAMES = ['Hurst', 'Z_ATR', 'ER', 'Z_Vol'] if USE_4TH_FEATURE else ['Hurst', 'Z_ATR', 'ER']


@jit(nopython=True)
def _compute_regime_features_numba(close, high, low, volume,
                                   atr, lookback_short, lookback_long,
                                   n_features):
    """Numba-compiled core: computes stationary regime features per bar.

    Features:
      0: Hurst exponent (R/S) — [0.01, 0.99]
      1: Z_ATR — z-score of ATR(14) short vs long — ~N(0,1)
      2: ER — Kaufman Efficiency Ratio — [0, 1]
      3: (optional) Z_Vol — z-score of volume short vs long — ~N(0,1)

    Bars are valid only from index (lookback_long - 1) onwards.
    """
    n = len(close)
    features = np.full((n, n_features), np.nan, dtype=np.float64)

    # We need lookback_long bars of history for Z-scores
    start_bar = lookback_long - 1

    for i in range(start_bar, n):
        # ---- 1. Hurst exponent (R/S) over lookback_short ----
        w_start = i - lookback_short + 1
        w_end = i + 1
        lb = lookback_short

        if lb < 20:
            features[i, 0] = 0.5
        else:
            n_ret = lb - 1
            returns = np.empty(n_ret, dtype=np.float64)
            for j in range(n_ret):
                c_prev = close[w_start + j]
                c_curr = close[w_start + j + 1]
                if c_prev < 1e-10:
                    c_prev = 1e-10
                if c_curr < 1e-10:
                    c_curr = 1e-10
                returns[j] = np.log(c_curr) - np.log(c_prev)

            mean_r = 0.0
            for j in range(n_ret):
                mean_r += returns[j]
            mean_r /= n_ret

            var_r = 0.0
            for j in range(n_ret):
                d = returns[j] - mean_r
                var_r += d * d

            if n_ret > 1:
                std_r = np.sqrt(var_r / (n_ret - 1))
            else:
                std_r = 0.0

            if std_r < 1e-12:
                features[i, 0] = 0.5
            else:
                cum = 0.0
                cum_max = -1e300
                cum_min = 1e300
                for j in range(n_ret):
                    cum += returns[j] - mean_r
                    if cum > cum_max:
                        cum_max = cum
                    if cum < cum_min:
                        cum_min = cum
                R = cum_max - cum_min
                if R < 1e-12:
                    features[i, 0] = 0.5
                else:
                    RS = R / std_r
                    H = np.log(RS) / np.log(lb)
                    if H < 0.01:
                        H = 0.01
                    elif H > 0.99:
                        H = 0.99
                    features[i, 0] = H

        # ---- 2. Z_ATR: z-score of short ATR mean vs long ATR mean/std ----
        # Short mean: mean of ATR over last lookback_short bars
        atr_short_sum = 0.0
        atr_short_count = 0
        for j in range(i - lookback_short + 1, i + 1):
            a_j = atr[j]
            if a_j == a_j:  # not NaN
                atr_short_sum += a_j
                atr_short_count += 1

        # Long mean and std: over last lookback_long bars
        atr_long_sum = 0.0
        atr_long_sq_sum = 0.0
        atr_long_count = 0
        for j in range(i - lookback_long + 1, i + 1):
            a_j = atr[j]
            if a_j == a_j:  # not NaN
                atr_long_sum += a_j
                atr_long_sq_sum += a_j * a_j
                atr_long_count += 1

        if atr_short_count > lookback_short // 2 and atr_long_count > lookback_long // 2:
            atr_short_mean = atr_short_sum / atr_short_count
            atr_long_mean = atr_long_sum / atr_long_count
            atr_long_var = atr_long_sq_sum / atr_long_count - atr_long_mean * atr_long_mean
            if atr_long_var < 1e-20:
                features[i, 1] = 0.0  # mercado muerto
            else:
                atr_long_std = np.sqrt(atr_long_var)
                features[i, 1] = (atr_short_mean - atr_long_mean) / atr_long_std
        # else stays NaN

        # ---- 3. ER: Kaufman Efficiency Ratio over lookback_short ----
        # direction = |close[i] - close[i - lookback_short]|
        # volatility = sum(|close[j] - close[j-1]|) for j in window
        direction = abs(close[i] - close[w_start])
        volatility = 0.0
        for j in range(w_start + 1, w_end):
            volatility += abs(close[j] - close[j - 1])
        if volatility > 1e-12:
            er = direction / volatility
            if er > 1.0:
                er = 1.0
            features[i, 2] = er
        else:
            features[i, 2] = 0.0

        # ---- 4. (optional) Z_Vol: z-score of short volume vs long volume ----
        if n_features >= 4:
            vol_short_sum = 0.0
            vol_short_count = 0
            for j in range(i - lookback_short + 1, i + 1):
                vol_short_sum += volume[j]
                vol_short_count += 1

            vol_long_sum = 0.0
            vol_long_sq_sum = 0.0
            vol_long_count = 0
            for j in range(i - lookback_long + 1, i + 1):
                vol_long_sum += volume[j]
                vol_long_sq_sum += volume[j] * volume[j]
                vol_long_count += 1

            if vol_short_count > 0 and vol_long_count > lookback_long // 2:
                vol_short_mean = vol_short_sum / vol_short_count
                vol_long_mean = vol_long_sum / vol_long_count
                vol_long_var = vol_long_sq_sum / vol_long_count - vol_long_mean * vol_long_mean
                if vol_long_var < 1e-20:
                    features[i, 3] = 0.0
                else:
                    vol_long_std = np.sqrt(vol_long_var)
                    features[i, 3] = (vol_short_mean - vol_long_mean) / vol_long_std
            # else stays NaN

    return features


def compute_regime_features(df, lookback=100):
    """Compute stationary regime features for each bar.

    Features (per bar, rolling windows):
    1. Hurst exponent (R/S method) — persistence character [0-1]
    2. Z_ATR — z-score of volatility (short vs long ATR) [~N(0,1)]
    3. ER — Kaufman Efficiency Ratio [0-1]
    4. (optional) Z_Vol — z-score of relative volume [~N(0,1)]

    The `lookback` parameter controls the short window (Hurst, ER, ATR short mean).
    The long window for z-scores is LOOKBACK_LONG (1000 bars).
    Bars need at least LOOKBACK_LONG bars of history to be valid.

    Returns: features array (n_bars, N_FEATURES), valid_mask (n_bars,)
    """
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)
    n = len(close)

    # Precompute ATR(14)
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[high[0] - low[0]], tr])
    atr = np.full(n, np.nan)
    atr[ATR_PERIOD - 1] = np.mean(tr[:ATR_PERIOD])
    for i in range(ATR_PERIOD, n):
        atr[i] = (atr[i-1] * (ATR_PERIOD - 1) + tr[i]) / ATR_PERIOD

    # Use the lookback parameter as short window, LOOKBACK_LONG as long window
    lookback_short = lookback
    lookback_long = LOOKBACK_LONG

    features = _compute_regime_features_numba(close, high, low, volume,
                                              atr, lookback_short, lookback_long,
                                              N_FEATURES)

    valid_mask = ~np.any(np.isnan(features), axis=1)
    return features, valid_mask
