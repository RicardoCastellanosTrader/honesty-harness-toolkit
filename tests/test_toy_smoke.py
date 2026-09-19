# -*- coding: utf-8 -*-
"""
test_toy_smoke.py — smoke de la tubería toy sobre sintético (< 30 s).

Cubre: contrato n_configs×7, determinismo, equivalencia kernel bit-packed ↔
kernel de pares (base de la escalera del placebo), efecto de la histéresis
(presets), y estructura del walk-forward toy.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'engine'))
sys.path.insert(0, os.path.join(_ROOT, 'pipelines', 'toy'))

import numpy as np
import pandas as pd
import pytest

from placebo_gen import gbm, make_ohlc
from toy_features import (TOY_FAST_PERIODS, TOY_SLOW_PERIODS, N_FAST,
                          precalculate_toy_features, _ema)
from toy_kernel import (generate_toy_configs, decode_toy_config, run_on_slice,
                        run_toy_simulation_numba, run_ema_cross_pairs_numba,
                        COMMISSION_ROUND_TRIP, N_METRICS)
import toy_walk_forward as twf

N_BARS = 3000
SEED = 424242


@pytest.fixture(scope="module")
def df_gbm():
    rng = np.random.default_rng(SEED)
    close = gbm(0.011, 50000.0, N_BARS, rng)
    return make_ohlc(close, 0.006, 1000.0, rng)


def test_universe_and_decode():
    configs = generate_toy_configs()
    assert len(configs) == 61
    for cfg in configs:
        d = decode_toy_config(int(cfg))
        assert d['fast_period'] < d['slow_period']


def test_contract_shape_and_determinism(df_gbm):
    configs = generate_toy_configs()
    r1, meta = run_on_slice(configs, df_gbm)
    r2, _ = run_on_slice(configs, df_gbm)
    assert r1.shape == (61, N_METRICS)
    assert np.all(np.isfinite(r1))
    assert np.array_equal(r1, r2)                      # determinismo bit a bit
    assert (r1[:, 1] > 0).any()                        # alguna config tradea
    # pnl == gp - gl (coherencia de las 7 métricas del contrato)
    np.testing.assert_allclose(r1[:, 0], r1[:, 5] - r1[:, 6], atol=1e-9)
    assert (r1[:, 3] == 0).all()                       # cancels siempre 0 en toy


def test_pairs_kernel_equivalent_to_bitpacked(df_gbm):
    """La escalera del placebo usa el kernel de pares; debe ser bit-idéntico
    al kernel canónico sobre el MISMO universo de 61."""
    configs = generate_toy_configs()
    close = np.ascontiguousarray(df_gbm['close'].values, dtype=np.float64)
    ema_canon = precalculate_toy_features(close)
    r_bit = run_toy_simulation_numba(configs, close, ema_canon,
                                     COMMISSION_ROUND_TRIP, 250, len(close), 0.0)
    # matriz de pares con ordenación de períodos propia (como placebo_verdict)
    periods = sorted(set(TOY_FAST_PERIODS.tolist()) | set(TOY_SLOW_PERIODS.tolist()))
    row = {p: i for i, p in enumerate(periods)}
    m = np.empty((len(periods), len(close)))
    for i, p in enumerate(periods):
        m[i, :] = _ema(close, p)
    pairs = np.array([[row[int(TOY_FAST_PERIODS[c & 0x7])],
                       row[int(TOY_SLOW_PERIODS[(c >> 3) & 0x7])]]
                      for c in configs], dtype=np.int64)
    r_pairs = run_ema_cross_pairs_numba(pairs, close, m,
                                        COMMISSION_ROUND_TRIP, 250, len(close), 0.0)
    np.testing.assert_array_equal(r_bit, r_pairs)


def test_hysteresis_reduces_trading(df_gbm):
    """Preset duro (h=0.005) debe generar en agregado menos trades que laxo."""
    configs = generate_toy_configs()
    r_lax, _ = run_on_slice(configs, df_gbm, hyst_frac=0.0)
    r_dur, _ = run_on_slice(configs, df_gbm, hyst_frac=0.005)
    assert r_dur[:, 1].sum() < r_lax[:, 1].sum()


def test_walk_forward_structure(df_gbm):
    df_anchors, df_agg = twf.walk_forward(df_gbm, opt_size=1000, fwd_size=400)
    n_anchors = df_anchors['anchor'].nunique()
    assert n_anchors >= 3
    winners = df_anchors[df_anchors['preset'].str.contains('ganador')]
    assert len(winners) == n_anchors                   # un ganador por anclaje
    # el ganador IS de cada anclaje es un máximo: su is_pnl >= cualquier fila
    for a, grp in df_anchors.groupby('anchor'):
        w = grp[grp['preset'].str.contains('ganador')]['is_pnl'].iloc[0]
        assert w >= grp[~grp['preset'].str.contains('ganador')]['is_pnl'].max() - 1e-9
    assert set(df_agg['seleccion']) >= {'laxo', 'medio', 'duro', 'GANADOR/anclaje'}
    assert (df_anchors['robustez_is'].dropna() <= 1.0).all()
    assert (df_anchors['robustez_is'].dropna() >= 0.0).all()
