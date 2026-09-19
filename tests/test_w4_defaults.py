# -*- coding: utf-8 -*-
"""
test_w4_defaults.py — T4 Sesión 2: demuestra que los defaults del toolkit
(_FWD_MIN_TRADES=15, _FWD_MIN_PF=1.0) reproducen el comportamiento EFECTIVO
de master.py en runtime en el estudio original.

Contexto (fuente primaria, repo del estudio):
  - regime_walk_forward.py:1093-1094 definía 25/1.1 (W4).
  - master.py:93-94 fijaba CONFIG fwd_min_trades=15 / fwd_min_pf=1.0 y
    master.py:446-447 SOBREESCRIBÍA los atributos del módulo en runtime:
        rwf._FWD_MIN_TRADES = CONFIG['fwd_min_trades']
        rwf._FWD_MIN_PF = CONFIG['fwd_min_pf']
  → los valores efectivos de todo el estudio fueron 15/1.0.

El toolkit hace ese valor efectivo el DEFAULT del módulo (única modificación
de código del motor autorizada) y añade fwd_min_trades/fwd_min_pf como kwargs
explícitos de extract_validated_specialists.

Ejecución: python -m pytest tests/test_w4_defaults.py  (desde la raíz del toolkit)
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'engine'))

import regime_walk_forward as rwf

# Valores que master.py fijaba en runtime (master.py:93-94 del estudio)
MASTER_RUNTIME_FWD_MIN_TRADES = 15
MASTER_RUNTIME_FWD_MIN_PF = 1.0


def _fwd_mask(df, min_trades, min_pf):
    """Réplica literal de la máscara forward de extract_validated_specialists
    (misma expresión que regime_walk_forward.py, bloque Phase 2)."""
    return (
        (df['trades_fwd'] >= min_trades) &
        (df['pnl_fwd'] > 0) &
        (df['pf_fwd'] >= min_pf)
    )


@pytest.fixture
def candidates():
    """Casos borde alrededor de ambos juegos de umbrales (15/1.0 y 25/1.1)."""
    return pd.DataFrame({
        'trades_fwd': [14, 15, 16, 24, 25, 26, 100, 15, 25],
        'pnl_fwd':    [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, -1.0, 5.0, 5.0],
        'pf_fwd':     [1.5, 1.0, 0.99, 1.09, 1.10, 1.11, 2.0, 1.05, 1.05],
    })


def test_module_defaults_are_study_effective_values():
    """El default del módulo toolkit == valor efectivo del estudio (15/1.0)."""
    assert rwf._FWD_MIN_TRADES == MASTER_RUNTIME_FWD_MIN_TRADES
    assert rwf._FWD_MIN_PF == MASTER_RUNTIME_FWD_MIN_PF


def test_master_runtime_override_is_noop_now():
    """Emula el override de master.py:446-447 sobre el módulo toolkit y
    verifica que es un no-op: los defaults ya SON los valores de master."""
    before = (rwf._FWD_MIN_TRADES, rwf._FWD_MIN_PF)
    rwf._FWD_MIN_TRADES = MASTER_RUNTIME_FWD_MIN_TRADES   # master.py:446
    rwf._FWD_MIN_PF = MASTER_RUNTIME_FWD_MIN_PF           # master.py:447
    assert (rwf._FWD_MIN_TRADES, rwf._FWD_MIN_PF) == before


def test_fwd_mask_defaults_equal_master_runtime(candidates):
    """La máscara forward con los defaults del módulo == la máscara con los
    valores que master inyectaba en runtime (igualdad de comportamiento)."""
    mask_toolkit = _fwd_mask(candidates, rwf._FWD_MIN_TRADES, rwf._FWD_MIN_PF)
    mask_master = _fwd_mask(candidates, MASTER_RUNTIME_FWD_MIN_TRADES,
                            MASTER_RUNTIME_FWD_MIN_PF)
    assert mask_toolkit.equals(mask_master)
    # y difiere de los W4 estrictos del módulo original (25/1.1):
    mask_w4_strict = _fwd_mask(candidates, 25, 1.1)
    assert not mask_toolkit.equals(mask_w4_strict), \
        "el parámetro debe ser load-bearing: 15/1.0 != 25/1.1 en los casos borde"
    # el caso borde exacto 15 trades / pf 1.0 / pnl>0 pasa con el default:
    assert bool(mask_toolkit.iloc[1]) is True
    assert bool(mask_w4_strict.iloc[1]) is False


def test_extract_accepts_explicit_kwargs(tmp_path):
    """La firma nueva acepta fwd_min_trades/fwd_min_pf explícitos.
    Se usa el early-return de 'sin clusters válidos' (no toca parquets)."""
    sym_result = {
        'symbol': 'TOY/USDT',
        'n_clusters': 1,
        'cluster_names': ['c0'],
        'split_info': [{'valid': False}],
        'parts_dir': str(tmp_path),
    }
    out = rwf.extract_validated_specialists(
        sym_result, str(tmp_path), fwd_min_trades=25, fwd_min_pf=1.1)
    assert out is None  # early-return documentado (sin clusters válidos)
    # y con defaults (None → módulo):
    out2 = rwf.extract_validated_specialists(sym_result, str(tmp_path))
    assert out2 is None
