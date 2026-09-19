# -*- coding: utf-8 -*-
"""
test_import_smokes.py — los 18 smokes de importación del toolkit como suite
pytest (los mismos verificados a mano en las Sesiones 2-3). Corre en CI sin
red ni datos: solo importa. `audit_mr_fidelity_sei` queda fuera a propósito
(ejecuta su auditoría al importar y requiere datos + la réplica live/ que no
se distribuye — documentado en pipelines/mr/KNOWN_DIVERGENCES.md §3).
"""
import importlib

import pytest

MODULES = [
    'master', 'lab_historico_numba_v8_3', 'lab_cuda', 'lab_lite_zonas_v5e',
    'regime_walk_forward', 'train_regime_model', 'regime_features',
    'download_full_history', 'data_cache', 'extractor_gemas',
    'placebo_gen', 'leakage_gate',
    'mean_reversion_kernel', 'mean_reversion_features',
    'mean_reversion_walk_forward',
    'toy_features', 'toy_kernel', 'toy_pipeline',
]


@pytest.mark.parametrize('name', MODULES)
def test_import(name):
    mod = importlib.import_module(name)
    assert mod is not None
