# -*- coding: utf-8 -*-
"""
test_leakage_gate.py — envoltorio pytest del self-test bidireccional del gate
anti-leakage canónico del estudio (engine/leakage_gate.py:82 _selftest):
el caso honesto PASA y el caso envenenado FALLA ruidoso. Si cualquiera de las
dos direcciones se rompe, el gate no protege nada.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'engine'))

import leakage_gate


def test_selftest_bidireccional():
    assert leakage_gate._selftest() == 0


def test_assert_asof_catches_poisoned_parquet(tmp_path):
    """Réplica mínima independiente: parquet con 1 barra >= ancla → leak cazado."""
    import pandas as pd
    A = pd.Timestamp("2026-02-01 00:00:00", tz="UTC")
    ts_bad = [int((A + pd.Timedelta(hours=1)).timestamp() * 1000)]
    p = tmp_path / "bad.parquet"
    pd.DataFrame({"timestamp_ms": ts_bad, "open": 1.0, "high": 1.0,
                  "low": 1.0, "close": 1.0, "volume": 1.0}).to_parquet(p)
    ok, errs = leakage_gate.assert_asof(A, parquet_path=str(p), label="test")
    assert not ok and any("L1 LEAK" in e for e in errs)
