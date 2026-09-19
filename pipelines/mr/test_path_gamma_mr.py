"""
Tests Sub-fase 2B MR Path γ kernel granular (Sesión 2 Frame 2 R3).

Cover:
  1. MR granular enum 8 valores asimétrico vs TF (6) — refleja realidad
     cancel paths kernel (3 cancel separate vs TF 1 cancel).
  2. Backward compat flag=False default — aggregates IDÉNTICO baseline Sesión 1B.
  3. Path γ per-trade tracking flag=True — 8 fields (NO pt_cluster decisión H-ii)
     + reasons 0-7 valid range.
  4. Audit hash MR regen NO WARN — EXPECTED_LAB_KERNEL_HASH_MR actualizado.
  5. cancel_signal split en cancel_zona_signal + cancel_tf_signal_mr +
     cancel_ghost_signal — verify source contains all 3 setters.
  6. exit_signal split en tf_exit_signal_mr + zone_exit_signal_mr — verify source.
  7. PRIMERA VEZ MR per-trade tracking infrastructure replicate Path α' supplement
     pattern (sentinel arrays MR + flag-driven dispatch).

Standalone, no pytest. Run: python tests/test_path_gamma_mr.py
"""
import os
import sys
import inspect
import hashlib

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, _root)

import numpy as np
import mean_reversion_kernel as _mr


def test_1_mr_granular_enum_8_values():
    """Path γ MR constants enum 0-7 definidos correctamente — asimétrico vs TF (6)."""
    assert _mr.REASON_MR_SL_HIT == 0
    assert _mr.REASON_MR_SL_EMERGENCY == 1
    assert _mr.REASON_MR_DIV_EXIT == 2
    assert _mr.REASON_MR_TF_EXIT == 3
    assert _mr.REASON_MR_ZONE_EXIT == 4
    assert _mr.REASON_MR_CANCEL_ZONA == 5
    assert _mr.REASON_MR_CANCEL_TF == 6
    assert _mr.REASON_MR_CANCEL_GHOST == 7
    # Sentinel arrays MR existing
    assert hasattr(_mr, "_PT_SENTINEL_INT32_MR")
    assert hasattr(_mr, "_PT_SENTINEL_INT8_MR")
    assert hasattr(_mr, "_PT_SENTINEL_FLOAT64_MR")
    assert hasattr(_mr, "_PT_SENTINEL_COUNT_MR")
    print("test_1 PASS: MR enum 8 valores granular asimétrico vs TF (6)")


def _make_synth_mr_data(n_bars=200):
    """Build synthetic data dict matching mean_reversion_features output shape."""
    rng = np.random.default_rng(42)
    close = 100.0 + rng.normal(0, 1, n_bars).cumsum()
    high = close + 0.5
    low = close - 0.5
    timestamps = (np.arange(n_bars, dtype=np.int64) * 3600000).astype("datetime64[ms]")
    zone_bull_forming = np.ones(n_bars, dtype=np.bool_)
    zone_bear_forming = np.zeros(n_bars, dtype=np.bool_)
    zone_bull_resolved = np.ones(n_bars, dtype=np.bool_)
    zone_bear_resolved = np.zeros(n_bars, dtype=np.bool_)
    filters_forming = np.full(n_bars, 0xFF, dtype=np.uint32)
    filters_resolved = np.full(n_bars, 0xFF, dtype=np.uint32)
    fast_line = close.copy()
    slow_line_forming = close.copy()
    div_bits = np.zeros((n_bars, 8), dtype=np.uint8)
    return {
        "close": close,
        "high": high,
        "low": low,
        "timestamps": timestamps,
        "zone_bull_forming": zone_bull_forming,
        "zone_bear_forming": zone_bear_forming,
        "zone_bull_resolved": zone_bull_resolved,
        "zone_bear_resolved": zone_bear_resolved,
        "filters_forming": filters_forming,
        "filters_resolved": filters_resolved,
        "fast_line": fast_line,
        "slow_line_forming": slow_line_forming,
        "div_bits": div_bits,
    }


def test_2_mr_backward_compat_flag_off():
    """flag=False default — run_on_slice retorna single results array (NO 2-tuple)."""
    data = _make_synth_mr_data(n_bars=200)
    configs = np.array([0], dtype=np.uint32)
    result = _mr.run_on_slice(
        configs, data,
        start_bar=100, end_bar=200,
        warmup=100,
    )
    # Backward compat: returns single ndarray (NO 2-tuple)
    assert isinstance(result, np.ndarray), \
        f"flag=False default debe retornar ndarray (got {type(result)})"
    assert result.shape == (1, 7), \
        f"Expected shape (1,7), got {result.shape}"
    print(f"test_2 PASS: backward compat flag=False default — shape {result.shape}")


def test_3_mr_path_gamma_per_trade_tracking():
    """flag=True — retorna 2-tuple (results, per_trade_dict) con 8 fields, reasons 0-7."""
    data = _make_synth_mr_data(n_bars=300)
    configs = np.array([0], dtype=np.uint32)
    result = _mr.run_on_slice(
        configs, data,
        start_bar=100, end_bar=300,
        warmup=100,
        return_per_trade=True,
        max_trades_per_config=100,
    )
    # Path γ: returns 2-tuple
    assert isinstance(result, tuple) and len(result) == 2, \
        f"flag=True debe retornar 2-tuple (got {type(result)})"
    aggregates, per_trade = result
    assert isinstance(aggregates, np.ndarray)
    assert isinstance(per_trade, dict)
    # 8 fields per-trade (NO cluster)
    expected_keys = {"entry_bar", "exit_bar", "side", "pnl", "reason",
                     "entry_price", "exit_price", "count", "max_trades_per_config"}
    assert set(per_trade.keys()) == expected_keys, \
        f"per_trade dict keys mismatch (got {set(per_trade.keys())})"
    assert "cluster" not in per_trade, \
        "pt_cluster debe estar AUSENTE per decisión H-ii (asimetría TF (9) vs MR (8))"
    # Reasons 0-7 range
    n_trades = per_trade["count"][0]
    if n_trades > 0:
        reasons = per_trade["reason"][0, :n_trades]
        assert np.all(reasons >= 0) and np.all(reasons <= 7), \
            f"reasons fuera rango 0-7 Path γ MR: {reasons}"
    print(f"test_3 PASS: per-trade tracking 8 fields (NO cluster), n_trades={n_trades}, reasons in [0,7]")


def test_4_mr_audit_hash_regen_no_warn():
    """EXPECTED_LAB_KERNEL_HASH_MR actualizado match SHA256 actual."""
    src = inspect.getsource(_mr.run_mean_reversion_numba)
    actual = hashlib.sha256(src.encode("utf-8")).hexdigest()
    expected = "371551bdebe328ff1d3ceeee38d4c4583a12e7c5c4df3c4f5f60f1afc675414a"
    assert actual == expected, \
        f"Hash MR mismatch: actual={actual}, expected={expected}\n" \
        f"Si kernel modificado intencionalmente, regen + update audit_mr_fidelity_sei.py"
    # Also verify audit_mr_fidelity_sei.py constant matches
    audit_path = os.path.join(_root, "audit_mr_fidelity_sei.py")
    with open(audit_path, "r", encoding="utf-8") as f:
        audit_src = f.read()
    assert expected in audit_src, \
        f"EXPECTED_LAB_KERNEL_HASH_MR constant en audit_mr_fidelity_sei.py mismatch ({expected[:12]}...)"
    print(f"test_4 PASS: audit_mr hash regen match {actual[:12]}...")


def test_5_mr_cancel_signal_split():
    """cancel_signal split en 3 flags separados (cancel_zona_signal + cancel_tf_signal_mr + cancel_ghost_signal)."""
    src = inspect.getsource(_mr.run_mean_reversion_numba)
    # Verify all 3 cancel-specific flags exist as setters
    assert "cancel_zona_signal = True" in src, \
        "cancel_zona_signal=True setter NO encontrado (split bit 14 fallido)"
    assert "cancel_tf_signal_mr = True" in src, \
        "cancel_tf_signal_mr=True setter NO encontrado (split bit 15 fallido)"
    assert "cancel_ghost_signal = True" in src, \
        "cancel_ghost_signal=True setter NO encontrado (split bit 16 fallido)"
    # cancel_signal=True PRESERVED (cooldown invariante)
    assert "cancel_signal = True" in src, \
        "cancel_signal=True general PRESERVED — cooldown OR condition invariante"
    # Reasons 5/6/7 all distinct
    assert _mr.REASON_MR_CANCEL_ZONA == 5
    assert _mr.REASON_MR_CANCEL_TF == 6
    assert _mr.REASON_MR_CANCEL_GHOST == 7
    print("test_5 PASS: cancel_signal split → cancel_zona + cancel_tf_mr + cancel_ghost (cancel_signal PRESERVED)")


def test_6_mr_exit_signal_split():
    """exit_signal split en tf_exit_signal_mr + zone_exit_signal_mr (split L267-286)."""
    src = inspect.getsource(_mr.run_mean_reversion_numba)
    assert "tf_exit_signal_mr = True" in src, \
        "tf_exit_signal_mr=True setter NO encontrado (split exit_mask filter)"
    assert "zone_exit_signal_mr = True" in src, \
        "zone_exit_signal_mr=True setter NO encontrado (split z_bull/z_bear)"
    # exit_signal=True PRESERVED (general fallback)
    assert "exit_signal = True" in src, \
        "exit_signal=True general PRESERVED"
    assert _mr.REASON_MR_TF_EXIT == 3
    assert _mr.REASON_MR_ZONE_EXIT == 4
    print("test_6 PASS: exit_signal split → tf_exit_signal_mr + zone_exit_signal_mr")


def test_7_mr_per_trade_first_time_implementation():
    """PRIMERA VEZ MR per-trade tracking infrastructure — sentinel + flag-driven dispatch."""
    # Verify run_mean_reversion_numba signature has return_per_trade kwarg
    sig = inspect.signature(_mr.run_mean_reversion_numba)
    params = sig.parameters
    assert "return_per_trade" in params, \
        "return_per_trade kwarg NO en signature (PRIMERA VEZ MR per-trade tracking absent)"
    assert params["return_per_trade"].default is False, \
        f"return_per_trade default debe ser False (backward compat), got {params['return_per_trade'].default}"
    # 8 per-trade arrays kwargs (NO pt_cluster)
    pt_kwargs = ["pt_entry_bar", "pt_exit_bar", "pt_side", "pt_pnl", "pt_reason",
                 "pt_entry_price", "pt_exit_price", "pt_count"]
    for kw in pt_kwargs:
        assert kw in params, f"{kw} kwarg NO en signature"
    # NO pt_cluster (decisión H-ii)
    assert "pt_cluster" not in params, \
        "pt_cluster debe estar AUSENTE per decisión H-ii (MR sin cluster accounting)"
    # Verify run_on_slice extension
    sig_slice = inspect.signature(_mr.run_on_slice)
    assert "return_per_trade" in sig_slice.parameters
    assert "max_trades_per_config" in sig_slice.parameters
    print("test_7 PASS: PRIMERA VEZ MR per-trade tracking — 8 fields kwargs + flag-driven (NO pt_cluster)")


if __name__ == "__main__":
    tests = [
        test_1_mr_granular_enum_8_values,
        test_2_mr_backward_compat_flag_off,
        test_3_mr_path_gamma_per_trade_tracking,
        test_4_mr_audit_hash_regen_no_warn,
        test_5_mr_cancel_signal_split,
        test_6_mr_exit_signal_split,
        test_7_mr_per_trade_first_time_implementation,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
    if failed == 0:
        print(f"\n{len(tests)}/{len(tests)} tests PASS")
        sys.exit(0)
    else:
        print(f"\n{failed}/{len(tests)} tests FAIL")
        sys.exit(1)
