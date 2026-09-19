"""
Audit Fidelidad 2 MR — SEI/USDT C2 config_id=45686.
Compara brain_engine._evaluate_bar_mr vs mean_reversion_kernel.run_on_slice
sobre el mismo slice de SEI histórico.

Uso: python audit_mr_fidelity_sei.py
"""
import os
import sys
import time
import hashlib
import inspect
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "live"))

SYMBOL = "SEI/USDT"
CONFIG_ID = 45686
CLUSTER_TEST = 2
N_TEST = 1500            # últimas 1500 barras (como live)
WARMUP = 500             # warmup brain
KERNEL_WARMUP = 100

# ============================================================================
# Hash check Path γ MR — Sesión 2 Frame 2 (decisión Ricardo D-i 2026-04-29).
# Patrón análogo audit_fidelity_v5.py + audit_fidelity_v5_2.py: SHA256 fija
# kernel run_mean_reversion_numba. Si lab kernel cambia sin actualizar hash,
# WARN early antes de invertir compute audit completo.
# Complementario al runtime parity check (no redundante): static hash detecta
# kernel modify, runtime parity detecta drift comportamental real.
# Para actualizar hash tras cambio intencional: recomputar con
# `python -c "import hashlib,inspect; import mean_reversion_kernel as _mr;
#  print(hashlib.sha256(inspect.getsource(_mr.run_mean_reversion_numba).encode('utf-8')).hexdigest())"`
# y actualizar constante.
# ============================================================================
EXPECTED_LAB_KERNEL_HASH_MR = "371551bdebe328ff1d3ceeee38d4c4583a12e7c5c4df3c4f5f60f1afc675414a"  # Sesión 2 Frame 2 R3 Path γ MR granular enum 8 valores 2026-04-29


def _verify_mr_kernel_hash():
    """SHA256 hash check run_mean_reversion_numba — WARN si mismatch."""
    try:
        from mean_reversion_kernel import run_mean_reversion_numba
        src = inspect.getsource(run_mean_reversion_numba)
        actual = hashlib.sha256(src.encode("utf-8")).hexdigest()
        if actual == EXPECTED_LAB_KERNEL_HASH_MR:
            print(f"[KERNEL_PARITY] mr_kernel.run_mean_reversion_numba: {actual[:12]}... OK")
            return True
        else:
            print(f"[KERNEL_PARITY][WARN] Expected {EXPECTED_LAB_KERNEL_HASH_MR[:12]}... "
                  f"got {actual[:12]}...")
            print(f"[KERNEL_PARITY][WARN] Revisar cambio vs mean_reversion_kernel y actualizar "
                  f"EXPECTED_LAB_KERNEL_HASH_MR.")
            return False
    except Exception as e:
        print(f"[KERNEL_PARITY][ERROR] {e}")
        return False


print(f"=== Audit Fidelidad 2 MR — {SYMBOL} C{CLUSTER_TEST} config {CONFIG_ID} ===")
_verify_mr_kernel_hash()
t0 = time.time()

# --- Load data ---
sym_clean = SYMBOL.replace("/", "").replace("USDT", "USDT")
parquet_path = os.path.join(ROOT, "data_cache", f"{sym_clean}_1h.parquet")
df_full = pd.read_parquet(parquet_path)
# Parquet uses 'timestamp_ms' (int64 unix ms). Add datetime 'timestamp' column expected by pipeline.
if "timestamp" not in df_full.columns:
    df_full["timestamp"] = pd.to_datetime(df_full["timestamp_ms"], unit="ms", utc=True)
print(f"Total bars available: {len(df_full)}")

n_slice = min(N_TEST, len(df_full))
df_test = df_full.tail(n_slice).reset_index(drop=True)
print(f"Using last {n_slice} bars from {df_test['timestamp'].iloc[0]} to {df_test['timestamp'].iloc[-1]}")

# --- PART 1: Kernel MR via run_on_slice ---
print(f"\n[1/2] Kernel MR (run_mean_reversion_numba)...")
tk = time.time()
from mean_reversion_features import precalculate_mean_reversion
from mean_reversion_kernel import run_on_slice

data = precalculate_mean_reversion(SYMBOL, df_test)
print(f"  Precalc: {time.time()-tk:.1f}s")

tk2 = time.time()
configs = np.array([CONFIG_ID], dtype=np.uint32)
results = run_on_slice(
    configs, data,
    start_bar=WARMUP, end_bar=n_slice,
    warmup=KERNEL_WARMUP,
)
print(f"  Kernel run: {time.time()-tk2:.1f}s")

kernel_pnl = results[0, 0]
kernel_trades = int(results[0, 1])
kernel_wins = int(results[0, 2])
kernel_cancels = int(results[0, 3])
kernel_maxdd = results[0, 4]
kernel_gp = results[0, 5]
kernel_gl = results[0, 6]
print(f"  Kernel: PnL={kernel_pnl:.4f}% trades={kernel_trades} wins={kernel_wins} "
      f"cancels={kernel_cancels} maxDD={kernel_maxdd:.2f}%")

# --- PART 2: Brain engine MR bar-by-bar ---
print(f"\n[2/2] Brain engine MR (_evaluate_bar_mr bar-by-bar)...")
from brain_engine import (
    load_models, _evaluate_bar_mr, decode_mr_config_bits, COMMISSION_ROUND_TRIP,
)

brain = load_models(
    regime_models_dir=os.path.join(ROOT, "regime_models"),
    specialist_configs_dir=os.path.join(ROOT, "regime_wf"),
    symbols=[SYMBOL],
)
state = brain.get_state(SYMBOL)
state.current_cluster = CLUSTER_TEST

mr_bits = decode_mr_config_bits(CONFIG_ID)
cfg = {
    "config_id": CONFIG_ID,
    "mr_bits": mr_bits,
    "cluster": CLUSTER_TEST,
    "strategy_type": "mean_reversion",
    "operable": True,
    "regime_changed": False,
}
# Initialize cluster_probs
n_clusters = brain.specialist_configs[SYMBOL].get("n_clusters", 3)
state.cluster_probs = np.ones(n_clusters) / n_clusters
state.cluster_probs[CLUSTER_TEST] = 0.9

tb = time.time()
signals_brain = []
for i in range(WARMUP, n_slice):
    # Build window with full history up to bar i (inclusive)
    window = df_test.iloc[:i+1].reset_index(drop=True)
    try:
        sig = _evaluate_bar_mr(brain, SYMBOL, window, cfg)
        if sig["action"] in ("LONG", "SHORT", "CLOSE_LONG", "CLOSE_SHORT"):
            signals_brain.append({
                "bar": i,
                "action": sig["action"],
                "price": sig["entry_price"],
                "reason": sig["reason"],
                "ts": df_test["timestamp"].iloc[i],
            })
    except Exception as e:
        print(f"    Bar {i} error: {type(e).__name__}: {e}")

print(f"  Brain bar-by-bar: {time.time()-tb:.1f}s")
print(f"  Signals captured: {len(signals_brain)}")

# Compute brain trades (pair LONG/SHORT with matching CLOSE_*)
brain_pnl_pct = 0.0
brain_trades = 0
brain_wins = 0
brain_gp = 0.0
brain_gl = 0.0
brain_cancels = 0
brain_maxdd = 0.0
brain_peak = 0.0
cur_pos = 0
entry_price = 0.0
for s in signals_brain:
    if s["action"] == "LONG" and cur_pos == 0:
        cur_pos = 1; entry_price = s["price"]
    elif s["action"] == "SHORT" and cur_pos == 0:
        cur_pos = -1; entry_price = s["price"]
    elif s["action"] in ("CLOSE_LONG", "CLOSE_SHORT") and cur_pos != 0:
        exit_price = s["price"]
        if cur_pos == 1:
            pnl_pct = (exit_price - entry_price) / entry_price * 100.0
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100.0
        pnl_net = pnl_pct - COMMISSION_ROUND_TRIP
        brain_pnl_pct += pnl_net
        brain_trades += 1
        if pnl_net > 0:
            brain_wins += 1
            brain_gp += pnl_net
        else:
            brain_gl += abs(pnl_net)
        if s["reason"] and s["reason"].startswith("cancel"):
            brain_cancels += 1
        if brain_pnl_pct > brain_peak:
            brain_peak = brain_pnl_pct
        dd = brain_peak - brain_pnl_pct
        if dd > brain_maxdd:
            brain_maxdd = dd
        cur_pos = 0; entry_price = 0.0

print(f"  Brain: PnL={brain_pnl_pct:.4f}% trades={brain_trades} wins={brain_wins} "
      f"cancels={brain_cancels} maxDD={brain_maxdd:.2f}%")

# --- Comparison ---
print("\n=== COMPARISON ===")
print(f"{'Metric':<12} {'Kernel':>12} {'Brain':>12} {'Diff':>12}")
print("-" * 52)
for name, k, b in [
    ("PnL %",      kernel_pnl,     brain_pnl_pct),
    ("Trades",     kernel_trades,  brain_trades),
    ("Wins",       kernel_wins,    brain_wins),
    ("Cancels",    kernel_cancels, brain_cancels),
    ("MaxDD %",    kernel_maxdd,   brain_maxdd),
    ("GrossProfit", kernel_gp,     brain_gp),
    ("GrossLoss",  kernel_gl,      brain_gl),
]:
    diff = b - k
    print(f"{name:<12} {k:>12.4f} {b:>12.4f} {diff:>+12.4f}")

pnl_ok = abs(brain_pnl_pct - kernel_pnl) < 0.01
trades_ok = brain_trades == kernel_trades
cancels_ok = brain_cancels == kernel_cancels

print(f"\nPnL match (tol 0.01%): {'YES' if pnl_ok else 'NO'}")
print(f"Trades match exact:     {'YES' if trades_ok else 'NO'}")
print(f"Cancels match exact:    {'YES' if cancels_ok else 'NO'}")

if pnl_ok and trades_ok and cancels_ok:
    print("\n=== FIDELIDAD CONFIRMADA ===")
else:
    print("\n=== DIVERGENCIA DETECTADA — investigar ===")
    # List all signals
    print("\nBrain signals:")
    for s in signals_brain[:40]:
        print(f"  Bar {s['bar']}: {s['action']} @ {s['price']:.4f} ({s['reason']}) ts={s['ts']}")
    if len(signals_brain) > 40:
        print(f"  ...+{len(signals_brain)-40} more")

print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
