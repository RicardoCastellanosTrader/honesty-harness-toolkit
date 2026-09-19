#!/usr/bin/env python
"""
mean_reversion_walk_forward.py -- Regime Walk-Forward para Mean-Reversion

Evalua la estrategia mean-reversion por cluster/regimen y, si es rentable en
clusters no operables por trend-following, inyecta en specialist_configs.json.

Uso:
    python mean_reversion_walk_forward.py                           # Todos
    python mean_reversion_walk_forward.py --symbols BTC/USDT        # Solo uno
    python mean_reversion_walk_forward.py --dry-run                 # Solo reporta
"""

import os
import sys
import json
import time
import shutil
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

from master import CONFIG
from regime_walk_forward import (
    load_full_history,
    load_regime_model,
    compute_cluster_labels,
    compute_gmm_probs,
    identify_episodes,
    build_regime_labels,
    _approx_sqn_vec,
    sym_key,
    sym_clean,
    MIN_EPISODES_VALIDATE,
)
from mean_reversion_kernel import (
    run_mean_reversion_numba,
    generate_mean_reversion_configs,
    decode_mean_reversion_config,
    SL_PERCENT, SL_EMERGENCY_PERCENT, TS_PERCENT,
    COOLDOWN_BARS, COMMISSION_ROUND_TRIP,
    DEFAULT_DIV_IND_MASK,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYMBOLS = CONFIG['symbols']
DATA_CACHE_DIR = CONFIG['data_cache_dir']
MODEL_DIR = 'regime_models'
WF_DIR = 'regime_wf'

TRAIN_RATIO = 0.70
MIN_EPISODE_BARS = 50
WARMUP = 100

# Filtering thresholds
MIN_PF_TRAIN = 1.2
MIN_PF_FORWARD = 1.0
MIN_TRADES_TRAIN = 20
MIN_TRADES_FORWARD = 8
MAX_DD_PCT = 25.0
TOP_TRAIN = 100       # top configs by PnL in train to evaluate in forward


# ============================================
# SQN (scalar version for single config)
# ============================================

def _sqn_scalar(pnl, gp, gl, wins, trades):
    """SQN approximation for a single config (scalar)."""
    if trades < 2:
        return 0.0
    losses = trades - wins
    mean_trade = pnl / trades
    avg_win = gp / wins if wins > 0 else 0.0
    avg_loss = gl / losses if losses > 0 else 0.0
    p = wins / trades
    variance = p * (avg_win - mean_trade)**2 + (1 - p) * (-avg_loss - mean_trade)**2
    if variance <= 0:
        return 0.0
    std_trade = np.sqrt(variance)
    return (mean_trade / std_trade) * np.sqrt(trades)


# ============================================
# SPECIALIST SCORE (simplified, no pf_robustness)
# ============================================

def _specialist_score(pf_combined, trades_total, sqn):
    """
    specialist_score = pf_combined * log(1 + trades/50) * max(sqn/3.0, 0.5)
    """
    trade_factor = np.log(1.0 + trades_total / 50.0)
    sqn_factor = max(sqn / 3.0, 0.5)
    return pf_combined * trade_factor * sqn_factor


# ============================================
# RUN PER-CLUSTER SIMULATION
# ============================================

def _run_cluster_simulation(configs, data, episodes, warmup=WARMUP):
    """
    Run MR kernel on a list of episodes, accumulating results per config.

    Returns: dict with arrays (n_configs,) for pnl, trades, wins, cancels,
    max_dd, gross_profit, gross_loss.
    """
    n_configs = len(configs)
    acc_pnl = np.zeros(n_configs, dtype=np.float64)
    acc_trades = np.zeros(n_configs, dtype=np.float64)
    acc_wins = np.zeros(n_configs, dtype=np.float64)
    acc_cancels = np.zeros(n_configs, dtype=np.float64)
    acc_maxdd = np.zeros(n_configs, dtype=np.float64)
    acc_gp = np.zeros(n_configs, dtype=np.float64)
    acc_gl = np.zeros(n_configs, dtype=np.float64)

    n_data = len(data['close'])
    ts_full = data['timestamps']

    for ep_start, ep_end in episodes:
        actual_start = max(0, ep_start - warmup)
        accounting_start = ep_start - actual_start
        s, e = actual_start, min(ep_end, n_data)

        if e - s < 10:
            continue

        ts_raw = ts_full[s:e]
        ts_i64 = ts_raw.astype('datetime64[ms]').astype(np.int64)

        results = run_mean_reversion_numba(
            configs,
            np.ascontiguousarray(data['close'][s:e]),
            np.ascontiguousarray(data['high'][s:e]),
            np.ascontiguousarray(data['low'][s:e]),
            ts_i64,
            np.ascontiguousarray(data['zone_bull_forming'][s:e]),
            np.ascontiguousarray(data['zone_bear_forming'][s:e]),
            np.ascontiguousarray(data['zone_bull_resolved'][s:e]),
            np.ascontiguousarray(data['zone_bear_resolved'][s:e]),
            np.ascontiguousarray(data['filters_forming'][s:e].astype(np.uint32)),
            np.ascontiguousarray(data['filters_resolved'][s:e].astype(np.uint32)),
            np.ascontiguousarray(data['fast_line'][s:e]),
            np.ascontiguousarray(data['slow_line_forming'][s:e]),
            np.ascontiguousarray(data['div_bits'][s:e]),
            DEFAULT_DIV_IND_MASK,
            SL_PERCENT, SL_EMERGENCY_PERCENT, TS_PERCENT,
            COOLDOWN_BARS, COMMISSION_ROUND_TRIP,
            accounting_start
        )

        acc_pnl += results[:, 0]
        acc_trades += results[:, 1]
        acc_wins += results[:, 2]
        acc_cancels += results[:, 3]
        # Max DD: take worst across episodes
        acc_maxdd = np.maximum(acc_maxdd, results[:, 4])
        acc_gp += results[:, 5]
        acc_gl += results[:, 6]

    return {
        'pnl': acc_pnl,
        'trades': acc_trades,
        'wins': acc_wins,
        'cancels': acc_cancels,
        'max_dd': acc_maxdd,
        'gross_profit': acc_gp,
        'gross_loss': acc_gl,
    }


# ============================================
# PROCESS ONE SYMBOL
# ============================================

def process_symbol(symbol, mr_data, model_data, df, dry_run=False):
    """
    Run mean-reversion walk-forward for one symbol.

    Returns: dict with per-cluster results and injection decisions.
    """
    sc = sym_clean(symbol)
    sk = sym_key(symbol)
    n_bars = len(df)

    # --- Classify bars ---
    cluster_labels, n_clusters = compute_cluster_labels(df, model_data)
    gmm_probs = compute_gmm_probs(df, model_data)

    # --- Extract episodes ---
    episodes = identify_episodes(cluster_labels, n_clusters, min_bars=MIN_EPISODE_BARS)

    # --- Build train/forward split ---
    regime_labels, n_doubled, split_info = build_regime_labels(
        cluster_labels, episodes, n_clusters,
        train_ratio=TRAIN_RATIO,
        toxic_tail=0,      # dynamic toxic tail
        gmm_probs=gmm_probs,
        confirm_threshold=0.75,
        max_toxic_tail=100,
        min_toxic_tail=5,
    )

    # --- Generate configs ---
    configs = generate_mean_reversion_configs()
    n_configs = len(configs)

    print(f"  [{sc}] {n_bars} bars, {n_clusters} clusters, {n_configs} configs")

    # --- Load existing JSON for comparison ---
    existing_json = _load_existing_json(symbol)
    existing_clusters = existing_json.get('clusters', {}) if existing_json else {}

    # --- Per-cluster walk-forward ---
    cluster_results = {}

    for k in range(n_clusters):
        info = split_info.get(k, {})

        if not info.get('valid', False):
            reason = info.get('reason', 'unknown')
            print(f"    Cluster {k}: SKIP ({reason})")
            cluster_results[k] = {'viable': False, 'reason': reason}
            continue

        # Extract train and forward episodes
        all_eps = episodes.get(k, [])
        n_train = info['train_episodes']
        train_eps = all_eps[:n_train]
        fwd_eps = all_eps[n_train:]

        if not train_eps or not fwd_eps:
            print(f"    Cluster {k}: SKIP (no train/fwd episodes)")
            cluster_results[k] = {'viable': False, 'reason': 'no episodes'}
            continue

        # --- Run on train episodes ---
        train_res = _run_cluster_simulation(configs, mr_data, train_eps)
        train_pnl = train_res['pnl']
        train_trades = train_res['trades']
        train_gp = train_res['gross_profit']
        train_gl = train_res['gross_loss']
        train_pf = np.where(train_gl > 0, train_gp / train_gl,
                            np.where(train_gp > 0, 99.9, 0.0))

        # --- Train filter ---
        train_pass = (
            (train_trades >= MIN_TRADES_TRAIN) &
            (train_pnl > 0) &
            (train_pf >= MIN_PF_TRAIN)
        )
        n_train_pass = int(np.sum(train_pass))

        if n_train_pass == 0:
            print(f"    Cluster {k}: 0 pass train filter "
                  f"(trades>={MIN_TRADES_TRAIN}, PnL>0, PF>={MIN_PF_TRAIN})")
            cluster_results[k] = {'viable': False, 'reason': 'no train pass'}
            continue

        # --- Top N by train PnL ---
        train_pass_idx = np.where(train_pass)[0]
        top_by_pnl = train_pass_idx[np.argsort(-train_pnl[train_pass_idx])[:TOP_TRAIN]]

        # --- Run top configs on forward episodes ---
        top_configs = configs[top_by_pnl]
        fwd_res = _run_cluster_simulation(top_configs, mr_data, fwd_eps)
        fwd_pnl = fwd_res['pnl']
        fwd_trades = fwd_res['trades']
        fwd_wins = fwd_res['wins']
        fwd_cancels = fwd_res['cancels']
        fwd_maxdd = fwd_res['max_dd']
        fwd_gp = fwd_res['gross_profit']
        fwd_gl = fwd_res['gross_loss']
        fwd_pf = np.where(fwd_gl > 0, fwd_gp / fwd_gl,
                          np.where(fwd_gp > 0, 99.9, 0.0))

        # --- Forward filter ---
        fwd_pass = (
            (fwd_trades >= MIN_TRADES_FORWARD) &
            (fwd_pnl > 0) &
            (fwd_pf >= MIN_PF_FORWARD) &
            (fwd_maxdd <= MAX_DD_PCT)
        )
        n_fwd_pass = int(np.sum(fwd_pass))

        if n_fwd_pass == 0:
            print(f"    Cluster {k}: {n_train_pass} train pass, 0 fwd pass")
            cluster_results[k] = {
                'viable': False,
                'reason': 'no forward pass',
                'n_train_pass': n_train_pass,
            }
            continue

        # --- Compute combined metrics for validated configs ---
        fwd_pass_local_idx = np.where(fwd_pass)[0]

        best_score = -1e9
        best_result = None

        for li in fwd_pass_local_idx:
            gi = top_by_pnl[li]  # global config index

            # Train metrics for this config
            tr_pnl = train_pnl[gi]
            tr_trades = int(train_trades[gi])
            tr_wins = int(train_res['wins'][gi])
            tr_gp = train_gp[gi]
            tr_gl = train_gl[gi]
            tr_pf = train_pf[gi]
            tr_maxdd = train_res['max_dd'][gi]

            # Forward metrics
            fw_pnl = fwd_pnl[li]
            fw_trades = int(fwd_trades[li])
            fw_wins = int(fwd_wins[li])
            fw_gp = fwd_gp[li]
            fw_gl = fwd_gl[li]
            fw_pf = fwd_pf[li]
            fw_maxdd = fwd_maxdd[li]
            fw_cancels = int(fwd_cancels[li])

            # Combined
            total_gp = tr_gp + fw_gp
            total_gl = tr_gl + fw_gl
            pf_combined = total_gp / total_gl if total_gl > 0 else (99.9 if total_gp > 0 else 0.0)
            trades_total = tr_trades + fw_trades
            maxdd_worst = max(tr_maxdd, fw_maxdd)

            # SQN on forward
            sqn_fwd = _sqn_scalar(fw_pnl, fw_gp, fw_gl, fw_wins, fw_trades)

            # Score
            score = _specialist_score(pf_combined, trades_total, sqn_fwd)

            if score > best_score:
                best_score = score
                best_result = {
                    'config_id': int(configs[gi]),
                    'config_decoded': decode_mean_reversion_config(int(configs[gi])),
                    'pf_train': round(float(tr_pf), 3),
                    'pf_forward': round(float(fw_pf), 3),
                    'pf_combined': round(float(pf_combined), 3),
                    'trades_train': tr_trades,
                    'trades_forward': fw_trades,
                    'pnl_train': round(float(tr_pnl), 2),
                    'pnl_forward': round(float(fw_pnl), 2),
                    'sqn_forward': round(float(sqn_fwd), 3),
                    'specialist_score': round(float(score), 3),
                    'max_dd': round(float(maxdd_worst), 2),
                    'cancels_forward': fw_cancels,
                }

        print(f"    Cluster {k}: {n_train_pass} train pass, {n_fwd_pass} fwd pass, "
              f"best score={best_result['specialist_score']:.2f} "
              f"PF={best_result['pf_combined']:.2f} "
              f"trades={best_result['trades_train']}+{best_result['trades_forward']}")

        # --- Check if TF cluster is operable ---
        tf_operable = _is_tf_cluster_operable(existing_clusters, k)

        cluster_results[k] = {
            'viable': True,
            'n_train_pass': n_train_pass,
            'n_fwd_pass': n_fwd_pass,
            'best': best_result,
            'tf_operable': tf_operable,
            'inject': not tf_operable,  # only inject if TF is NOT operable
        }

    return {
        'symbol': symbol,
        'n_clusters': n_clusters,
        'cluster_results': cluster_results,
        'existing_json': existing_json,
    }


# ============================================
# HELPERS
# ============================================

def _load_existing_json(symbol):
    """Load existing specialist_configs.json for comparison."""
    sc = sym_clean(symbol)
    # Check both comboclaude and combolab paths
    for base in [WF_DIR, os.path.join('..', 'combolab', 'regime_wf')]:
        path = os.path.join(base, f"{sc}_specialist_configs.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None


def _is_tf_cluster_operable(existing_clusters, cluster_id):
    """Check if a cluster has viable trend-following configs.

    A cluster is operable only if at least one config has sqn_p5 > 0.
    Clusters with configs but all sqn_p5 <= 0 are effectively non-operable.
    """
    cluster_data = existing_clusters.get(str(cluster_id), {})
    top_configs = cluster_data.get('top_configs', [])
    if not top_configs:
        return False
    for cfg in top_configs:
        if cfg.get('sqn_p5', 0) > 0:
            return True
    return False


def _find_json_path(symbol):
    """Find the correct path for the specialist_configs.json."""
    sc = sym_clean(symbol)
    for base in [WF_DIR, os.path.join('..', 'combolab', 'regime_wf')]:
        path = os.path.join(base, f"{sc}_specialist_configs.json")
        if os.path.exists(path):
            return path
    # Default to WF_DIR if nothing found
    os.makedirs(WF_DIR, exist_ok=True)
    return os.path.join(WF_DIR, f"{sc}_specialist_configs.json")


def _inject_into_json(symbol, cluster_id, mr_result, dry_run=False):
    """
    Inject mean-reversion specialist into existing JSON.
    Backs up original before modifying.
    """
    json_path = _find_json_path(symbol)
    if not os.path.exists(json_path):
        print(f"      WARNING: No JSON found for {symbol}, skipping injection")
        return False

    with open(json_path) as f:
        data = json.load(f)

    cluster_key = str(cluster_id)
    cluster_data = data.get('clusters', {}).get(cluster_key, {})

    # Build the MR entry
    best = mr_result['best']
    decoded = best['config_decoded']

    mr_entry = {
        'strategy_type': 'mean_reversion',
        'config_id': best['config_id'],
        'config_decoded': {
            'entry_tfs': "+".join(decoded['entry_tfs']) if decoded['entry_tfs'] else "-",
            'exit_tfs': "+".join(decoded['exit_tfs']),
            'div_entry_mode': decoded['div_entry_mode'],
            'div_exit': decoded['div_exit'],
            'div_type': decoded['div_type'],
            'cancel_zona': decoded['cancel_zona'],
            'cancel_tf': decoded['cancel_tf'],
            'cancel_ghost': decoded['cancel_ghost'],
        },
        'pf_train': best['pf_train'],
        'pf_forward': best['pf_forward'],
        'pf_combined': best['pf_combined'],
        'trades_train': best['trades_train'],
        'trades_forward': best['trades_forward'],
        'sqn_forward': best['sqn_forward'],
        'specialist_score': best['specialist_score'],
        'max_dd': best['max_dd'],
    }

    if dry_run:
        print(f"      DRY-RUN: Would inject MR into cluster {cluster_key}")
        print(f"        config_id={best['config_id']}, "
              f"PF={best['pf_combined']:.2f}, "
              f"score={best['specialist_score']:.2f}")
        return True

    # Backup original
    backup_path = json_path + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(json_path, backup_path)

    # Inject
    if 'clusters' not in data:
        data['clusters'] = {}
    if cluster_key not in data['clusters']:
        data['clusters'][cluster_key] = {
            'name': f'cluster_{cluster_id}',
            'n_validated': 0,
            'survival_rate_pct': 0.0,
            'top_configs': [],
        }

    data['clusters'][cluster_key]['mean_reversion'] = mr_entry

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"      INJECTED MR into {json_path} cluster {cluster_key}")
    return True


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description='Mean-Reversion Regime Walk-Forward'
    )
    parser.add_argument('--symbols', nargs='+', default=None,
                        help='Symbols to process (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report only, do not modify JSONs')
    parser.add_argument('--div-ind-mask', type=int, default=DEFAULT_DIV_IND_MASK,
                        help='Divergence indicator mask (default: 1 = RSI)')
    args = parser.parse_args()

    symbols = args.symbols if args.symbols else SYMBOLS

    print("=== Mean-Reversion Walk-Forward ===")
    print(f"Symbols: {len(symbols)}")
    print(f"Dry-run: {args.dry_run}")
    print()

    t_total = time.time()

    # Counters for summary
    total_clusters = 0
    mr_viable = 0
    mr_injected = 0
    tf_operable_kept = 0
    non_operable = 0

    # JIT warmup: generate configs once
    configs = generate_mean_reversion_configs()
    print(f"MR configs: {len(configs)}")
    print()

    for symbol in symbols:
        sc = sym_clean(symbol)
        sk = sym_key(symbol)
        t0 = time.time()
        print(f"[{sc}] Processing...")

        # --- Load MR data ---
        npz_path = os.path.join(DATA_CACHE_DIR, f"{sc}_mean_reversion.npz")
        if not os.path.exists(npz_path):
            print(f"  SKIP: no MR data ({npz_path})")
            continue
        mr_data = dict(np.load(npz_path, allow_pickle=True))

        # --- Load regime model ---
        model_data = load_regime_model(symbol)
        if model_data is None:
            print(f"  SKIP: no regime model")
            continue

        # --- Load OHLCV for cluster classification ---
        df = load_full_history(symbol)
        if df is None:
            print(f"  SKIP: no OHLCV data")
            continue

        # Ensure MR data and OHLCV have compatible lengths
        n_mr = len(mr_data['close'])
        n_df = len(df)
        if n_mr != n_df:
            # Trim to shorter
            n = min(n_mr, n_df)
            df = df.iloc[:n].reset_index(drop=True)
            for key in mr_data:
                if hasattr(mr_data[key], '__len__') and len(mr_data[key]) > n:
                    mr_data[key] = mr_data[key][:n]

        # --- Process symbol ---
        result = process_symbol(symbol, mr_data, model_data, df, dry_run=args.dry_run)

        # --- Report and inject ---
        n_clusters = result['n_clusters']

        for k in range(n_clusters):
            total_clusters += 1
            cr = result['cluster_results'].get(k, {})

            if not cr.get('viable', False):
                non_operable += 1
                continue

            mr_viable += 1
            tf_op = cr.get('tf_operable', True)
            inject = cr.get('inject', False)

            if tf_op:
                tf_operable_kept += 1
                best = cr['best']
                print(f"    Cluster {k}: TF operable (kept). "
                      f"MR alternative: PF={best['pf_combined']:.2f}, "
                      f"score={best['specialist_score']:.2f}")
            elif inject:
                best = cr['best']
                success = _inject_into_json(symbol, k, cr, dry_run=args.dry_run)
                if success:
                    mr_injected += 1
                    print(f"    Cluster {k}: TF NOT operable -> MR INJECTED. "
                          f"PF={best['pf_combined']:.2f}, "
                          f"score={best['specialist_score']:.2f}")

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")
        print()

    # --- Summary ---
    total_elapsed = time.time() - t_total
    print("=" * 60)
    print("=== SUMMARY ===")
    print(f"Total clusters analyzed: {total_clusters}")
    print(f"MR viable (pass train+fwd): {mr_viable}")
    print(f"  TF operable (kept TF): {tf_operable_kept}")
    print(f"  MR injected (rescued): {mr_injected}")
    print(f"Non-operable (neither): {non_operable}")
    print(f"Total time: {total_elapsed:.1f}s")
    if args.dry_run:
        print("\n** DRY-RUN: No JSONs were modified **")
    print("=" * 60)


if __name__ == '__main__':
    main()
