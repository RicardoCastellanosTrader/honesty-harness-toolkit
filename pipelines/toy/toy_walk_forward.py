# -*- coding: utf-8 -*-
"""
toy_walk_forward.py — Walk-forward de la tubería TOY, SIN regímenes.

Qué enseña: el "recitador de respuestas". En cada anclaje se elige la config
que mejor recitó el pasado (ventana de optimización) y se le pregunta por el
futuro (ventana forward). La tabla agregada retrata la caída IS→FWD.

Diseño (decisiones del director, Sesión 3):
  - SIN clustering de regímenes: anclajes rodantes con ventana de optimización
    y ventana forward, proporción opt/fwd = 5000/2000 del experimento real
    multi-anclaje (fuente primaria: walk_forward_experiment.py:56-58 del
    estudio, CLI :16). El nº de anclajes escala con el tamaño de los datos.
  - 3 presets de histéresis del cruce: laxo (h=0), medio (h=0.002),
    duro (h=0.005) — espejo didáctico del bucle preset × histéresis del motor
    (regime_walk_forward.py:610-616).
  - Índice de robustez por configuración y anclaje: fracción de VECINOS
    paramétricos (±1 paso en período rápido y lento) con PnL IS > 0 —
    versión mínima del filtro de mesetas del estudio (extractor_gemas.py:229
    `_evaluate_plateau`, gate ≥60% :333-392) y del haircut por vecindario
    (regime_walk_forward.py:1879 `_compute_sqn_haircut`).

PUNTERO AL MOTOR REAL: la versión completa de esta idea es
engine/regime_walk_forward.py — split 70/30 por EPISODIOS de régimen GMM
(TRAIN_RATIO regime_walk_forward.py:201, build_regime_labels :335), bootstrap
de CI sobre pf_fwd (W3, :1130), filtros W4 (:1248) y ranking por
pf_fwd_ci_low. La toy es el mismo esqueleto sin regímenes ni CI.

Anti-lookahead: las EMAs se calculan una sola vez sobre la serie completa —
EMA[t] depende SOLO de barras ≤ t, así que ninguna ventana ve el futuro; la
selección IS usa exclusivamente [opt_start, opt_end).

Uso (desde la raíz del toolkit):
  python pipelines/toy/toy_walk_forward.py --data examples/data/TOYGBM1USDT_1h.parquet
  python pipelines/toy/toy_walk_forward.py --symbol BTC/USDT
  (opcional: --opt-size 5000 --fwd-size 2000 --out prefix)
"""
import os
import sys
import time
import argparse

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith('cp'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd

from toy_features import (TOY_FAST_PERIODS, TOY_SLOW_PERIODS, WARMUP_BARS)
from toy_kernel import (generate_toy_configs, decode_toy_config, run_on_slice)

# Presets de histéresis (eje "preset" de la toy — decisión director Sesión 3)
PRESETS = [("laxo", 0.0), ("medio", 0.002), ("duro", 0.005)]

# Proporción opt/fwd del experimento multi-anclaje real
# (walk_forward_experiment.py:56-58: opt_size=5000, fwd_size=2000)
DEFAULT_OPT = 5000
DEFAULT_FWD = 2000


def _neighbors_idx(configs, cfg):
    """Vecinos paramétricos de cfg: ±1 paso en fast_idx y en slow_idx,
    restringidos al universo válido. Devuelve índices dentro de `configs`."""
    fi = cfg & 0x7
    si = (cfg >> 3) & 0x7
    cand = []
    for dfi, dsi in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nfi, nsi = fi + dfi, si + dsi
        if 0 <= nfi < len(TOY_FAST_PERIODS) and 0 <= nsi < len(TOY_SLOW_PERIODS):
            ncfg = nfi | (nsi << 3)
            cand.append(ncfg)
    lookup = {int(c): i for i, c in enumerate(configs)}
    return [lookup[c] for c in cand if c in lookup]


def compute_anchor_points(n_bars, opt_size, fwd_size, warmup):
    """Anclajes rodantes: opt [s, s+O) + fwd [s+O, s+O+F), paso F.
    (Versión sin ext_size de compute_anchor_points,
    walk_forward_experiment.py:253 del estudio.)"""
    anchors = []
    s = warmup
    while s + opt_size + fwd_size <= n_bars:
        anchors.append({
            'opt_start': s, 'opt_end': s + opt_size,
            'fwd_start': s + opt_size, 'fwd_end': s + opt_size + fwd_size,
        })
        s += fwd_size
    return anchors


def _pf(gp, gl):
    return gp / gl if gl > 0 else (float('inf') if gp > 0 else 0.0)


def walk_forward(df, opt_size=DEFAULT_OPT, fwd_size=DEFAULT_FWD):
    """Corre el walk-forward toy. Devuelve (df_anchors, df_aggregate)."""
    configs = generate_toy_configs()
    n = len(df)
    anchors = compute_anchor_points(n, opt_size, fwd_size, WARMUP_BARS)
    if not anchors:
        raise ValueError(
            f"Serie demasiado corta para opt={opt_size}+fwd={fwd_size} "
            f"con warmup={WARMUP_BARS}: {n} barras")

    rows = []
    for a_i, a in enumerate(anchors):
        best_row = None
        for pname, h in PRESETS:
            res_is, _ = run_on_slice(configs, df,
                                     accounting_start=a['opt_start'],
                                     accounting_end=a['opt_end'], hyst_frac=h)
            res_fw, _ = run_on_slice(configs, df,
                                     accounting_start=a['fwd_start'],
                                     accounting_end=a['fwd_end'], hyst_frac=h)
            k = int(np.argmax(res_is[:, 0]))          # el recitador: max PnL IS
            cfg = int(configs[k])
            d = decode_toy_config(cfg)
            nb = _neighbors_idx(configs, cfg)
            robustez = (float(np.mean(res_is[nb, 0] > 0.0)) if nb else float('nan'))
            row = {
                'anchor': a_i,
                'opt': f"[{a['opt_start']},{a['opt_end']})",
                'fwd': f"[{a['fwd_start']},{a['fwd_end']})",
                'preset': pname, 'hyst': h,
                'config_id': cfg, 'fast': d['fast_period'], 'slow': d['slow_period'],
                'is_pnl': round(res_is[k, 0], 2),
                'is_pf': round(_pf(res_is[k, 5], res_is[k, 6]), 3),
                'is_trades': int(res_is[k, 1]),
                'fwd_pnl': round(res_fw[k, 0], 2),
                'fwd_pf': round(_pf(res_fw[k, 5], res_fw[k, 6]), 3),
                'fwd_trades': int(res_fw[k, 1]),
                'robustez_is': round(robustez, 2),
                'n_vecinos': len(nb),
            }
            rows.append(row)
            if best_row is None or row['is_pnl'] > best_row['is_pnl']:
                best_row = dict(row, preset=f"*{pname}")
        rows.append(dict(best_row, preset=best_row['preset'] + ' (ganador anclaje)'))

    df_anchors = pd.DataFrame(rows)

    # Agregado: por preset y para el ganador global por anclaje
    agg_rows = []
    for label, sel in ([(p, df_anchors['preset'] == p) for p, _ in PRESETS] +
                       [('GANADOR/anclaje', df_anchors['preset'].str.startswith('*'))]):
        sub = df_anchors[sel]
        if len(sub) == 0:
            continue
        agg_rows.append({
            'seleccion': label,
            'n_anclajes': len(sub),
            'is_pnl_medio': round(sub['is_pnl'].mean(), 2),
            'is_pnl_mediana': round(sub['is_pnl'].median(), 2),
            'fwd_pnl_medio': round(sub['fwd_pnl'].mean(), 2),
            'fwd_pnl_mediana': round(sub['fwd_pnl'].median(), 2),
            'pct_fwd_positivo': round(100.0 * (sub['fwd_pnl'] > 0).mean(), 1),
            'robustez_media': round(sub['robustez_is'].mean(), 2),
        })
    df_agg = pd.DataFrame(agg_rows)
    return df_anchors, df_agg


def main():
    ap = argparse.ArgumentParser(description="Walk-forward toy sin regímenes")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--data', help="Parquet formato caché")
    src.add_argument('--symbol', help="Símbolo en data_cache/, ej. BTC/USDT")
    ap.add_argument('--opt-size', type=int, default=DEFAULT_OPT)
    ap.add_argument('--fwd-size', type=int, default=DEFAULT_FWD)
    ap.add_argument('--out', default=None, help="Prefijo de los CSVs de salida")
    args = ap.parse_args()

    path = args.data or os.path.join('data_cache', f"{args.symbol.replace('/','')}_1h.parquet")
    if not os.path.exists(path):
        sys.exit(f"ERROR: no existe {path}")
    df = pd.read_parquet(path)
    tag = os.path.splitext(os.path.basename(path))[0]

    t0 = time.perf_counter()
    df_anchors, df_agg = walk_forward(df, args.opt_size, args.fwd_size)
    elapsed = time.perf_counter() - t0

    prefix = args.out or f"toy_wf_{tag}"
    df_anchors.to_csv(f"{prefix}_anclajes.csv", index=False)
    df_agg.to_csv(f"{prefix}_agregado.csv", index=False)

    n_anchors = df_anchors['anchor'].nunique()
    print(f"[TOY-WF] {tag}: {len(df)} barras | opt={args.opt_size} fwd={args.fwd_size} "
          f"(proporción del experimento real, walk_forward_experiment.py:56-58) | "
          f"{n_anchors} anclajes × {len(PRESETS)} presets | {elapsed:.1f}s\n")
    winners = df_anchors[df_anchors['preset'].str.contains('ganador')]
    cols = ['anchor', 'preset', 'fast', 'slow', 'is_pnl', 'is_pf', 'fwd_pnl',
            'fwd_pf', 'robustez_is']
    print("[TOY-WF] Ganador de cada anclaje (mejor IS de 61 configs × 3 presets):")
    print(winners[cols].to_string(index=False))
    print("\n[TOY-WF] AGREGADO — el retrato del recitador:")
    print(df_agg.to_string(index=False))
    print(f"\n[TOY-WF] Salidas: {prefix}_anclajes.csv, {prefix}_agregado.csv")
    print("[TOY-WF] Lectura: is_pnl_medio es lo que el recitador PROMETE; "
          "fwd_pnl_medio es lo que ENTREGA. La versión con regímenes, CI "
          "bootstrap y filtros W4 es engine/regime_walk_forward.py.")


if __name__ == "__main__":
    main()
