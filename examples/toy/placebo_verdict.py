# -*- coding: utf-8 -*-
"""
placebo_verdict.py — El placebo como VEREDICTO para la tubería toy.

Rito completo del Arnés (docs/ACTAS.md): prerregistro congelado → hash →
run → veredicto en §10 del propio prerregistro. ESTE SCRIPT IMPONE EL ORDEN:
no corre sobre datos reales si examples/toy/PREREGISTRO_TOY.md no está
congelado (sidecar .sha256 presente y coincidente con la parte congelada).

Qué mide (decisiones del director, Sesión 3):
  (a) 20 mundos GBM drift-0 (engine/placebo_gen.gbm + make_ohlc, σ/rango/vol
      calibrados sobre BTC real con placebo_gen.calib) — semillas fijas listadas.
  (b) En cada mundo: el MEJOR PnL de una búsqueda de 10, 61 y 1000
      configuraciones EMA-cross → distribución del suelo de ruido por tamaño
      de búsqueda (media, mediana, [p2.5, p97.5]).
  (c) El mismo procedimiento sobre BTC real.
  (d) Comparación real vs suelo con el criterio congelado en el prerregistro.

Universos (congelados en el prerregistro):
  - U61  = universo canónico de la toy (toy_kernel.generate_toy_configs).
  - U10  = 10 configs de U61, índices 0,6,12,...,54 del array canónico.
  - U1000 = las primeras 1000 (orden lexicográfico fast,slow) del grid denso
    fast ∈ {3,5,...,49} (paso 2) × slow ∈ {20,27,...,335} (paso 7), fast<slow.
  Warmup unificado 350 barras (máx. período del grid denso + margen).

Uso (desde la raíz del toolkit):
  python examples/toy/placebo_verdict.py --universe   # imprime universos (sin datos)
  python examples/toy/placebo_verdict.py --freeze     # sella el prerregistro (sha256)
  python examples/toy/placebo_verdict.py --run        # floor GBM + real BTC (exige hash)

Sello de tiempo OPCIONAL (OpenTimestamps, como los prerregistros del estudio):
  ots stamp examples/toy/PREREGISTRO_TOY.md.sha256
  (genera .ots con atestación Bitcoin diferida; `ots upgrade` horas después)
"""
import os
import sys
import json
import time
import hashlib
import argparse

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith('cp'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'engine'))
sys.path.insert(0, os.path.join(_ROOT, 'pipelines', 'toy'))

import numpy as np
import pandas as pd

from placebo_gen import gbm, make_ohlc, calib          # canónicos del estudio
from toy_features import TOY_FAST_PERIODS, TOY_SLOW_PERIODS, _ema
from toy_kernel import (generate_toy_configs, run_ema_cross_pairs_numba,
                        COMMISSION_ROUND_TRIP)

PREREG = os.path.join(_HERE, 'PREREGISTRO_TOY.md')
SIDECAR = PREREG + '.sha256'
FREEZE_MARKER = '<!-- FREEZE-BOUNDARY'
RESULTS_JSON = os.path.join(_HERE, 'placebo_verdict_results.json')

# --- Parámetros CONGELADOS (deben coincidir con el prerregistro) ---
N_WORLDS = 20
BASE_SEED = 20260919
SEEDS = [BASE_SEED + i for i in range(N_WORLDS)]
WARMUP = 350
COMMISSION = COMMISSION_ROUND_TRIP    # 0.10% RT, igual que el motor
HYST = 0.0                            # preset laxo
DENSE_FAST = list(range(3, 50, 2))    # 24 valores: 3..49
DENSE_SLOW = list(range(20, 336, 7))  # 46 valores: 20..335
N_DENSE_TARGET = 1000
U10_INDICES = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]
BTC_PARQUET = os.path.join(_ROOT, 'data_cache', 'BTCUSDT_1h.parquet')


def build_universes():
    """Devuelve (periods, pairs_dense, pairs_61, pairs_10).

    periods: lista ordenada de períodos únicos (filas de la matriz EMA).
    pairs_*: arrays (n,2) de índices de fila (fast_row, slow_row).
    """
    canon = generate_toy_configs()
    canon_pairs_periods = []
    for cfg in canon:
        fi = cfg & 0x7
        si = (cfg >> 3) & 0x7
        canon_pairs_periods.append((int(TOY_FAST_PERIODS[fi]), int(TOY_SLOW_PERIODS[si])))

    dense_pairs_periods = [(f, s) for f in DENSE_FAST for s in DENSE_SLOW if f < s]
    dense_pairs_periods.sort()
    dense_pairs_periods = dense_pairs_periods[:N_DENSE_TARGET]

    periods = sorted({p for pair in canon_pairs_periods + dense_pairs_periods for p in pair})
    row = {p: i for i, p in enumerate(periods)}

    def to_rows(pairs):
        return np.array([[row[f], row[s]] for f, s in pairs], dtype=np.int64)

    pairs_61 = to_rows(canon_pairs_periods)
    pairs_dense = to_rows(dense_pairs_periods)
    pairs_10 = pairs_61[U10_INDICES]
    return periods, pairs_dense, pairs_61, pairs_10


def ema_matrix(close, periods):
    close = np.ascontiguousarray(close, dtype=np.float64)
    m = np.empty((len(periods), len(close)), dtype=np.float64)
    for i, p in enumerate(periods):
        m[i, :] = _ema(close, p)
    return m


def best_of(close, periods, pairs):
    """PnL neto % del mejor config del universo `pairs` sobre `close`."""
    m = ema_matrix(close, periods)
    res = run_ema_cross_pairs_numba(pairs, np.ascontiguousarray(close, dtype=np.float64),
                                    m, float(COMMISSION), WARMUP, len(close), HYST)
    k = int(np.argmax(res[:, 0]))
    return float(res[k, 0]), k


def frozen_hash():
    """sha256 de la parte CONGELADA del prerregistro (hasta la línea del
    marcador FREEZE-BOUNDARY inclusive). El §10 (veredicto) va después del
    marcador y no altera el hash."""
    if not os.path.exists(PREREG):
        return None
    with open(PREREG, 'rb') as f:
        raw = f.read()
    marker = FREEZE_MARKER.encode('utf-8')
    idx = raw.find(marker)
    if idx < 0:
        return None
    end = raw.find(b'\n', idx)
    end = len(raw) if end < 0 else end + 1
    return hashlib.sha256(raw[:end]).hexdigest()


def cmd_freeze():
    h = frozen_hash()
    if h is None:
        sys.exit(f"ERROR: {PREREG} no existe o no contiene el marcador "
                 f"'{FREEZE_MARKER}'. Redacta las secciones 0-9 primero.")
    with open(SIDECAR, 'w') as f:
        f.write(h + '\n')
    print(f"[FREEZE] sha256 de la parte congelada (secciones 0-9): {h}")
    print(f"[FREEZE] Sidecar escrito: {SIDECAR}")
    print("[FREEZE] Sello de tiempo opcional: ots stamp " + os.path.relpath(SIDECAR, _ROOT))


def _require_frozen():
    h = frozen_hash()
    if h is None:
        sys.exit("ABORT: el prerregistro no existe o no tiene marcador de freeze. "
                 "El rito es: redactar §0-9 → --freeze → --run. No se corre sobre real sin hash.")
    if not os.path.exists(SIDECAR):
        sys.exit("ABORT: no hay sidecar .sha256 — el prerregistro NO está congelado. "
                 "Ejecuta --freeze antes de --run.")
    with open(SIDECAR) as f:
        recorded = f.read().strip()
    if recorded != h:
        sys.exit(f"ABORT: el prerregistro fue MODIFICADO tras el freeze "
                 f"(hash actual {h[:16]}… != sellado {recorded[:16]}…). "
                 f"Un cambio post-freeze exige enmienda fechada + re-freeze explícito.")
    return h


def cmd_run():
    h = _require_frozen()
    print(f"[GATE] Prerregistro congelado verificado (sha256 {h[:16]}…). Procediendo.\n")

    if not os.path.exists(BTC_PARQUET):
        sys.exit(f"ABORT: falta {BTC_PARQUET}. Descárgalo con "
                 "`python pipelines/toy/toy_pipeline.py --symbol BTC/USDT --download`.")

    periods, pairs_dense, pairs_61, pairs_10 = build_universes()
    sizes = {'10': pairs_10, '61': pairs_61, '1000': pairs_dense}
    print(f"[UNIVERSOS] filas EMA: {len(periods)} períodos | "
          f"|U10|={len(pairs_10)} |U61|={len(pairs_61)} |U1000|={len(pairs_dense)}")

    btc = pd.read_parquet(BTC_PARQUET)
    n = len(btc)
    sigma, hl_med, _oc, p0, vol_med = calib(btc)
    print(f"[CALIB] BTC: {n} barras | sigma_1h={sigma:.5f} | hl_med={hl_med:.5f} "
          f"| p0={p0:.0f} (placebo_gen.calib, como E1 del estudio)\n")

    # (a)+(b) suelo de ruido: 20 mundos GBM drift-0
    t0 = time.perf_counter()
    floor = {k: [] for k in sizes}
    for w, seed in enumerate(SEEDS):
        rng = np.random.default_rng(seed)
        close = gbm(sigma, p0, n, rng)
        # make_ohlc genera el OHLC completo pero la toy solo usa close;
        # se invoca igualmente para mantener el procedimiento del estudio.
        for k, pairs in sizes.items():
            b, _ = best_of(close, periods, pairs)
            floor[k].append(b)
        print(f"  mundo {w+1:2d}/20 seed={seed}: "
              + " | ".join(f"best-{k}={floor[k][-1]:+8.1f}%" for k in sizes))
    t_floor = time.perf_counter() - t0

    def stats(v):
        a = np.array(v)
        return {'media': round(float(a.mean()), 1),
                'mediana': round(float(np.median(a)), 1),
                'p2_5': round(float(np.percentile(a, 2.5)), 1),
                'p97_5': round(float(np.percentile(a, 97.5)), 1),
                'min': round(float(a.min()), 1), 'max': round(float(a.max()), 1)}

    floor_stats = {k: stats(v) for k, v in floor.items()}

    # (c) mismo procedimiento sobre BTC real
    t1 = time.perf_counter()
    real = {}
    for k, pairs in sizes.items():
        b, _ = best_of(btc['close'].values, periods, pairs)
        real[k] = round(b, 1)
    t_real = time.perf_counter() - t1

    # (d) criterio congelado (§4 del prerregistro): real_61 vs [p2.5, p97.5] floor_61
    lo, hi = floor_stats['61']['p2_5'], floor_stats['61']['p97_5']
    if real['61'] > hi:
        verdict = ("SUPERA_EL_SUELO_DRIFT0 — NO se interpreta como edge "
                   "(confundido con la deriva de BTC, §9 del prerregistro); "
                   "en un experimento serio solo habilitaría la fase siguiente "
                   "(walk-forward + placebo con deriva modelada)")
    elif real['61'] < lo:
        verdict = "PEOR_QUE_RUIDO"
    else:
        verdict = "INDISTINGUIBLE_DEL_RUIDO_DE_SELECCION"

    # Contexto post-hoc (Sesión 4, NO forma parte del veredicto pre-registrado
    # ni altera el prerregistro sellado): buy & hold de BTC en la MISMA ventana
    # de accounting [WARMUP, n) — el sofá contra el que se relativiza todo.
    close_real = btc['close'].values
    buy_hold = (float(close_real[-1]) / float(close_real[WARMUP]) - 1.0) * 100.0

    out = {
        'prereg_sha256': h, 'seeds': SEEDS, 'n_bars': n,
        'calib': {'sigma': sigma, 'hl_med': hl_med, 'p0': p0},
        'warmup': WARMUP, 'commission_pct': COMMISSION, 'hyst': HYST,
        'floor_values': floor, 'floor_stats': floor_stats, 'real': real,
        'criterio': 'real_61 vs [p2.5, p97.5] de floor_61 (3 zonas)',
        'verdict_class': verdict.split(' ')[0],
        'post_hoc_context': {
            'buy_and_hold_pct': round(buy_hold, 1),
            'note': 'post-hoc context (not part of the pre-registered verdict)',
            'window': f'[{WARMUP}, {n}) — misma ventana de accounting que la métrica',
        },
        'tiempos_s': {'floor_20_mundos': round(t_floor, 1), 'real': round(t_real, 1)},
    }
    with open(RESULTS_JSON, 'w') as f:
        json.dump(out, f, indent=1)

    print(f"\n[SUELO] Distribución del mejor-de-N sobre RUIDO drift-0 (20 mundos, {t_floor:.1f}s):")
    print(f"{'N':>6} {'media':>9} {'mediana':>9} {'p2.5':>9} {'p97.5':>9} {'min':>9} {'max':>9}")
    for k in ('10', '61', '1000'):
        s = floor_stats[k]
        print(f"{k:>6} {s['media']:>9} {s['mediana']:>9} {s['p2_5']:>9} "
              f"{s['p97_5']:>9} {s['min']:>9} {s['max']:>9}")
    print(f"\n[REAL] BTC {n} barras ({t_real:.1f}s): "
          + " | ".join(f"best-{k}={real[k]:+.1f}%" for k in ('10', '61', '1000')))
    print(f"\n[VEREDICTO mecánico] real_61={real['61']:+.1f}% vs suelo_61 [{lo}, {hi}] → {verdict}")
    print(f"\n[SOFÁ] post-hoc context (not part of the pre-registered verdict): "
          f"buy & hold BTC en la misma ventana [{WARMUP}, {n}) = {buy_hold:+.1f}%")
    print(f"[SALIDA] {RESULTS_JSON}")
    print("\nAhora redacta §10 del PREREGISTRO_TOY.md con estos números "
          "(DEBAJO del marcador de freeze — el hash sellado no cambia).")


def cmd_universe():
    periods, pairs_dense, pairs_61, pairs_10 = build_universes()
    print(f"períodos únicos: {len(periods)} → {periods}")
    print(f"|U10|={len(pairs_10)} |U61|={len(pairs_61)} |U1000|={len(pairs_dense)}")
    print(f"U1000: fast {DENSE_FAST[0]}..{DENSE_FAST[-1]} paso 2 × "
          f"slow {DENSE_SLOW[0]}..{DENSE_SLOW[-1]} paso 7, fast<slow, "
          f"primeras {N_DENSE_TARGET} lexicográficas")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--universe', action='store_true', help="imprime universos (sin datos)")
    g.add_argument('--freeze', action='store_true', help="sella el prerregistro (sidecar sha256)")
    g.add_argument('--run', action='store_true', help="floor GBM + BTC real (exige freeze)")
    args = ap.parse_args()
    if args.universe:
        cmd_universe()
    elif args.freeze:
        cmd_freeze()
    else:
        cmd_run()
