# -*- coding: utf-8 -*-
"""
run_sandbox.py — Esqueleto de sandbox del Arnés de Honestidad.

Contrato de sandbox (extraído de los sandboxes reales cs_mom y mfe):
  1. CERO imports del núcleo del toolkit (aislamiento deliberado: si tu
     hipótesis contamina el motor, no sabrás cuál de los dos mintió).
     Solo stdlib + numpy/pandas (+requests si descargas datos públicos).
  2. Loaders de datos PROPIOS, point-in-time (nada de mirar el futuro).
  3. Control/placebo PROPIO y congelado en el prerregistro: entradas
     aleatorias simétricas con semilla fija (patrón mfe), benchmark trivial
     point-in-time (patrón cs_mom), o GBM drift-0.
  4. El código computa su parte del veredicto y la vuelca a results/*.json;
     el veredicto humano se escribe en §10 de PREREGISTRO.md.
  5. Fases separadas si aplica (patrón mfe): fase 0 mecánica (PROHIBIDO
     concluir edge), smoke de calibración (PROHIBIDO mirar la métrica
     cardinal), run decisivo con gate anti-lookahead que ABORTA si falla.

Uso: python run_sandbox.py --phase0 | --run
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SEED = 0  # FIJAR y registrar en PREREGISTRO.md §6 antes del freeze


def load_data():
    """Loader propio del sandbox. Point-in-time. Documenta la fuente en §2."""
    raise NotImplementedError("implementa tu loader (y decláralo en PREREGISTRO.md §2)")


def run_control(rng):
    """Control/placebo congelado en §5. Misma maquinaria que la estrategia."""
    raise NotImplementedError


def run_strategy():
    """La hipótesis de §1, con los parámetros a priori de §6. Nada más."""
    raise NotImplementedError


def verdict(strategy_metrics, control_metrics):
    """Computa la parte mecánica del veredicto contra los umbrales de §4.
    El veredicto HUMANO se escribe en PREREGISTRO.md §10, no aquí."""
    out = {
        "metric_cardinal": None,      # la métrica de §3
        "threshold": None,            # el umbral de §4
        "control": control_metrics,
        "gates_passed": False,        # anti-leakage de §7
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "verdict.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0", action="store_true",
                    help="Solo mecánica de la tubería. PROHIBIDO concluir edge.")
    ap.add_argument("--run", action="store_true",
                    help="Run decisivo. Requiere PREREGISTRO.md CONGELADO.")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    if args.phase0:
        load_data()
        print("[FASE 0] Tubería mecánica OK. Recuerda: aquí NO se concluye edge.")
        return
    if args.run:
        s = run_strategy()
        c = run_control(rng)
        v = verdict(s, c)
        print("[RUN] verdict.json escrito. Ahora escribe §10 del PREREGISTRO "
              "contra los criterios congelados — la métrica cardinal primero.")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
