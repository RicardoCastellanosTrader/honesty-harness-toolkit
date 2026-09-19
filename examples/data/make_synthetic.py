# -*- coding: utf-8 -*-
"""
make_synthetic.py — Genera la muestra sintética de la toy con el generador de
placebo CANÓNICO del estudio (engine/placebo_gen.py: gbm + make_ohlc), sin
necesitar velas reales.

GBM con drift 0: por construcción NO hay edge en esta serie. Es el dato de
entrada honesto para el primer contacto con la tubería toy — cualquier PnL
positivo que veas es selección sobre ruido.

Uso (desde la raíz del toolkit):
  python examples/data/make_synthetic.py
  python examples/data/make_synthetic.py --bars 25000 --sigma 0.011 --p0 50000 --seed 20260919

Salida: examples/data/TOYGBM1USDT_1h.parquet (formato caché de producción:
timestamp_ms, open, high, low, close, volume — el mismo que escribe
download_full_history.py).
"""
import os
import sys
import argparse

# Fix Windows cp1252 encoding (mismo patrón que engine/master.py:30-32)
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith('cp'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, 'engine'))

import numpy as np
from placebo_gen import gbm, make_ohlc  # funciones canónicas del estudio, intactas


def main():
    ap = argparse.ArgumentParser(description="Muestra sintética GBM para la toy")
    ap.add_argument('--bars', type=int, default=25000,
                    help="Barras 1h (default 25000, como el placebo del estudio)")
    ap.add_argument('--sigma', type=float, default=0.011,
                    help="Vol horaria de log-returns (default 0.011 ~ cripto líquida)")
    ap.add_argument('--p0', type=float, default=50000.0, help="Precio inicial")
    ap.add_argument('--hl-med', type=float, default=0.006,
                    help="Mediana rango intrabar relativo para sintetizar high/low")
    ap.add_argument('--vol-med', type=float, default=1000.0, help="Volumen mediano")
    ap.add_argument('--seed', type=int, default=20260919, help="Semilla RNG")
    ap.add_argument('--out', default=os.path.join(_HERE, 'TOYGBM1USDT_1h.parquet'))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    close = gbm(args.sigma, args.p0, args.bars, rng)
    df = make_ohlc(close, args.hl_med, args.vol_med, rng)
    df.to_parquet(args.out, index=False)
    print(f"[SYNTH] GBM drift-0: {len(df)} barras, sigma={args.sigma}, "
          f"seed={args.seed} -> {args.out}")
    print("[SYNTH] Esta serie NO tiene edge por construcción. Úsala como "
          "primer input de la toy y como referencia de noise floor.")


if __name__ == "__main__":
    main()
