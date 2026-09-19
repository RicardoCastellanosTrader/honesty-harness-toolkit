# -*- coding: utf-8 -*-
"""
E1 — Generador de series PLACEBO (noise floor). Pre-registrado, semilla fija.
6 series → data_cache/{NOMBRE}_1h.parquet (formato producción: timestamp_ms,open,high,low,close,volume).
  - GBM (iid log-returns gaussianos, σ calibrada a vol horaria real de BTC/SOL/SEI; drift 0).
  - Block-shuffle: barras OHLC reales reordenadas en bloques de 168h (1 semana) —
    preserva marginales + autocorrelación intra-semana, DESTRUYE la estructura multi-semana
    que el MA-cross (MAs ~75-340 barras) necesita.
NO toca lógica del pipeline: solo produce el parquet de entrada (como haría el download).

[honesty-harness-toolkit — sanitización Sesión 2, funciones INTACTAS]:
  1. ROOT era una ruta absoluta personal del entorno original; ahora es la raíz
     del toolkit (padre de engine/). Regla 3 de la sesión: sin rutas personales.
  2. La generación de las 6 series (que en el original corría a nivel de módulo)
     está ahora bajo `if __name__ == "__main__"` (función main()), para que
     `from placebo_gen import gbm, make_ohlc` sea importable sin efectos
     secundarios (lo requiere examples/data/make_synthetic.py).
  Ejecutar como script requiere BTCUSDT/SOLUSDT/SEIUSDT_1h.parquet en
  data_cache/ (descárgalos con download_full_history.py); el toolkit NO
  distribuye velas.
"""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data_cache"
N_BARS = 25000           # ~2.85 años horario
SEED = 20260613          # semilla registrada (pre-registro)
END_MS = int(pd.Timestamp("2026-06-01 00:00:00", tz="UTC").timestamp() * 1000)

SOURCES = {"BTC": "BTCUSDT_1h.parquet", "SOL": "SOLUSDT_1h.parquet", "SEI": "SEIUSDT_1h.parquet"}

def load_real(fn):
    df = pd.read_parquet(CACHE / fn)
    return df.tail(30000).reset_index(drop=True)

def calib(df):
    c = df["close"].values.astype(float)
    logret = np.diff(np.log(c))
    sigma = np.std(logret)                       # vol horaria real
    hl = (df["high"] - df["low"]) / df["close"]  # rango intrabar relativo
    oc = (df["close"] - df["open"]).abs() / df["close"]
    return sigma, float(np.median(hl)), float(np.median(oc)), float(df["close"].iloc[-1]), float(df["volume"].median())

def make_ohlc(close, hl_med, vol_med, rng):
    """Sintetiza OHLC desde una serie close: open=prev close, rango calibrado."""
    n = len(close)
    o = np.empty(n); o[0] = close[0]; o[1:] = close[:-1]
    hi = np.maximum(o, close) * (1 + np.abs(rng.normal(0, hl_med / 2, n)))
    lo = np.minimum(o, close) * (1 - np.abs(rng.normal(0, hl_med / 2, n)))
    vol = np.abs(rng.normal(vol_med, vol_med * 0.5, n)) + 1
    ts = END_MS - (np.arange(n)[::-1]) * 3_600_000
    return pd.DataFrame({"timestamp_ms": ts, "open": o, "high": hi, "low": lo, "close": close, "volume": vol})

def gbm(sigma, p0, n, rng):
    steps = rng.normal(0, sigma, n)              # drift 0
    return p0 * np.exp(np.cumsum(steps))

def block_shuffle(df, block=168, n=N_BARS, rng=None):
    """Reordena barras OHLC reales en bloques de 168h; reconstruye precio continuo."""
    d = df.tail(n + block).reset_index(drop=True)
    # retornos relativos por barra (close/open, high/open, low/open) preservados; encadenamos bloques
    o = d["open"].values; h = d["high"].values; l = d["low"].values; c = d["close"].values; v = d["volume"].values
    nb = len(d) // block
    idx_blocks = list(range(nb)); rng.shuffle(idx_blocks)
    rows = []
    price = c[0]
    for b in idx_blocks:
        sl = slice(b * block, (b + 1) * block)
        bo, bh, bl, bc, bv = o[sl], h[sl], l[sl], c[sl], v[sl]
        # ratios intrabar relativos al open del bloque-origen
        for i in range(len(bo)):
            ro_open = price
            scale = ro_open / bo[i]
            rows.append((ro_open, bh[i] * scale, bl[i] * scale, bc[i] * scale, bv[i]))
            price = bc[i] * scale
    rows = rows[:n]
    arr = np.array(rows)
    ts = END_MS - (np.arange(len(arr))[::-1]) * 3_600_000
    return pd.DataFrame({"timestamp_ms": ts, "open": arr[:,0], "high": arr[:,1], "low": arr[:,2], "close": arr[:,3], "volume": arr[:,4]})


def main():
    rng = np.random.default_rng(SEED)
    import random as _r
    manifest = []
    for i, (tag, fn) in enumerate(SOURCES.items()):
        real = load_real(fn)
        sigma, hl, oc, p0, volm = calib(real)
        # GBM
        g = gbm(sigma, p0, N_BARS, rng)
        dfg = make_ohlc(g, hl, volm, rng)
        name_g = f"PLBGB{i+1}"
        dfg.to_parquet(CACHE / f"{name_g}USDT_1h.parquet")
        manifest.append((name_g, f"GBM~{tag}", sigma, len(dfg)))
        # block-shuffle
        pyrng = _r.Random(SEED + i)
        dfs = block_shuffle(real, 168, N_BARS, pyrng)
        name_s = f"PLBSH{i+1}"
        dfs.to_parquet(CACHE / f"{name_s}USDT_1h.parquet")
        manifest.append((name_s, f"SHUF~{tag}", sigma, len(dfs)))
        print(f"{name_g} GBM~{tag} sigma={sigma:.5f} bars={len(dfg)}  |  {name_s} SHUF~{tag} bars={len(dfs)}", flush=True)

    import json
    json.dump([{"name":m[0],"type":m[1],"sigma":m[2],"bars":m[3]} for m in manifest],
              open(ROOT / "placebo_manifest.json","w"), indent=1)
    print("PLACEBO_DONE seed", SEED)


if __name__ == "__main__":
    main()
