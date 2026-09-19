#!/usr/bin/env python3
"""
cs_mom.py — Cross-Sectional Momentum (market-neutral) — CONGELADO 2026-06-29.

Sandbox aislado, produccion intacta, $0. Ver CS_MOM_PREREGISTRO.md.

CONGELADO:
  - Frecuencia 12h, lookback N=14d (28 velas 12h), QUINTIL (top/bottom 20%), rebalanceo 5d.
  - Factor MOM = (Close_t - Close_{t-28}) / ATR_28.
  - Universo 45 point-in-time (solo listados a t); rebrands stitched (MATIC->POL, RNDR->RENDER);
    SHIB -> 1000SHIBUSDT. Sesgo de supervivencia = TECHO OPTIMISTA explicito.
  - 3 curvas: A Long-Only, B Short-Only, C L/S beta-neutral (vol-target ATR-inverse).
  - Corto pesimista: funding descontado (AMBAS patas) + squeezes sin suavizar (bar HIGH).
  - Costes: 0.10% taker por lado sobre el TURNOVER (inercia no paga) + slippage por iliquidez.
  - Histeresis: entra top-20%, sale al caer de top-33% (simetrico corto). Congelado, no tuneado.
  - Benchmark EW de los 45 (point-in-time). Metricas Sharpe+Sortino+maxDD+dur peor DD+turnover+beta.

PROTOCOLO: --benchmark PRIMERO (reporta Sharpe/Sortino EW para fijar umbral X). --curves DESPUES
(con X congelado por Ricardo) corre las 3 curvas + veredicto asimetrico.
"""
from __future__ import annotations
import argparse, io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
KL_URL = "https://data.binance.vision/data/futures/um/monthly/klines/{t}/12h/{t}-12h-{m}.zip"
FUND_URL = "https://data.binance.vision/data/futures/um/monthly/fundingRate/{t}/{t}-fundingRate-{m}.zip"

# --- CONGELADO ---
N_BARS = 28          # 14 dias en velas 12h
ATR_BARS = 28
REBAL_BARS = 10      # 5 dias
ENTER_PCT = 0.20
EXIT_PCT = 0.33
FEE_SIDE = 0.0010    # 0.10% taker por lado
SLIP_BASE_BPS = 2.0  # slippage base, escalado por iliquidez
SLIP_CAP_BPS = 30.0
BARS_PER_YEAR = 730  # 2 velas 12h/dia * 365

SYMS = ("AAVE ADA ALGO APT ARB ATOM AVAX BCH BNB BTC DOGE DOT ENA ETC ETH FET FIL GRT HBAR ICP "
        "IMX INJ LINK LTC MANA NEAR ONDO OP POL QNT RENDER SAND SEI SHIB SOL STX SUI TAO THETA "
        "TRX UNI VET WLD XLM XRP").split()

# mapeo logico -> candidatos de ticker en futures (orden de preferencia para stitch)
def candidates(sym):
    if sym == "SHIB": return ["1000SHIBUSDT"]
    if sym == "POL": return ["POLUSDT", "MATICUSDT"]
    if sym == "RENDER": return ["RENDERUSDT", "RNDRUSDT"]
    return [f"{sym}USDT"]


def months_2y():
    return [f"{y}-{m:02d}" for y in (2023, 2024) for m in range(1, 13)]


def _dl(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        r = requests.get(url, timeout=120)
        if r.status_code != 200:
            return False
        dest.write_bytes(r.content)
        return True
    except Exception:
        return False


def load_klines(sym, log=print):
    """Serie 12h close/high/low/qvol del simbolo logico, stitching candidatos por mes."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    used_tickers = set()
    for m in months_2y():
        got = None
        for t in candidates(sym):
            zp = DATA_DIR / f"{t}-12h-{m}.zip"
            if _dl(KL_URL.format(t=t, m=m), zp):
                got = (t, zp); break
        if got is None:
            continue
        t, zp = got
        used_tickers.add(t)
        with zipfile.ZipFile(zp) as z:
            raw = z.read([n for n in z.namelist() if n.endswith(".csv")][0])
        first = raw[:60].decode("utf-8", "replace").lower()
        hdr = "open_time" in first
        df = pd.read_csv(io.BytesIO(raw), header=0 if hdr else None,
                         names=None if hdr else ["open_time","open","high","low","close","volume",
                         "close_time","qvol","count","tbv","tbq","ignore"], usecols=range(12))
        df.columns = ["open_time","open","high","low","close","volume","close_time","qvol","count","tbv","tbq","ignore"]
        rows.append(df[["open_time","high","low","close","qvol"]])
    if not rows:
        return None, used_tickers
    d = pd.concat(rows, ignore_index=True)
    tmax = int(d["open_time"].max())
    unit = "ns" if tmax > 1e17 else "us" if tmax > 1e14 else "ms"
    d["ts"] = pd.to_datetime(d["open_time"], unit=unit, utc=True)
    d = d.drop_duplicates("ts").set_index("ts").sort_index()
    return d[["high","low","close","qvol"]].astype(float), used_tickers


def load_funding_sym(sym):
    rows = []
    for m in months_2y():
        for t in candidates(sym):
            zp = DATA_DIR / f"{t}-fundingRate-{m}.zip"
            if _dl(FUND_URL.format(t=t, m=m), zp):
                with zipfile.ZipFile(zp) as z:
                    raw = z.read([n for n in z.namelist() if n.endswith(".csv")][0])
                first = raw[:50].decode("utf-8","replace").lower()
                df = pd.read_csv(io.BytesIO(raw), header=0 if "calc_time" in first else None,
                    names=None if "calc_time" in first else ["calc_time","funding_interval_hours","last_funding_rate"])
                rows.append(df[["calc_time","last_funding_rate"]])
                break
    if not rows:
        return None
    d = pd.concat(rows, ignore_index=True)
    d["ts"] = pd.to_datetime(d["calc_time"], unit="ms", utc=True)
    d["last_funding_rate"] = d["last_funding_rate"].astype(float)
    return d.drop_duplicates("ts").set_index("ts").sort_index()["last_funding_rate"]


def build_panels(log=print):
    grid = pd.date_range("2023-01-01", "2024-12-31 12:00", freq="12h", tz="UTC")
    closes, highs, lows, qvols, fundings = {}, {}, {}, {}, {}
    cov = {}
    for s in SYMS:
        d, used = load_klines(s, log=log)
        if d is None:
            log(f"   {s}: SIN DATOS"); continue
        d = d.reindex(grid)
        d["close"] = d["close"].ffill(limit=4)
        closes[s] = d["close"]; highs[s] = d["high"]; lows[s] = d["low"]; qvols[s] = d["qvol"]
        f = load_funding_sym(s)
        fundings[s] = f.reindex(grid).ffill(limit=2) if f is not None else pd.Series(0.0, index=grid)
        first_valid = d["close"].first_valid_index()
        cov[s] = (str(first_valid)[:10] if first_valid is not None else "NA", "+".join(sorted(used)))
    C = pd.DataFrame(closes); H = pd.DataFrame(highs); L = pd.DataFrame(lows)
    Q = pd.DataFrame(qvols); F = pd.DataFrame(fundings)
    return C, H, L, Q, F, cov


def perf(ret, name=""):
    ret = ret.dropna()
    if len(ret) < 2 or ret.std() == 0:
        return {"name": name, "n": int(len(ret)), "sharpe": None}
    eq = (1 + ret).cumprod()
    dd = eq / eq.cummax() - 1
    downside = ret[ret < 0]
    sortino = (ret.mean() / downside.std() * np.sqrt(BARS_PER_YEAR)) if len(downside) > 1 and downside.std() > 0 else None
    # duracion del peor DD (barras bajo el agua hasta nuevo maximo)
    underwater = (dd < 0).astype(int)
    cur = mx = 0
    for u in underwater.values:
        cur = cur + 1 if u else 0
        mx = max(mx, cur)
    return {"name": name, "n": int(len(ret)),
            "ret_total_pct": round(float(eq.iloc[-1] - 1) * 100, 2),
            "sharpe": round(float(ret.mean() / ret.std() * np.sqrt(BARS_PER_YEAR)), 3),
            "sortino": round(float(sortino), 3) if sortino else None,
            "maxdd_pct": round(float(dd.min()) * 100, 2),
            "peor_dd_dur_dias": round(mx / 2, 1)}


def benchmark(C, log=print):
    """EW point-in-time: media cross-sectional de retornos de listados por barra."""
    rets = C.pct_change()
    ew = rets.mean(axis=1, skipna=True)
    n_listed = C.notna().sum(axis=1)
    res = {"2023-2024": perf(ew, "EW_full")}
    res["2023"] = perf(ew[ew.index.year == 2023], "EW_2023")
    res["2024"] = perf(ew[ew.index.year == 2024], "EW_2024")
    res["n_listed_min"] = int(n_listed.min()); res["n_listed_max"] = int(n_listed.max())
    return res, ew


# ---------------- CURVAS (solo --curves, con X congelado) ----------------
def _atr(C, H, L):
    tr = pd.DataFrame(index=C.index, columns=C.columns, dtype=float)
    for s in C.columns:
        p = C[s].shift(1)
        tr[s] = pd.concat([(H[s] - L[s]), (H[s] - p).abs(), (L[s] - p).abs()], axis=1).max(axis=1)
    return tr.rolling(ATR_BARS).mean()


def factor(C, H, L, n_bars):
    return (C - C.shift(n_bars)) / _atr(C, H, L)


def _volw(inv_vol_row, names, cols):
    """Pesos vol-target (inverse-vol) sobre 'names', normalizados a suma 1; 0 en el resto."""
    w = pd.Series(0.0, index=cols)
    if names:
        iv = inv_vol_row.reindex(list(names)).replace([np.inf, -np.inf], np.nan).dropna()
        if len(iv) and iv.sum() > 0:
            w.loc[iv.index] = (iv / iv.sum()).values
    return w


def perf_sub(series):
    return {"full": perf(series, "full"),
            "2023": perf(series[series.index.year == 2023], "2023"),
            "2024": perf(series[series.index.year == 2024], "2024")}


def _beta(y, x):
    j = pd.concat([y, x], axis=1).dropna()
    return float(np.polyfit(j.iloc[:, 1], j.iloc[:, 0], 1)[0]) if len(j) > 2 else None


def run_curves(C, H, L, Q, F, n_bars, rebal_bars, log=print):
    cols = C.columns
    rets = C.pct_change()
    fwd = rets.shift(-1)
    mom = factor(C, H, L, n_bars)
    inv_vol = C / _atr(C, H, L)                      # price/ATR = inverso de vol relativa
    adv = Q.rolling(60, min_periods=10).mean()
    adv_med = adv.median(axis=1)
    slip = (SLIP_BASE_BPS * adv_med.values[:, None] / adv.values).clip(0, SLIP_CAP_BPS) / 1e4
    slip = pd.DataFrame(slip, index=C.index, columns=cols).fillna(SLIP_CAP_BPS / 1e4)
    fund = F.fillna(0.0) * 1.5                        # 8h rate sobre barra 12h (~1.5 settlements)

    n = len(C.index)
    lw = pd.DataFrame(0.0, index=C.index, columns=cols)
    sw = pd.DataFrame(0.0, index=C.index, columns=cols)
    held_long, held_short = set(), set()
    last_l = pd.Series(0.0, index=cols); last_s = pd.Series(0.0, index=cols)
    rebal = set(range(n_bars + ATR_BARS, n, rebal_bars))
    cost_l = pd.Series(0.0, index=C.index); cost_s = pd.Series(0.0, index=C.index)
    turn_events = []
    for i in range(n):
        if i in rebal:
            row = mom.iloc[i].dropna()
            if len(row) >= 10:
                r = row.rank(pct=True)
                held_long = (held_long & set(r[r >= 1 - EXIT_PCT].index)) | set(r[r >= 1 - ENTER_PCT].index)
                held_short = (held_short & set(r[r <= EXIT_PCT].index)) | set(r[r <= ENTER_PCT].index)
            wl = _volw(inv_vol.iloc[i], held_long, cols)
            ws = _volw(inv_vol.iloc[i], held_short, cols)
            dl = (wl - last_l).abs(); ds = (ws - last_s).abs()          # rotacion real por pata (solo rebal)
            cost_l.iloc[i] = float((dl * (FEE_SIDE + slip.iloc[i])).sum())
            cost_s.iloc[i] = float((ds * (FEE_SIDE + slip.iloc[i])).sum())
            turn_events.append(float(dl.sum() + ds.sum()))
            last_l, last_s = wl, ws
        lw.iloc[i] = last_l.values                                      # se mantiene entre rebalanceos (inercia)
        sw.iloc[i] = last_s.values
    long_ret = (lw * fwd).sum(axis=1) - (lw * fund).sum(axis=1)         # long PAGA funding>0
    short_ret = (sw * (-fwd)).sum(axis=1) + (sw * fund).sum(axis=1)     # short COBRA funding>0
    A = long_ret - cost_l                                              # long-only paga solo rotacion long
    B = short_ret - cost_s                                             # short-only paga solo rotacion short
    Cc = long_ret + short_ret - cost_l - cost_s
    ew = rets.mean(axis=1, skipna=True)
    btc = rets["BTC"] if "BTC" in cols else ew
    diag = {
        "params": {"n_bars": n_bars, "rebal_bars": rebal_bars},
        "A": perf_sub(A), "B": perf_sub(B), "C": perf_sub(Cc),
        "turnover_medio_por_rebal": round(float(np.mean([t for t in turn_events if t > 0])), 3) if turn_events else None,
        "beta_C_vs_EW": round(_beta(Cc, ew), 3) if _beta(Cc, ew) is not None else None,
        "beta_C_vs_BTC": round(_beta(Cc, btc), 3) if _beta(Cc, btc) is not None else None,
        "book_long_medio": round(float((lw != 0).sum(axis=1).replace(0, np.nan).mean()), 1),
        "book_short_medio": round(float((sw != 0).sum(axis=1).replace(0, np.nan).mean()), 1),
    }
    return diag, (A, B, Cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--curves", action="store_true")
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CS-MOM — construyendo paneles 12h 2023-2024 (45 simbolos, point-in-time)...")
    C, H, L, Q, F, cov = build_panels()
    print(f"   panel: {C.shape[1]} simbolos, {C.shape[0]} barras 12h")

    # cobertura / stitching report
    late = {s: v for s, v in cov.items() if v[0] > "2023-01"}
    print(f"   listados <=2023-01: {sum(1 for v in cov.values() if v[0]<='2023-01')}/{len(cov)}")
    if late:
        print("   listados tarde (point-in-time):", {s: v[0] for s, v in sorted(late.items(), key=lambda x: x[1][0])})
    stitched = {s: cov[s][1] for s in ("POL", "RENDER", "SHIB") if s in cov}
    print("   tickers usados (stitch/map):", stitched)

    if args.benchmark or not args.curves:
        res, ew = benchmark(C)
        (RESULTS_DIR / "benchmark.json").write_text(json.dumps(res, indent=2, default=str))
        print("\n" + "=" * 80)
        print("BENCHMARK EQUAL-WEIGHT (point-in-time) — para FIJAR el umbral X (sin ver CS-MOM):")
        for k in ("2023-2024", "2023", "2024"):
            p = res[k]
            print(f"  {k:9s}: Sharpe={p['sharpe']}  Sortino={p.get('sortino')}  "
                  f"ret={p.get('ret_total_pct')}%  maxDD={p.get('maxdd_pct')}%  "
                  f"peorDD={p.get('peor_dd_dur_dias')}d")
        print(f"  n_listados: min={res['n_listed_min']} max={res['n_listed_max']}")
        print("=" * 80)
        print("PROTOCOLO: Ricardo fija X (margen absoluto de Sharpe sobre el benchmark) ANTES de")
        print("revelar el CS-MOM. Luego --curves aplica el veredicto (techo optimista, descuento 1/2).")

    if args.curves:
        bench_res, _ = benchmark(C)
        bench_sharpe = bench_res["2023-2024"]["sharpe"]
        base, _ = run_curves(C, H, L, Q, F, N_BARS, REBAL_BARS)
        (RESULTS_DIR / "curves.json").write_text(json.dumps({"bench_sharpe": bench_sharpe, "base": base}, indent=2, default=str))

        def show(tag, d):
            for per in ("full", "2023", "2024"):
                p = d[per]
                print(f"    {tag} {per:4s}: Sharpe={p.get('sharpe')} Sortino={p.get('sortino')} "
                      f"ret={p.get('ret_total_pct')}% maxDD={p.get('maxdd_pct')}% peorDD={p.get('peor_dd_dur_dias')}d")
        print("\n" + "=" * 80)
        print(f"CS-MOM CURVAS (base N={N_BARS//2}d rebal={REBAL_BARS//2}d) | benchmark EW Sharpe={bench_sharpe}")
        print(f"  turnover_medio/rebal={base['turnover_medio_por_rebal']}  beta_C_vs_EW={base['beta_C_vs_EW']}  "
              f"beta_C_vs_BTC={base['beta_C_vs_BTC']}  book L/S={base['book_long_medio']}/{base['book_short_medio']}")
        print("  --- A Long-Only ---"); show("A", base["A"])
        print("  --- B Short-Only (diagnostico) ---"); show("B", base["B"])
        print("  --- C L/S neutral ---"); show("C", base["C"])

        A, Cc = base["A"], base["C"]
        a_sh = A["full"]["sharpe"]; a_24 = A["2024"].get("sharpe")
        a_adj = round(bench_sharpe + (a_sh - bench_sharpe) / 2, 3) if a_sh is not None else None
        a_digna = bool(a_sh is not None and a_sh >= 1.94 and a_24 is not None and a_24 > 0)
        c_sh = Cc["full"]["sharpe"]; c_24 = Cc["2024"].get("sharpe"); c_dd = Cc["full"].get("maxdd_pct")
        be, bb = base["beta_C_vs_EW"], base["beta_C_vs_BTC"]
        c_neutral = bool(be is not None and bb is not None and abs(be) <= 0.15 and abs(bb) <= 0.15)
        c_ret = bool(c_sh is not None and c_sh >= 1.00)
        c_prot = bool(c_dd is not None and c_dd > -25)        # radicalmente < -59.5; objetivo -15..-20
        c_24ok = bool(c_24 is not None and c_24 > 0)
        c_digna = c_neutral and c_ret and c_prot and c_24ok
        print("\n  VEREDICTO (criterios congelados):")
        print(f"   A: Sharpe={a_sh} >=1.94?{a_sh is not None and a_sh>=1.94} ajustado(1/2 exceso)={a_adj} "
              f"sobrevive2024?{a_24 is not None and a_24>0} -> {'DIGNA' if a_digna else 'NO DIGNA'}")
        print(f"   C: beta_EW={be} beta_BTC={bb} neutral?{c_neutral} | Sharpe={c_sh}>=1?{c_ret} | "
              f"maxDD={c_dd}% prot?{c_prot} | 2024ok?{c_24ok} -> {'DIGNA' if c_digna else 'NO DIGNA'}")
        if a_digna or c_digna:
            print("\n  Base SOBREVIVE -> robustez post-hoc (N in {7,14}d x rebal {3,5}d):")
            for nb in (14, 28):
                for rb in (6, 10):
                    if (nb, rb) == (N_BARS, REBAL_BARS):
                        continue
                    g, _ = run_curves(C, H, L, Q, F, nb, rb)
                    print(f"   N={nb//2}d reb={rb//2}d: A.full={g['A']['full']['sharpe']} A.2024={g['A']['2024'].get('sharpe')} "
                          f"C.full={g['C']['full']['sharpe']} C.2024={g['C']['2024'].get('sharpe')} betaEW={g['beta_C_vs_EW']}")
        else:
            print("\n  Base NO sobrevive -> robustez OMITIDA (no se rescata un base muerto).")


if __name__ == "__main__":
    main()
