#!/usr/bin/env python3
"""
run_edge.py — MFE Fase de EDGE — RUN 2 ANIOS (supervivencia + control estratificado).

CONGELADO (ver MFE_FASE_EDGE_PREREGISTRO.md). Sandbox aislado, produccion intacta, $0.

Pipeline (todo lo congelado):
  - Deteccion: firma 0.3%/15s & vol vendedor z>=3.0, SOLO bajistas, 2023-01..2024-12 BTC+ETH.
  - Entrada (ii): exhaustion = 1er seg tras disparo con ret trailing-15s >= 0, max 60s.
  - Slippage ENTRADA (falsador duro): peor precio (mas ALTO) de los siguientes 3s de raw
    aggTrades, precomputado como columna high3s. + 0.20% RT fees.
  - Salida: tiempo 1/3/5min + hard stop -2% MAE desde el precio de entrada.
  - Metrica: tasa de supervivencia (% neto>0) global y por bucket, estrategia Y control.
  - Control simetrico estratificado: 10 entradas aleatorias por cascada (>=5000/sym),
    semilla fija, maquinaria COMPLETA IDENTICA (exhaustion+stop+salida+slippage), por bucket.
  - Funding: anotacion as-of (ultimo calc_time <= t_disparo), buckets <=0 / 0-5e-4 / >5e-4,
    regla N>=30 (debajo -> reporta pero NO interpreta), auto-seleccion primario/secundario.
  - CI: cluster-bootstrap por ventana-8h (refinamiento 1).

NO look-ahead (verificado): baseline/velocidad usan solo pasado; exhaustion usa (disparo,
entrada]; slippage/forward usan [entrada, entrada+H] (el resultado del trade, no look-ahead);
funding as-of <= disparo. Gate prefix-invariance (modo --verify) DEBE pasar antes del run.

Modos:
  python run_edge.py --verify           # gate prefix-invariance (2 meses BTC), sin run
  python run_edge.py --run              # run completo 2 anios BTC+ETH
"""
from __future__ import annotations
import argparse, io, json, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
AGG_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades/{s}/{s}-aggTrades-{d}.zip"
FUND_URL = "https://data.binance.vision/data/futures/um/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip"
COLS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
        "transact_time", "is_buyer_maker"]

# --- CONGELADO ---
DROP_PCT = 0.003
VOL_Z = 3.0
VEL_WIN = 15
BASE_WIN = 3600
BASE_MIN = 600
COOLDOWN = 300
MAX_WAIT = 60          # exhaustion
SLIP_WIN = 3           # peor precio en los siguientes 3s
FEE_RT = 0.0020        # 0.20% round-trip
HARD_STOP = -0.02      # -2% MAE
HORIZONS = (60, 180, 300)
N_CTRL_PER_CASC = 10
N_CTRL_MIN = 5000
SEED = 20260629
BUCKETS = ("neg", "mid", "hi")
N_MIN_BUCKET = 30
N_BOOT = 10000


# ------------------------------------------------------------------ descarga/carga
def download_day(sym, date):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zp = DATA_DIR / f"{sym}-aggTrades-{date}.zip"
    if zp.exists() and zp.stat().st_size > 0:
        return zp
    r = requests.get(AGG_URL.format(s=sym, d=date), stream=True, timeout=240); r.raise_for_status()
    with open(zp, "wb") as f:
        for c in r.iter_content(1 << 20):
            f.write(c)
    return zp


def per_second(zp):
    with zipfile.ZipFile(zp) as z:
        raw = z.read([n for n in z.namelist() if n.endswith(".csv")][0])
    first = raw[:200].split(b"\n", 1)[0].decode("utf-8", "replace").lower()
    hdr = "transact_time" in first or "agg_trade_id" in first
    df = pd.read_csv(io.BytesIO(raw), header=0 if hdr else None,
                     names=None if hdr else COLS, usecols=range(7))
    df.columns = COLS
    bm = df["is_buyer_maker"]
    if bm.dtype == object:
        bm = bm.astype(str).str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False})
    df["is_buyer_maker"] = bm.astype(bool)
    tmax = int(df["transact_time"].max())
    unit = "ns" if tmax > 1e17 else "us" if tmax > 1e14 else "ms"
    ts = pd.to_datetime(df["transact_time"], unit=unit, utc=True)
    s = pd.DataFrame({"price": df["price"].values,
                      "sell": np.where(df["is_buyer_maker"].values, df["quantity"].values, 0.0)},
                     index=ts).sort_index()
    g = pd.DataFrame({
        "price": s["price"].resample("1s").last(),
        "hi": s["price"].resample("1s").max(),
        "lo": s["price"].resample("1s").min(),
        "sell_vol": s["sell"].resample("1s").sum(),
    })
    return g


def build_grid(sym, days, keep_raw=False, log=print):
    cache = DATA_DIR / f"grid_{sym}_{days[0]}_{days[-1]}.parquet"
    if cache.exists():
        log(f"   {sym}: grid cache HIT {cache.name} (sin re-descarga)")
        return pd.read_parquet(cache), 0.0
    grids, gb, t0 = [], 0, time.time()
    for i, d in enumerate(days):
        try:
            zp = download_day(sym, d)
        except Exception as e:
            log(f"   {d}: SKIP dl ({e})"); continue
        gb += zp.stat().st_size / 1e9
        try:
            grids.append(per_second(zp))
        except Exception as e:
            log(f"   {d}: SKIP parse ({e})")
        if not keep_raw:
            try: zp.unlink()
            except OSError: pass
        if (i + 1) % 60 == 0:
            log(f"   {sym}: {i+1}/{len(days)} dias, {gb:.1f} GB, {(time.time()-t0)/60:.1f} min")
    g = pd.concat(grids).sort_index()
    full = pd.date_range(g.index[0].floor("D"), g.index[-1].floor("D") + pd.Timedelta(days=1),
                         freq="1s", tz="UTC", inclusive="left")
    g = g.reindex(full)
    g["price"] = g["price"].ffill()
    g["hi"] = g["hi"].fillna(g["price"]); g["lo"] = g["lo"].fillna(g["price"])
    g["sell_vol"] = g["sell_vol"].fillna(0.0)
    # high3s: peor (mas alto) precio de los siguientes 3s (incl. el seg actual)
    g["high3s"] = g["hi"].rolling(SLIP_WIN).max().shift(-(SLIP_WIN - 1))
    log(f"   {sym}: grid {len(g):,} seg, {gb:.1f} GB, build {(time.time()-t0)/60:.1f} min")
    try:
        g.to_parquet(cache)
        log(f"   {sym}: grid cacheado -> {cache.name}")
    except Exception as e:
        log(f"   {sym}: WARN no se pudo cachear grid ({e})")
    return g, gb


def load_funding(sym, months):
    fr = []
    for m in months:
        zp = DATA_DIR / f"{sym}-fundingRate-{m}.zip"
        if not zp.exists():
            r = requests.get(FUND_URL.format(s=sym, m=m), timeout=60); r.raise_for_status()
            zp.write_bytes(r.content)
        with zipfile.ZipFile(zp) as z:
            raw = z.read([n for n in z.namelist() if n.endswith(".csv")][0])
        first = raw[:50].decode("utf-8", "replace").lower()
        d = pd.read_csv(io.BytesIO(raw), header=0 if "calc_time" in first else None,
                        names=None if "calc_time" in first else ["calc_time", "funding_interval_hours", "last_funding_rate"])
        fr.append(d)
    d = pd.concat(fr, ignore_index=True)
    d["calc_time"] = pd.to_datetime(d["calc_time"], unit="ms", utc=True)
    d["last_funding_rate"] = d["last_funding_rate"].astype(float)
    return d.sort_values("calc_time")[["calc_time", "last_funding_rate"]].reset_index(drop=True)


# ------------------------------------------------------------------ nucleo (arrays)
class Series:
    """Vista en arrays de un grid 1s continuo para medicion posicional rapida."""
    def __init__(self, g):
        self.start = g.index[0]
        self.price = g["price"].to_numpy()
        self.hi = g["hi"].to_numpy()
        self.lo = g["lo"].to_numpy()
        self.high3s = g["high3s"].to_numpy()
        self.n = len(self.price)
        p = g["price"]
        self.ret15 = (p / p.shift(VEL_WIN) - 1.0).to_numpy()
        sv = g["sell_vol"]
        cur = sv.rolling(VEL_WIN).sum() / VEL_WIN
        base = sv.shift(VEL_WIN + 1)
        bmean = base.rolling(BASE_WIN, min_periods=BASE_MIN).mean()
        bstd = base.rolling(BASE_WIN, min_periods=BASE_MIN).std()
        self.volz = ((cur - bmean) / bstd).to_numpy()
        self.bstd = bstd.to_numpy()

    def pos(self, ts):
        return int((ts - self.start).total_seconds())

    def ts(self, pos):
        return self.start + pd.Timedelta(seconds=int(pos))


def detect_triggers(S):
    """Posiciones de disparo (cascada bajista) con cooldown."""
    mask = (S.ret15 <= -DROP_PCT) & (S.volz >= VOL_Z) & np.isfinite(S.bstd) & (S.bstd > 0)
    cand = np.flatnonzero(mask)
    out, last = [], -10 ** 9
    for p in cand:
        if p - last >= COOLDOWN:
            out.append(p); last = p
    return np.array(out, dtype=np.int64)


def exhaustion_pos(S, ptrig):
    """1er seg en (trig, trig+MAX_WAIT] con ret15>=0; si no, trig+MAX_WAIT."""
    hi = min(ptrig + MAX_WAIT, S.n - 1)
    for p in range(ptrig + 1, hi + 1):
        r = S.ret15[p]
        if np.isfinite(r) and r >= 0:
            return p
    return hi


def measure(S, pentry):
    """Supervivencia por horizonte desde la entrada (slippage + stop + salida tiempo)."""
    epx = S.high3s[pentry]
    res = {"entry_pos": pentry, "entry_price": float(epx) if np.isfinite(epx) else np.nan}
    if not np.isfinite(epx) or epx <= 0:
        for H in HORIZONS:
            m = H // 60
            res[f"net_{m}m"] = np.nan; res[f"surv_{m}m"] = np.nan
            res[f"mae_{m}m"] = np.nan; res[f"mfe_{m}m"] = np.nan
        return res
    stop_level = epx * (1 + HARD_STOP)
    for H in HORIZONS:
        m = H // 60
        end = pentry + H
        if end >= S.n:
            res[f"net_{m}m"] = np.nan; res[f"surv_{m}m"] = np.nan
            res[f"mae_{m}m"] = np.nan; res[f"mfe_{m}m"] = np.nan
            continue
        lo_win = S.lo[pentry:end + 1]
        hi_win = S.hi[pentry:end + 1]
        stopped = np.any(lo_win <= stop_level)
        exit_px = stop_level if stopped else S.price[end]
        net = exit_px / epx - 1.0 - FEE_RT
        res[f"net_{m}m"] = float(net)
        res[f"surv_{m}m"] = bool(net > 0)
        res[f"mae_{m}m"] = float(lo_win.min() / epx - 1.0)
        res[f"mfe_{m}m"] = float(hi_win.max() / epx - 1.0)
    return res


def build_entries(S, trig_positions, kind):
    rows = []
    for pt in trig_positions:
        pe = exhaustion_pos(S, pt)
        r = measure(S, pe)
        r["trigger_pos"] = int(pt)
        r["trigger_ts"] = S.ts(pt)
        r["entry_ts"] = S.ts(pe)
        r["wait_s"] = int(pe - pt)
        r["kind"] = kind
        rows.append(r)
    return pd.DataFrame(rows)


def make_control_positions(S, n_ctrl, seed):
    """Disparos aleatorios uniformes donde el baseline existe y cabe el forward maximo."""
    lo = BASE_WIN  # baseline disponible
    hi = S.n - max(HORIZONS) - MAX_WAIT - SLIP_WIN - 2
    valid = np.flatnonzero(np.isfinite(S.bstd[lo:hi]) & (S.bstd[lo:hi] > 0)) + lo
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(valid, size=min(n_ctrl, len(valid)), replace=False))


def annotate_funding(df, funding):
    df = df.sort_values("trigger_ts").reset_index(drop=True).copy()
    df["trigger_ts"] = pd.to_datetime(df["trigger_ts"], utc=True).dt.as_unit("ns")
    fund = funding.copy()
    fund["calc_time"] = fund["calc_time"].dt.as_unit("ns")
    m = pd.merge_asof(df, fund, left_on="trigger_ts", right_on="calc_time", direction="backward")
    m = m.rename(columns={"last_funding_rate": "funding"})
    def buck(f):
        if pd.isna(f): return "na"
        if f <= 0: return "neg"
        if f <= 5e-4: return "mid"
        return "hi"
    m["bucket"] = m["funding"].map(buck)
    # ventana-8h (cluster id) = calc_time de la funding vigente
    m["fwin"] = m["calc_time"].astype("int64")
    return m


# ------------------------------------------------------------------ stats
def cluster_bootstrap(casc, ctrl, m, seed=SEED):
    """CI95 cluster-bootstrap por ventana-8h del gap(b)=surv_casc-surv_ctrl y contrastes."""
    col = f"surv_{m}m"
    c = casc.dropna(subset=[col]); k = ctrl.dropna(subset=[col])
    wins = np.union1d(c["fwin"].unique(), k["fwin"].unique())
    widx = {w: i for i, w in enumerate(wins)}
    nb = len(BUCKETS)
    # matrices [n_win, n_bucket]: sum survived y n, para casc y ctrl
    cs = np.zeros((len(wins), nb)); cn = np.zeros((len(wins), nb))
    ks = np.zeros((len(wins), nb)); kn = np.zeros((len(wins), nb))
    for arr_s, arr_n, dat in ((cs, cn, c), (ks, kn, k)):
        for bi, b in enumerate(BUCKETS):
            sub = dat[dat.bucket == b]
            for w, sv in zip(sub["fwin"].to_numpy(), sub[col].to_numpy()):
                j = widx[w]; arr_n[j, bi] += 1; arr_s[j, bi] += 1.0 if sv else 0.0
    rng = np.random.default_rng(seed)
    nwin = len(wins)
    gaps = {b: [] for b in BUCKETS}
    hi_neg, mid_neg = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, nwin, nwin)
        csn, cnn = cs[idx].sum(0), cn[idx].sum(0)
        ksn, knn = ks[idx].sum(0), kn[idx].sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            gc = csn / cnn; gk = ksn / knn
        gap = gc - gk
        for bi, b in enumerate(BUCKETS):
            gaps[b].append(gap[bi])
        hi_neg.append(gap[BUCKETS.index("hi")] - gap[BUCKETS.index("neg")])
        mid_neg.append(gap[BUCKETS.index("mid")] - gap[BUCKETS.index("neg")])

    def ci(x):
        x = np.array(x); x = x[np.isfinite(x)]
        if len(x) == 0: return (None, None, None)
        return (round(float(np.nanpercentile(x, 2.5)), 4),
                round(float(np.nanmedian(x)), 4),
                round(float(np.nanpercentile(x, 97.5)), 4))
    return {"gap_por_bucket": {b: ci(gaps[b]) for b in BUCKETS},
            "hi_minus_neg": ci(hi_neg), "mid_minus_neg": ci(mid_neg)}


def survival_table(df, m):
    col = f"surv_{m}m"
    out = {}
    for b in ("neg", "mid", "hi", "na"):
        sub = df[(df.bucket == b)].dropna(subset=[col])
        out[b] = {"n": int(len(sub)),
                  "surv": round(float(sub[col].mean()), 4) if len(sub) else None,
                  "worst_mae": round(float(df[df.bucket == b][f"mae_{m}m"].min()), 4) if len(sub) else None}
    g = df.dropna(subset=[col])
    out["GLOBAL"] = {"n": int(len(g)), "surv": round(float(g[col].mean()), 4) if len(g) else None}
    return out


# ------------------------------------------------------------------ gate look-ahead
def detect_and_measure_casc(S):
    trig = detect_triggers(S)
    return build_entries(S, trig, "cascade")


def verify_no_lookahead(g, log=print):
    """Prefix-invariance: anadir datos futuros NO cambia disparos ni supervivencia pasadas."""
    S = Series(g)
    full = detect_and_measure_casc(S)
    M = S.n // 2
    gt = g.iloc[:M]
    St = Series(gt)
    trunc = detect_and_measure_casc(St)
    cutoff = S.ts(M)
    safe = full[full["entry_ts"] + pd.Timedelta(seconds=max(HORIZONS)) < cutoff].copy()
    saft = trunc[trunc["entry_ts"] + pd.Timedelta(seconds=max(HORIZONS)) < cutoff].copy()
    ok = True
    msgs = []
    if len(safe) != len(saft):
        ok = False; msgs.append(f"conteo difiere full={len(safe)} trunc={len(saft)}")
    else:
        a = safe.sort_values("trigger_ts").reset_index(drop=True)
        b = saft.sort_values("trigger_ts").reset_index(drop=True)
        if not (a["trigger_ts"].values == b["trigger_ts"].values).all():
            ok = False; msgs.append("trigger_ts difieren")
        for c in ("entry_ts",):
            if not (a[c].values == b[c].values).all():
                ok = False; msgs.append(f"{c} difiere")
        for c in ("net_1m", "net_3m", "net_5m", "entry_price"):
            va, vb = a[c].to_numpy(), b[c].to_numpy()
            both = np.isfinite(va) & np.isfinite(vb)
            if not np.allclose(va[both], vb[both], rtol=0, atol=0):
                ok = False; msgs.append(f"{c} difiere (max|d|={np.nanmax(np.abs(va[both]-vb[both])):.2e})")
    log(f"   prefix-invariance: cascadas seguras comparadas={len(safe)} | {'PASS' if ok else 'FAIL: '+'; '.join(msgs)}")
    return ok


# ------------------------------------------------------------------ run
def months_range(y0, m0, y1, m1):
    out = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        out.append(f"{y}-{m:02d}")
        m += 1
        if m > 12: m = 1; y += 1
    return out


def days_range(start, end):
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D", tz="UTC")]


def process_symbol(sym, days, fmonths, log=print):
    g, gb = build_grid(sym, days, log=log)
    if not verify_no_lookahead(g, log=log):
        raise RuntimeError(f"GATE look-ahead FAIL en {sym} — abortado antes de medir")
    S = Series(g)
    funding = load_funding(sym, fmonths)
    trig = detect_triggers(S)
    casc = build_entries(S, trig, "cascade")
    n_ctrl = max(N_CTRL_PER_CASC * len(casc), N_CTRL_MIN)
    ctrl_pos = make_control_positions(S, n_ctrl, SEED)
    ctrl = build_entries(S, ctrl_pos, "control")
    casc = annotate_funding(casc, funding)
    ctrl = annotate_funding(ctrl, funding)
    casc.to_csv(RESULTS_DIR / f"edge_cascades_{sym}.csv", index=False)
    ctrl.to_csv(RESULTS_DIR / f"edge_control_{sym}.csv", index=False)

    rep = {"sym": sym, "gb": round(gb, 1), "n_cascadas": int(len(casc)), "n_control": int(len(ctrl)),
           "conteo_bucket": {b: int((casc.bucket == b).sum()) for b in ("neg", "mid", "hi", "na")},
           "por_horizonte": {}}
    for H in HORIZONS:
        m = H // 60
        st_c = survival_table(casc, m)
        st_k = survival_table(ctrl, m)
        boot = cluster_bootstrap(casc, ctrl, m)
        gap_point = {b: (round(st_c[b]["surv"] - st_k[b]["surv"], 4)
                         if st_c[b]["surv"] is not None and st_k[b]["surv"] is not None else None)
                     for b in BUCKETS}
        rep["por_horizonte"][f"{m}m"] = {
            "surv_cascada": st_c, "surv_control": st_k,
            "gap_point": gap_point, "gap_ci_cluster8h": boot}
    return rep, casc


def verdict(reports):
    """Auto-seleccion primario/secundario + monotonia + consistencia BTC+ETH."""
    v = {"por_simbolo": {}}
    for sym, rep in reports.items():
        n_hi = rep["conteo_bucket"]["hi"]; n_neg = rep["conteo_bucket"]["neg"]; n_mid = rep["conteo_bucket"]["mid"]
        test = "primario" if (n_hi >= N_MIN_BUCKET and n_neg >= N_MIN_BUCKET) else "secundario"
        per_h = {}
        for hk, d in rep["por_horizonte"].items():
            gp = d["gap_point"]; boot = d["gap_ci_cluster8h"]
            mono = (gp["neg"] is not None and gp["mid"] is not None and gp["hi"] is not None
                    and gp["neg"] <= gp["mid"] <= gp["hi"])
            if test == "primario":
                key = boot["hi_minus_neg"]
            else:
                key = boot["mid_minus_neg"]
            ci_excl0 = key[0] is not None and (key[0] > 0)
            per_h[hk] = {"gap_point": gp, "contraste_ci": key, "ci_excluye_0_por_arriba": ci_excl0,
                         "monotono": mono}
        v["por_simbolo"][sym] = {"test_operativo": test,
                                 "N": {"neg": n_neg, "mid": n_mid, "hi": n_hi},
                                 "por_horizonte": per_h}
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.verify:
        print("=== GATE prefix-invariance (no look-ahead) — 2 meses BTCUSDT 2024-02..03 ===")
        days = days_range("2024-02-01", "2024-03-31")
        g, _ = build_grid("BTCUSDT", days)
        ok = verify_no_lookahead(g)
        print("RESULTADO GATE:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    if args.run:
        t0 = time.time()
        print("=" * 80)
        print("MFE RUN EDGE 2 ANIOS (2023-01-01..2024-12-31) — supervivencia + control 10x")
        print("=" * 80)
        days = days_range("2023-01-01", "2024-12-31")
        fmonths = months_range(2022, 12, 2025, 1)
        reports = {}
        for sym in args.symbols:
            print(f"\n[{sym}] {len(days)} dias...")
            rep, _ = process_symbol(sym, days, fmonths)
            reports[sym] = rep
            print(f"  {sym}: cascadas={rep['n_cascadas']} control={rep['n_control']} "
                  f"buckets={rep['conteo_bucket']}")
        vd = verdict(reports)
        out = {"reports": reports, "verdict": vd, "horas": round((time.time() - t0) / 3600, 2)}
        (RESULTS_DIR / "edge_summary.json").write_text(json.dumps(out, indent=2, default=str))

        print("\n" + "=" * 80)
        print("RESUMEN EDGE (estrategia vs control, por bucket, gap + CI cluster-8h):")
        for sym, rep in reports.items():
            print(f"\n--- {sym} | test operativo: {vd['por_simbolo'][sym]['test_operativo']} "
                  f"| N {vd['por_simbolo'][sym]['N']} ---")
            for m in (1, 3, 5):
                d = rep["por_horizonte"][f"{m}m"]
                sc, sk = d["surv_cascada"], d["surv_control"]
                print(f"  {m}min: surv_casc G={sc['GLOBAL']['surv']} "
                      f"[neg={sc['neg']['surv']}/{sc['neg']['n']} mid={sc['mid']['surv']}/{sc['mid']['n']} "
                      f"hi={sc['hi']['surv']}/{sc['hi']['n']}] | gap_point={d['gap_point']}")
                vh = vd["por_simbolo"][sym]["por_horizonte"][f"{m}m"]
                print(f"        contraste(CI cluster-8h)={vh['contraste_ci']} excl0+={vh['ci_excluye_0_por_arriba']} "
                      f"monotono={vh['monotono']}")
        print(f"\nhoras: {out['horas']}")
        print("=" * 80)
        print("RECORDATORIO: backtest = FALSADOR. Un 'si' da derecho a paper-trading en vivo,")
        print("NO confirma edge (senal incompleta sin ping forceOrder + slippage real no simulable).")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
