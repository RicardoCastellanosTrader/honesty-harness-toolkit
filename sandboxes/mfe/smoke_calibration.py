#!/usr/bin/env python3
"""
smoke_calibration.py — MFE Fase de EDGE — SMOKE DE CALIBRACION (mecanica + conteo).

AUTORIZADO (freeze 2026-06-29). Sandbox aislado, produccion intacta, $0.

OBJETIVO (NO es el run de edge):
  1) Validar coste/disco del pipeline real (descarga + parseo dia-a-dia + cleanup).
  2) OUTPUT PRIMARIO = CONTEO de cascadas por bucket de funding, ANTES de cualquier
     tasa de supervivencia. Decide si el bucket >5e-4 (Escenario A) promete N>=30 a 2
     anios o es fantasma -> define si el test PRIMARIO o el SECUNDARIO es el operativo.

PROHIBIDO en el smoke: medir/mirar supervivencia (Blindaje 1 recursivo). Solo mecanica.

Firma CONGELADA: caida >= drop_pct en 15s (def 0.3%) & vol vendedor z >= vol_z (def 3.0).
Entrada CONGELADA (ii): exhaustion = 1er segundo tras el disparo con ret trailing-15s >= 0
  (descompresion de velocidad), espera maxima 60s; si no, entrada forzada a +60s.
Funding CONGELADO: anotacion as-of, last_funding_rate con calc_time <= t_disparo.
Buckets CONGELADOS: <=0 (neg) / 0..5e-4 (mid) / >5e-4 (hi). N-minimo 30 por bucket.

Disco (Caveat #13): raw aggTrades se descarga -> se procesa a grid 1s -> se BORRA. Solo
  el grid 1s (ligero) se acumula en memoria para deteccion sobre serie continua del mes
  (sin artefacto de frontera diaria; el run de 2 anios chunkea por mes con warmup 1h).
"""
from __future__ import annotations
import argparse, io, json, os, time, zipfile
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

# >5e-4 windows 2023-2024 (verificado primary-source) para escalar la proyeccion del bucket hi
HI_WINDOWS_2YR = {"BTCUSDT": 26, "ETHUSDT": 30}
HI_WINDOWS_MAR2024 = {"BTCUSDT": 15, "ETHUSDT": 15}


def days_in_month(year: int, month: int):
    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    end = (start + pd.offsets.MonthBegin(1))
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end - pd.Timedelta(days=1), freq="D")]


def download_agg_day(sym: str, date: str) -> tuple[Path, int]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zpath = DATA_DIR / f"{sym}-aggTrades-{date}.zip"
    if zpath.exists() and zpath.stat().st_size > 0:
        return zpath, zpath.stat().st_size
    r = requests.get(AGG_URL.format(s=sym, d=date), stream=True, timeout=180)
    r.raise_for_status()
    with open(zpath, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    return zpath, zpath.stat().st_size


def load_agg(zpath: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zpath) as z:
        raw = z.read([n for n in z.namelist() if n.endswith(".csv")][0])
    first = raw[:200].split(b"\n", 1)[0].decode("utf-8", "replace").lower()
    has_header = "transact_time" in first or "agg_trade_id" in first
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None,
                     names=None if has_header else COLS, usecols=range(7))
    df.columns = COLS
    if df["is_buyer_maker"].dtype == object:
        df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).str.strip().str.lower().map(
            {"true": True, "false": False, "1": True, "0": False})
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)
    tmax = int(df["transact_time"].max())
    unit = "ns" if tmax > 1e17 else "us" if tmax > 1e14 else "ms"
    df["ts"] = pd.to_datetime(df["transact_time"], unit=unit, utc=True)
    return df.set_index("ts").sort_index()


def grid_from_df(df: pd.DataFrame) -> pd.DataFrame:
    price = df["price"].resample("1s").last()
    sell = df["quantity"].where(df["is_buyer_maker"], 0.0).resample("1s").sum()
    buy = df["quantity"].where(~df["is_buyer_maker"], 0.0).resample("1s").sum()
    return pd.DataFrame({"price": price, "sell_vol": sell, "buy_vol": buy})


def load_funding(sym: str, months: list[str]) -> pd.DataFrame:
    frames = []
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
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["calc_time"] = pd.to_datetime(d["calc_time"], unit="ms", utc=True)
    d["last_funding_rate"] = d["last_funding_rate"].astype(float)
    return d.sort_values("calc_time")[["calc_time", "last_funding_rate"]].reset_index(drop=True)


def detect(grid: pd.DataFrame, drop_pct: float, vol_z: float,
           vel_win=15, base_win=3600, base_min=600, cooldown_s=300) -> pd.DataFrame:
    price, sv = grid["price"], grid["sell_vol"]
    ret_win = price.pct_change(vel_win)
    cur_rate = sv.rolling(vel_win).sum() / vel_win
    base = sv.shift(vel_win + 1)
    bmean = base.rolling(base_win, min_periods=base_min).mean()
    bstd = base.rolling(base_win, min_periods=base_min).std()
    z = (cur_rate - bmean) / bstd
    mask = (ret_win <= -drop_pct) & (z >= vol_z) & bstd.notna() & (bstd > 0)
    cand = grid.index[mask.fillna(False)]
    out, last = [], None
    for t in cand:
        if last is None or (t - last).total_seconds() >= cooldown_s:
            out.append({"trigger_ts": t, "trigger_price": float(price.loc[t]),
                        "ret_15s": float(ret_win.loc[t]), "vol_z": float(z.loc[t])})
            last = t
    return pd.DataFrame(out)


def exhaustion_entry(price: pd.Series, ret15: pd.Series, triggers: pd.DataFrame, max_wait=60) -> pd.DataFrame:
    ent_ts, ent_px, waits = [], [], []
    for t in triggers["trigger_ts"]:
        t1 = t + pd.Timedelta(seconds=max_wait)
        seg = ret15.loc[t + pd.Timedelta(seconds=1): t1]
        hit = seg[seg >= 0]
        et = hit.index[0] if len(hit) else (seg.index[-1] if len(seg) else t)
        ent_ts.append(et)
        ent_px.append(float(price.loc[et]) if et in price.index else float(price.loc[t]))
        waits.append(int((et - t).total_seconds()))
    triggers = triggers.copy()
    triggers["entry_ts"] = ent_ts
    triggers["entry_price"] = ent_px
    triggers["wait_s"] = waits
    return triggers


def annotate_funding(triggers: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    tr = triggers.sort_values("trigger_ts").reset_index(drop=True)
    merged = pd.merge_asof(tr, funding, left_on="trigger_ts", right_on="calc_time", direction="backward")
    merged = merged.rename(columns={"last_funding_rate": "funding_at_signal"})
    def buck(f):
        if pd.isna(f): return "na"
        if f <= 0: return "neg"
        if f <= 5e-4: return "mid"
        return "hi"
    merged["bucket"] = merged["funding_at_signal"].map(buck)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--month", type=int, default=3)
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    ap.add_argument("--drop-pct", type=float, default=0.003)
    ap.add_argument("--vol-z", type=float, default=3.0)
    ap.add_argument("--keep-raw", action="store_true", help="no borrar zips (debug)")
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MFE SMOKE DE CALIBRACION (mecanica + conteo). NO supervivencia. $0.")
    print(f"  mes={args.year}-{args.month:02d}  symbols={args.symbols}")
    print(f"  firma CONGELADA: caida>={args.drop_pct*100:.2f}%/15s & vol z>={args.vol_z}")
    print("=" * 80)

    days = days_in_month(args.year, args.month)
    fmonths = [f"{args.year}-{m:02d}" for m in (args.month - 1, args.month, args.month + 1) if 1 <= m <= 12]
    report = {"mes": f"{args.year}-{args.month:02d}", "params": vars(args), "por_simbolo": {}}
    t_global = time.time()

    for sym in args.symbols:
        print(f"\n[{sym}] descargando+procesando {len(days)} dias (cleanup dia-a-dia)...")
        grids, bytes_total, t_dl, t_proc = [], 0, 0.0, 0.0
        for d in days:
            t0 = time.time()
            try:
                zp, nb = download_agg_day(sym, d)
            except Exception as e:
                print(f"   {d}: SKIP descarga ({e})"); continue
            t_dl += time.time() - t0
            bytes_total += nb
            t1 = time.time()
            try:
                g = grid_from_df(load_agg(zp))
                grids.append(g)
            except Exception as e:
                print(f"   {d}: SKIP parseo ({e})")
            t_proc += time.time() - t1
            if not args.keep_raw:
                try: zp.unlink()
                except OSError: pass
        # serie continua del mes (1s), reindex a rango completo
        gm = pd.concat(grids).sort_index()
        full = pd.date_range(gm.index[0].floor("D"), gm.index[-1].ceil("D"), freq="1s", tz="UTC")[:-1]
        gm = gm.reindex(full)
        gm["price"] = gm["price"].ffill()
        gm[["sell_vol", "buy_vol"]] = gm[["sell_vol", "buy_vol"]].fillna(0.0)

        triggers = detect(gm, args.drop_pct, args.vol_z)
        ret15 = gm["price"].pct_change(15)
        triggers = exhaustion_entry(gm["price"], ret15, triggers)
        funding = load_funding(sym, fmonths)
        ann = annotate_funding(triggers, funding)

        counts = ann["bucket"].value_counts().to_dict()
        gb = bytes_total / 1e9
        # proyeccion bucket hi a 2 anios via ratio de ventanas >5e-4
        hi_mar = counts.get("hi", 0)
        scale = HI_WINDOWS_2YR.get(sym, 1) / max(HI_WINDOWS_MAR2024.get(sym, 1), 1)
        hi_proj = hi_mar * scale
        rep = {
            "n_cascadas": int(len(ann)),
            "conteo_por_bucket": {b: int(counts.get(b, 0)) for b in ("neg", "mid", "hi", "na")},
            "bucket_hi_marzo": int(hi_mar),
            "bucket_hi_proyeccion_2anios": round(float(hi_proj), 1),
            "coste": {"gb_descargados": round(gb, 2), "gb_por_dia": round(gb / len(days), 3),
                      "seg_descarga": round(t_dl, 1), "seg_parseo": round(t_proc, 1),
                      "seg_por_dia": round((t_dl + t_proc) / len(days), 1)},
            "wait_s_mediano": int(ann["wait_s"].median()) if len(ann) else None,
        }
        report["por_simbolo"][sym] = rep
        ann.to_csv(RESULTS_DIR / f"smoke_cascadas_{sym}_{args.year}-{args.month:02d}.csv", index=False)

        print(f"\n  >>> CONTEO POR BUCKET (output primario, SIN supervivencia) <<<")
        print(f"      neg (<=0)   : {rep['conteo_por_bucket']['neg']}")
        print(f"      mid (0-5e-4): {rep['conteo_por_bucket']['mid']}")
        print(f"      hi  (>5e-4) : {rep['conteo_por_bucket']['hi']}   "
              f"[proyeccion 2 anios ~{rep['bucket_hi_proyeccion_2anios']} via ratio ventanas]")
        print(f"      total cascadas: {rep['n_cascadas']}  | wait exhaustion mediano: {rep['wait_s_mediano']}s")
        print(f"  coste: {gb:.2f} GB ({rep['coste']['gb_por_dia']} GB/dia), "
              f"{rep['coste']['seg_por_dia']}s/dia (dl {t_dl:.0f}s + parse {t_proc:.0f}s)")

    # proyeccion run completo
    tot_gbday = np.mean([r["coste"]["gb_por_dia"] for r in report["por_simbolo"].values()])
    tot_sday = np.mean([r["coste"]["seg_por_dia"] for r in report["por_simbolo"].values()])
    ndays_run = 731 * len(args.symbols)
    report["proyeccion_run_2anios"] = {
        "dias_x_simbolo": 731, "celdas": ndays_run,
        "GB_transfer_total_aprox": round(tot_gbday * ndays_run, 1),
        "horas_wall_aprox_sin_supervivencia": round(tot_sday * ndays_run / 3600, 1),
        "nota": "supervivencia+control 10x anaden coste extra (raw 3s slippage); recalibrar. Factor 5-10x.",
    }
    (RESULTS_DIR / f"smoke_summary_{args.year}-{args.month:02d}.json").write_text(
        json.dumps(report, indent=2, default=str))

    print("\n" + "=" * 80)
    print("PROYECCION RUN 2 ANIOS (sin supervivencia, solo deteccion):")
    pr = report["proyeccion_run_2anios"]
    print(f"  ~{pr['GB_transfer_total_aprox']} GB transfer (dia-a-dia, pico disco <1-2GB con cleanup)")
    print(f"  ~{pr['horas_wall_aprox_sin_supervivencia']}h wall (solo deteccion); +supervivencia+control 10x.")
    print(f"  tiempo total smoke: {(time.time()-t_global)/60:.1f} min")
    print("=" * 80)
    print("RECORDATORIO: smoke = mecanica + conteo. La supervivencia es el run de 2 anios,")
    print("post-decision sobre el conteo. El bucket hi decide test PRIMARIO vs SECUNDARIO.")


if __name__ == "__main__":
    main()
