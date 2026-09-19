#!/usr/bin/env python3
"""
sandbox_mfe.py — MFE (Mid-Frequency Liquidation Exhaustion) — FASE 0: prueba de MECANICA.

=============================================================================
AISLADO. NO toca produccion (brain/portfolio/kernel/Numba/walk-forward).
Entorno analitico CPU puro: pandas + numpy + requests. $0 capital.
=============================================================================

OBJETIVO FASE 0: validar que la TUBERIA funciona
    (1) descarga de aggTrades historicos de data.binance.vision,
    (2) deteccion de la "Firma de Cascada",
    (3) medicion del rebote a 1/3/5 min.
NO concluir NADA sobre EDGE.

----------------------------------------------------------------------------
BLINDAJE 1 (CARDINAL): el dia usado es EXTREMO (flash crash) = MEJOR CASO
  ABSOLUTO. El rebote observado NO es representativo. PROHIBIDO sacar
  conclusiones de edge del dia extremo. Aqui solo se confirma que el codigo
  detecta y mide. El edge se decide DESPUES (periodo continuo, slippage
  pesimista, tasa de supervivencia), en una fase aparte pre-registrada.

BLINDAJE 2 (CARDINAL): el cherry-picking de "meses destructivos" sesga el
  experimento. La FASE DE EDGE (posterior) usara la firma OBJETIVA sobre un
  PERIODO CONTINUO largo (todas las cascadas, las que revirtieron y las que
  no). Si se necesitan mas eventos -> ALARGAR el periodo, NO seleccionar los
  violentos. Esta Fase 0 usa un dia extremo deliberadamente y SOLO para
  garantizar que hay eventos que detectar.
----------------------------------------------------------------------------

FIRMA DE CASCADA (en BACKTEST historico):
  - Velocidad:  caida >= X% en una ventana de <= 15 s.
  - Volumen direccional: volumen vendedor-agresor (is_buyer_maker=True, el
    taker vendio contra el bid) en la ventana de 15 s, expresado como tasa
    por segundo, >= media + Z * desv.est. de la tasa de la ULTIMA HORA.
  - forceOrder (pings de liquidacion): VERIFICADO primary-source 2026-06-29
    que data.binance.vision NO publica liquidaciones historicas
    (futures/um/daily = aggTrades, bookDepth, bookTicker, *Klines, metrics,
    trades; sin liquidationSnapshot/forceOrder). => En BACKTEST la firma se
    reduce a velocidad + volumen. La confirmacion forceOrder solo existe EN
    VIVO (WebSocket), por tanto solo entra en la fase de paper-trading.

Uso:
    python sandbox_mfe.py --symbol BTCUSDT --date 2024-08-05 --market futures-um
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"

BASE_URLS = {
    "futures-um": "https://data.binance.vision/data/futures/um/daily/aggTrades/{sym}/{sym}-aggTrades-{date}.zip",
    "spot": "https://data.binance.vision/data/spot/daily/aggTrades/{sym}/{sym}-aggTrades-{date}.zip",
}

# aggTrades columnas canonicas (futures um = 7; spot = 8 con is_best_match al final)
COLS = ["agg_trade_id", "price", "quantity", "first_trade_id",
        "last_trade_id", "transact_time", "is_buyer_maker"]


# --------------------------------------------------------------------------
# 1) DESCARGA
# --------------------------------------------------------------------------
def download_day(symbol: str, date: str, market: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = BASE_URLS[market].format(sym=symbol, date=date)
    zpath = DATA_DIR / f"{symbol}-aggTrades-{date}.zip"
    if zpath.exists() and zpath.stat().st_size > 0:
        print(f"[download] ya presente: {zpath.name} ({zpath.stat().st_size/1e6:.1f} MB)")
        return zpath
    print(f"[download] {url}")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(zpath, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    print(f"[download] OK {zpath.name} ({zpath.stat().st_size/1e6:.1f} MB)")
    return zpath


# --------------------------------------------------------------------------
# 2) CARGA + NORMALIZACION (robusta a header / unidad de timestamp)
# --------------------------------------------------------------------------
def load_aggtrades(zpath: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zpath) as z:
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        raw = z.read(name)
    first = raw[:200].split(b"\n", 1)[0].decode("utf-8", "replace").lower()
    has_header = "agg_trade_id" in first or "transact_time" in first
    df = pd.read_csv(
        io.BytesIO(raw),
        header=0 if has_header else None,
        names=None if has_header else COLS + (["is_best_match"] if first.count(",") >= 7 else []),
        usecols=range(7),
    )
    df.columns = COLS  # forzamos nombres canonicos sobre las 7 primeras
    # is_buyer_maker -> bool (true/false strings o 0/1)
    if df["is_buyer_maker"].dtype == object:
        df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).str.strip().str.lower().map(
            {"true": True, "false": False, "1": True, "0": False})
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)
    # autodeteccion de unidad de timestamp (ms vs us vs ns)
    tmax = int(df["transact_time"].max())
    if tmax > 1e17:
        unit = "ns"
    elif tmax > 1e14:
        unit = "us"
    else:
        unit = "ms"
    df["ts"] = pd.to_datetime(df["transact_time"], unit=unit, utc=True)
    df = df.sort_values("ts").set_index("ts")
    print(f"[load] {len(df):,} aggTrades | unidad={unit} | "
          f"{df.index[0]} -> {df.index[-1]} | precio {df['price'].iloc[0]:.1f} -> {df['price'].iloc[-1]:.1f}")
    return df


# --------------------------------------------------------------------------
# 3) GRID 1s + FIRMA DE CASCADA
# --------------------------------------------------------------------------
def build_grid(df: pd.DataFrame) -> pd.DataFrame:
    price_1s = df["price"].resample("1s").last().ffill()
    sell = df["quantity"].where(df["is_buyer_maker"], 0.0)   # taker vendio (bajista)
    buy = df["quantity"].where(~df["is_buyer_maker"], 0.0)   # taker compro (alcista)
    g = pd.DataFrame({
        "price": price_1s,
        "sell_vol": sell.resample("1s").sum().reindex(price_1s.index, fill_value=0.0),
        "buy_vol": buy.resample("1s").sum().reindex(price_1s.index, fill_value=0.0),
    })
    return g


def detect_cascades(g: pd.DataFrame, drop_pct: float, vol_z: float,
                    vel_win: int = 15, base_win: int = 3600, base_min: int = 600,
                    cooldown_s: int = 300) -> pd.DataFrame:
    price = g["price"]
    sv = g["sell_vol"]

    # --- Velocidad: caida en ventana de vel_win segundos ---
    ret_win = price.pct_change(vel_win)
    vel_cond = ret_win <= -drop_pct

    # --- Volumen direccional: tasa por segundo en la ventana vs baseline ultima hora ---
    cur_rate = sv.rolling(vel_win).sum() / vel_win               # tasa/seg en la ventana actual
    base = sv.shift(vel_win + 1)                                 # excluye la ventana actual
    bmean = base.rolling(base_win, min_periods=base_min).mean()
    bstd = base.rolling(base_win, min_periods=base_min).std()
    z = (cur_rate - bmean) / bstd
    vol_cond = (z >= vol_z) & bstd.notna() & (bstd > 0)

    mask = vel_cond & vol_cond
    cand = g.index[mask.fillna(False)]

    # --- Clustering con cooldown (un evento por cascada) ---
    events = []
    last = None
    for t in cand:
        if last is None or (t - last).total_seconds() >= cooldown_s:
            events.append(t)
            last = t

    rows = []
    for t in events:
        rows.append({
            "ts": t,
            "price": float(price.loc[t]),
            "ret_15s": float(ret_win.loc[t]),
            "sell_rate_per_s": float(cur_rate.loc[t]),
            "vol_zscore": float(z.loc[t]),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4) MEDICION DE REBOTE (sin look-ahead: entrada en t0, mido solo hacia delante)
#    Convencion: FADE de la cascada bajista = LONG en la exhaustion.
#    rebote positivo = el precio sube tras el evento.
# --------------------------------------------------------------------------
def measure_rebound(g: pd.DataFrame, events: pd.DataFrame,
                    horizons_s=(60, 180, 300)) -> pd.DataFrame:
    price = g["price"]
    idx = price.index
    out = events.copy()
    for H in horizons_s:
        rets_end, mfe, mae = [], [], []
        for t0, p0 in zip(events["ts"], events["price"]):
            t1 = t0 + pd.Timedelta(seconds=H)
            win = price.loc[t0:t1]
            if len(win) < 2:
                rets_end.append(np.nan); mfe.append(np.nan); mae.append(np.nan)
                continue
            p_end = win.iloc[-1]
            rets_end.append(p_end / p0 - 1.0)        # rebote "a" H minutos
            mfe.append(win.max() / p0 - 1.0)         # max excursion favorable (mejor rebote)
            mae.append(win.min() / p0 - 1.0)         # max excursion adversa (caida adicional)
        m = H // 60
        out[f"ret_{m}m"] = rets_end
        out[f"mfe_{m}m"] = mfe
        out[f"mae_{m}m"] = mae
    return out


def summarize(ev: pd.DataFrame, horizons_s=(60, 180, 300)) -> dict:
    summ = {"n_events": int(len(ev))}
    per_h = {}
    for H in horizons_s:
        m = H // 60
        r = ev[f"ret_{m}m"].dropna()
        f = ev[f"mfe_{m}m"].dropna()
        a = ev[f"mae_{m}m"].dropna()
        per_h[f"{m}min"] = {
            "n": int(len(r)),
            "rebote_medio_ret": round(float(r.mean()), 5) if len(r) else None,
            "rebote_mediano_ret": round(float(r.median()), 5) if len(r) else None,
            "rebote_max_mfe": round(float(f.max()), 5) if len(f) else None,
            "rebote_medio_mfe": round(float(f.mean()), 5) if len(f) else None,
            "caida_max_adicional_mae": round(float(a.min()), 5) if len(a) else None,
            "pct_ret_positivo": round(float((r > 0).mean()), 4) if len(r) else None,
        }
    summ["por_horizonte"] = per_h
    return summ


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="MFE Fase 0 — prueba de mecanica (sandbox aislado)")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--date", default="2024-08-05", help="YYYY-MM-DD (dia extremo para Fase 0)")
    ap.add_argument("--market", default="futures-um", choices=list(BASE_URLS))
    ap.add_argument("--drop-pct", type=float, default=0.003, help="caida minima en 15s (0.003 = 0.3%%)")
    ap.add_argument("--vol-z", type=float, default=3.0, help="z-score minimo de volumen vendedor")
    ap.add_argument("--cooldown", type=int, default=300, help="segundos entre eventos (dedupe)")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("MFE FASE 0 — PRUEBA DE MECANICA (NO de edge). Sandbox aislado, $0 capital.")
    print(f"  symbol={args.symbol} date={args.date} market={args.market}")
    print(f"  firma: caida>={args.drop_pct*100:.2f}%/15s  &  vol vendedor z>={args.vol_z}")
    print("=" * 78)

    zpath = download_day(args.symbol, args.date, args.market)
    df = load_aggtrades(zpath)
    g = build_grid(df)
    ev = detect_cascades(g, drop_pct=args.drop_pct, vol_z=args.vol_z, cooldown_s=args.cooldown)

    print(f"\n[deteccion] {len(ev)} cascadas detectadas")
    if len(ev) == 0:
        print("  (cero eventos con estos umbrales — la maquinaria corre pero el dia/umbral no dispara)")
        return

    ev = measure_rebound(g, ev)
    summ = summarize(ev)

    # --- Reporte de eventos ---
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 30)
    show = ev.copy()
    show["ts"] = show["ts"].dt.strftime("%H:%M:%S")
    cols = ["ts", "price", "ret_15s", "vol_zscore",
            "ret_1m", "mfe_1m", "mae_1m", "ret_3m", "mfe_3m", "ret_5m", "mfe_5m"]
    print("\n--- EVENTOS (rebote como FADE long; ret/mfe/mae en fraccion) ---")
    with pd.option_context("display.float_format", lambda v: f"{v:+.4f}"):
        print(show[cols].to_string(index=False))

    print("\n--- RESUMEN POR HORIZONTE ---")
    for m, d in summ["por_horizonte"].items():
        print(f"  {m:>5}: N={d['n']:>3}  rebote_medio(ret)={d['rebote_medio_ret']:+.4f}  "
              f"mediano={d['rebote_mediano_ret']:+.4f}  max(mfe)={d['rebote_max_mfe']:+.4f}  "
              f"medio(mfe)={d['rebote_medio_mfe']:+.4f}  %ret>0={d['pct_ret_positivo']:.2f}  "
              f"peor_mae={d['caida_max_adicional_mae']:+.4f}")

    # --- Persistencia ---
    stem = f"{args.symbol}_{args.date}_drop{args.drop_pct}_z{args.vol_z}"
    ev_path = RESULTS_DIR / f"events_{stem}.csv"
    ev.to_csv(ev_path, index=False)
    summ_path = RESULTS_DIR / f"summary_{stem}.json"
    summ_meta = {
        "fase": "0_mecanica",
        "AVISO": "DIA EXTREMO = MEJOR CASO. PROHIBIDO concluir edge. Solo valida la tuberia.",
        "params": vars(args),
        "resumen": summ,
    }
    summ_path.write_text(json.dumps(summ_meta, indent=2, default=str))
    print(f"\n[out] {ev_path.name}\n[out] {summ_path.name}")
    print("\n" + "=" * 78)
    print("RECORDATORIO BLINDAJE 1: dia extremo = mejor caso absoluto, NO representativo.")
    print("La forma del rebote aqui NO informa edge. Edge = fase posterior con periodo")
    print("continuo, slippage pesimista y tasa de supervivencia. Mecanica validada arriba.")
    print("=" * 78)


if __name__ == "__main__":
    main()
