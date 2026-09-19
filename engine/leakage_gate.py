# -*- coding: utf-8 -*-
"""
E2-full LEAKAGE GATE (ajuste #6, la pieza más crítica de la fase).
Verifica primary-source que un run as-of NO vio ningún dato > ancla en NINGÚN paso,
y que el holdout no cruza la frontera del mes del audit (A2). Un leakage no cazado
convertiría E2-full en un espejismo con sello de holdout.

Checks (todos por timestamp, no por nombre):
  L1. parquet truncado: max(timestamp) < ancla.
  L2. GMM joblib: training_date_range[1] <= ancla.
  L3. ventana servida a lab_lite (si se registró): max-ts < ancla.
  L4. specialist JSON 'generated': posterior al ancla en wall-clock (se genera HOY) — OK;
      pero su contenido depende solo de datos <= ancla (asegurado por L1/L2).
  H1. holdout: todo trade en [ancla, fin_holdout); para A2, fin_holdout <= 2026-05-17 (excluye mes audit).

Uso:
  - importable: assert_asof(...), assert_holdout(...)
  - CLI self-test bidireccional: python leakage_gate.py --selftest  (honesto PASA + envenenado FALLA)
Exit 0 = limpio; 1 = leakage cazado.
"""
import sys, json, argparse
from datetime import datetime, timezone
import pandas as pd
import joblib

AUDIT_MONTH_START = pd.Timestamp("2026-05-17 00:00:00", tz="UTC")  # frontera E2-lite / mes audit


def _max_ts_parquet(path):
    df = pd.read_parquet(path)
    col = "timestamp_ms" if "timestamp_ms" in df.columns else "timestamp"
    ts = pd.to_datetime(df[col], unit="ms", utc=True) if col == "timestamp_ms" else pd.to_datetime(df[col], utc=True)
    return ts.max()


def assert_asof(anchor, parquet_path=None, gmm_path=None, served_max_ts=None, label=""):
    """Devuelve (ok, errors). anchor: pd.Timestamp tz-aware."""
    errs = []
    A = pd.Timestamp(anchor) if not isinstance(anchor, pd.Timestamp) else anchor
    if A.tz is None:
        A = A.tz_localize("UTC")
    if parquet_path:
        mx = _max_ts_parquet(parquet_path)
        if mx >= A:
            errs.append(f"L1 LEAK [{label}]: parquet max-ts {mx} >= ancla {A}")
    if gmm_path:
        m = joblib.load(gmm_path)
        rng = m.get("training_date_range")
        if rng:
            cutoff = pd.Timestamp(rng[1], tz="UTC")
            if cutoff > A:
                errs.append(f"L2 LEAK [{label}]: GMM training cutoff {cutoff} > ancla {A}")
        else:
            errs.append(f"L2 [{label}]: GMM sin training_date_range (no verificable)")
    if served_max_ts is not None:
        s = pd.Timestamp(served_max_ts)
        if s.tz is None:
            s = s.tz_localize("UTC")
        if s >= A:
            errs.append(f"L3 LEAK [{label}]: ventana lab_lite servida max-ts {s} >= ancla {A}")
    return (len(errs) == 0, errs)


def assert_holdout(anchor, holdout_end, trades_ts, exclude_audit_month=False, label=""):
    """trades_ts: iterable de pd.Timestamp de entradas del holdout."""
    errs = []
    A = pd.Timestamp(anchor).tz_localize("UTC") if pd.Timestamp(anchor).tz is None else pd.Timestamp(anchor)
    E = pd.Timestamp(holdout_end).tz_localize("UTC") if pd.Timestamp(holdout_end).tz is None else pd.Timestamp(holdout_end)
    if exclude_audit_month and E > AUDIT_MONTH_START:
        errs.append(f"H1 [{label}]: fin_holdout {E} cruza frontera mes-audit {AUDIT_MONTH_START} — debe cortar antes")
    for t in trades_ts:
        t = pd.Timestamp(t)
        if t.tz is None:
            t = t.tz_localize("UTC")
        if t < A or t >= E:
            errs.append(f"H1 [{label}]: trade holdout {t} fuera de [{A}, {E})")
        if exclude_audit_month and t >= AUDIT_MONTH_START:
            errs.append(f"H1 [{label}]: trade holdout {t} dentro del mes audit (>= {AUDIT_MONTH_START})")
    return (len(errs) == 0, errs)


def _selftest():
    """Validación BIDIRECCIONAL (ajuste #6): honesto PASA + envenenado FALLA ruidoso."""
    import numpy as np, tempfile, os
    A = pd.Timestamp("2026-02-01 00:00:00", tz="UTC")
    tmp = tempfile.mkdtemp()
    # --- honesto: parquet termina A-1h, GMM cutoff A-1h ---
    n = 1000
    ts_ok = [int((A - pd.Timedelta(hours=n - i)).timestamp() * 1000) for i in range(n)]
    pd.DataFrame({"timestamp_ms": ts_ok, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}).to_parquet(os.path.join(tmp, "ok.parquet"))
    gmm_ok = {"gmm": None, "scaler": None, "training_date_range": ["2018-01-01 00:00:00", str((A - pd.Timedelta(hours=1)).tz_localize(None))]}
    joblib.dump(gmm_ok, os.path.join(tmp, "ok.joblib"))
    ok1, e1 = assert_asof(A, os.path.join(tmp, "ok.parquet"), os.path.join(tmp, "ok.joblib"), served_max_ts=A - pd.Timedelta(hours=1), label="honesto")
    h1ok, h1e = assert_holdout(A, pd.Timestamp("2026-05-16 00:00:00", tz="UTC"),
                               [A + pd.Timedelta(days=10), A + pd.Timedelta(days=50)], exclude_audit_month=True, label="honesto")

    # --- envenenado: parquet con 1 barra A+1h, GMM cutoff A+10d, holdout cruza mes-audit ---
    ts_bad = ts_ok + [int((A + pd.Timedelta(hours=1)).timestamp() * 1000)]
    pd.DataFrame({"timestamp_ms": ts_bad, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}).to_parquet(os.path.join(tmp, "bad.parquet"))
    gmm_bad = {"training_date_range": ["2018-01-01 00:00:00", str((A + pd.Timedelta(days=10)).tz_localize(None))]}
    joblib.dump(gmm_bad, os.path.join(tmp, "bad.joblib"))
    ok2, e2 = assert_asof(A, os.path.join(tmp, "bad.parquet"), os.path.join(tmp, "bad.joblib"), served_max_ts=A + pd.Timedelta(hours=5), label="envenenado")
    h2ok, h2e = assert_holdout(A, pd.Timestamp("2026-06-01 00:00:00", tz="UTC"),
                               [A + pd.Timedelta(days=120)], exclude_audit_month=True, label="envenenado")

    print("=== VALIDACIÓN BIDIRECCIONAL DEL LEAKAGE GATE ===")
    print(f"[honesto]   asof PASA={ok1} (errs={len(e1)})  holdout PASA={h1ok} (errs={len(h1e)})")
    print(f"[envenenado] asof CAZADO={not ok2} ({len(e2)} leaks)  holdout CAZADO={not h2ok} ({len(h2e)} leaks)")
    for e in e2 + h2e:
        print("   →", e)
    passed = ok1 and h1ok and (not ok2) and (not h2ok)
    print(f"\nGATE BIDIRECCIONAL: {'✅ VÁLIDO (honesto pasa + envenenado falla)' if passed else '❌ INVÁLIDO'}")
    return 0 if passed else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    if args.selftest:
        sys.exit(_selftest())
    print("Uso: --selftest, o importar assert_asof/assert_holdout")
