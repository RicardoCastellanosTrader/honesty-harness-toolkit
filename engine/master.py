#!/usr/bin/env python3
"""
master.py — Script maestro de producción

Punto de entrada único que ejecuta todo el pipeline:
  1. Descarga de históricos
  2. Entrenamiento de modelos de régimen (GMM)
  3. Generación de presets (lab_lite)
  4. Regime walk-forward validation

Configuración centralizada en CONFIG para evitar parámetros inconsistentes.

Uso:
    python master.py                                  # Todo, 45 símbolos
    python master.py --symbols BTC/USDT ETH/USDT      # Solo estos
    python master.py --recycle                         # Fuerza re-entrenar y re-generar todo
    python master.py --skip-download                   # Salta descarga (datos ya actualizados)
    python master.py --from-step regime-wf             # Desde regime walk-forward
    python master.py --from-step lite                  # Desde lab_lite (modelos ya entrenados)
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime, timedelta

# Fix Windows cp1252 encoding
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith('cp'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ============================================
# CONFIGURACIÓN CENTRALIZADA
# ============================================

CONFIG = {
    # --- Símbolos (45, fuente única — live/ y backend importan de aquí) ---
    # Excluidos: TON, PEPE, BONK (no disponibles en BingX Futures swap)
    'symbols': [
        'BNB/USDT', 'BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'SOL/USDT',
        'TRX/USDT', 'DOGE/USDT', 'ADA/USDT', 'BCH/USDT', 'LINK/USDT',
        'XLM/USDT', 'SUI/USDT', 'LTC/USDT', 'AVAX/USDT', 'HBAR/USDT',
        'SHIB/USDT', 'DOT/USDT', 'UNI/USDT', 'TAO/USDT',
        'AAVE/USDT', 'NEAR/USDT', 'ICP/USDT', 'ETC/USDT',
        'ONDO/USDT', 'ENA/USDT', 'POL/USDT', 'WLD/USDT', 'APT/USDT',
        'ATOM/USDT', 'ALGO/USDT', 'RENDER/USDT', 'ARB/USDT', 'FIL/USDT',
        'QNT/USDT', 'VET/USDT', 'SEI/USDT', 'OP/USDT',
        'IMX/USDT', 'INJ/USDT', 'FET/USDT', 'STX/USDT', 'SAND/USDT',
        'MANA/USDT', 'GRT/USDT', 'THETA/USDT',
    ],

    # --- Exchange ---
    'exchange': 'binance',
    'timeframe': '1h',

    # --- Simulación (parámetros de lab_historico_numba_v8_3) ---
    'sl_percent': 3.0,
    'sl_emergency_percent': 5.0,
    'ts_percent': 0.5,
    'cooldown_bars': 1,
    'commission_round_trip': 0.10,

    # --- Lab lite ---
    'total_candles': 5000,
    'top_n_presets': 7,             # presets globales por símbolo
    'diversity_mode': True,
    'cluster_min_trades': 30,       # mínimo trades por cluster
    'cluster_top_n_per': 3,         # top N presets por cluster
    'cluster_min_pf': 1.2,         # PF mínimo para especialista de cluster

    # --- Clustering / Régimen (regime_features.py) ---
    'regime_lookback_short': 100,
    'regime_lookback_long': 1000,
    'regime_atr_period': 14,
    'regime_max_k': 3,

    # --- Regime walk-forward ---
    'toxic_tail': 'dynamic',
    'confirm_threshold': 0.75,
    'max_toxic_tail': 100,
    'min_toxic_tail': 5,
    'train_ratio': 0.70,
    'min_episode_bars': 50,
    'min_trades_cluster': 30,
    'min_episodes_validate': 5,
    'top_train': 30000,

    # Filtros internos de regime_wf
    'train_min_trades': 30,
    'train_min_pf': 1.2,
    'fwd_min_trades': 15,
    'fwd_min_pf': 1.0,
    'top_keep_ram': 50000,
    'top_json': 100,

    # Cross-cluster survival
    'cross_cluster_pf_min': 0.7,
    'cross_cluster_p_min': 0.10,

    # --- Directorios ---
    'data_cache_dir': 'data_cache',
    'regime_models_dir': 'regime_models',
    'output_dir': 'output/production',
    'regime_wf_dir': 'regime_wf',

    # --- Data freshness (horas) ---
    'data_max_age_hours': 24,
}

# Nombres de pasos válidos para --from-step
STEP_NAMES = ['download', 'train', 'lite', 'regime-wf']
STEP_ORDER = {name: i for i, name in enumerate(STEP_NAMES)}


# ============================================
# HELPERS
# ============================================

def sym_key(symbol):
    """BTC/USDT → BTC"""
    return symbol.replace("/USDT", "").replace("/", "")


def sym_clean(symbol):
    """BTC/USDT → BTCUSDT"""
    return symbol.replace("/", "")


def parquet_path(symbol):
    return os.path.join(CONFIG['data_cache_dir'], f"{sym_clean(symbol)}_{CONFIG['timeframe']}.parquet")


def model_path(symbol):
    return os.path.join(CONFIG['regime_models_dir'], f"{sym_key(symbol)}_regime.joblib")


def presets_path(symbol):
    sc = sym_clean(symbol)
    return os.path.join(CONFIG['output_dir'], f"presets_{sc}.csv")


def specialist_json_path(symbol):
    sc = sym_clean(symbol)
    return os.path.join(CONFIG['regime_wf_dir'], f"{sc}_specialist_configs.json")


def parquet_is_fresh(symbol, max_age_hours):
    """True si el parquet existe y su última modificación es < max_age_hours."""
    path = parquet_path(symbol)
    if not os.path.exists(path):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    return (datetime.now() - mtime) < timedelta(hours=max_age_hours)


def format_duration(seconds):
    """Formatea duración en formato legible."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    parts = []
    if d: parts.append(f"{d}d")
    if h: parts.append(f"{h}h")
    if m: parts.append(f"{m}m")
    if s and not d: parts.append(f"{s}s")
    return " ".join(parts)


def check_prerequisites(symbol, step):
    """Verifica prerrequisitos para un paso. Retorna (ok, msg)."""
    if step == 'train':
        p = parquet_path(symbol)
        if not os.path.exists(p):
            return False, f"falta parquet ({p}). Ejecuta paso 1 (download) primero o usa --from-step download"
    elif step == 'lite':
        p = model_path(symbol)
        if not os.path.exists(p):
            return False, f"falta regime model ({p}). Ejecuta paso 2 (train) primero o usa --from-step train"
        p2 = parquet_path(symbol)
        if not os.path.exists(p2):
            return False, f"falta parquet ({p2}). Ejecuta paso 1 (download) primero"
    elif step == 'regime-wf':
        p = presets_path(symbol)
        if not os.path.exists(p):
            return False, f"falta presets ({p}). Ejecuta paso 3 (lite) primero o usa --from-step lite"
        p2 = model_path(symbol)
        if not os.path.exists(p2):
            return False, f"falta regime model ({p2}). Ejecuta paso 2 (train) primero"
        p3 = parquet_path(symbol)
        if not os.path.exists(p3):
            return False, f"falta parquet ({p3}). Ejecuta paso 1 (download) primero"
    return True, ""


# ============================================
# PASO 1: DESCARGA DE HISTÓRICOS
# ============================================

def step_download(symbols, recycle=False):
    """Descarga históricos de Binance para todos los símbolos."""
    print("\n" + "=" * 70)
    print("📥 PASO 1: DESCARGA DE HISTÓRICOS")
    print("=" * 70)

    import download_full_history as dlh

    # Sobreescribir globals del módulo
    dlh.CACHE_DIR = CONFIG['data_cache_dir']
    dlh.TIMEFRAME = CONFIG['timeframe']

    os.makedirs(CONFIG['data_cache_dir'], exist_ok=True)

    import ccxt
    exchange = ccxt.binance({'enableRateLimit': True})

    skipped = 0
    downloaded = 0
    failed = []

    for i, symbol in enumerate(symbols, 1):
        p = parquet_path(symbol)

        # Skip si datos son frescos y no --recycle
        if not recycle and parquet_is_fresh(symbol, CONFIG['data_max_age_hours']):
            print(f"   [{i}/{len(symbols)}] {symbol} — datos frescos, saltando")
            skipped += 1
            continue

        print(f"   [{i}/{len(symbols)}] {symbol} — descargando...")
        try:
            count = dlh.download_full(symbol, exchange)
            downloaded += 1
            print(f"      {count} velas totales")
        except Exception as e:
            print(f"      FAILED: {e}")
            failed.append(symbol)

    print(f"\n   Resultado: {downloaded} descargados, {skipped} saltados, {len(failed)} fallidos")
    if failed:
        print(f"   Fallidos: {', '.join(failed)}")

    # Verificar que todos tienen parquet
    missing = [s for s in symbols if not os.path.exists(parquet_path(s))]
    if missing:
        print(f"\n   ⚠️  Símbolos sin parquet: {', '.join(missing)}")

    return failed


# ============================================
# PASO 2: ENTRENAMIENTO DE MODELOS DE RÉGIMEN
# ============================================

def step_train(symbols, recycle=False):
    """Entrena modelos GMM de régimen para cada símbolo."""
    print("\n" + "=" * 70)
    print("🏋️ PASO 2: ENTRENAMIENTO DE MODELOS DE RÉGIMEN")
    print("=" * 70)

    import train_regime_model as trm
    import regime_features as rf

    # Sobreescribir globals
    trm.MODEL_DIR = CONFIG['regime_models_dir']
    trm.CACHE_DIR = CONFIG['data_cache_dir']
    trm.TIMEFRAME = CONFIG['timeframe']
    trm.LOOKBACK = CONFIG['regime_lookback_short']
    trm.MAX_K = CONFIG['regime_max_k']

    rf.LOOKBACK_SHORT = CONFIG['regime_lookback_short']
    rf.LOOKBACK_LONG = CONFIG['regime_lookback_long']
    rf.ATR_PERIOD = CONFIG['regime_atr_period']

    os.makedirs(CONFIG['regime_models_dir'], exist_ok=True)

    # Warm up sklearn (Windows)
    from sklearn.mixture import GaussianMixture
    import numpy as np
    GaussianMixture(n_components=2, n_init=1, max_iter=1, random_state=42).fit(
        np.random.randn(20, 3))

    skipped = 0
    trained = 0
    failed = []

    for i, symbol in enumerate(symbols, 1):
        # Verificar prerrequisito
        ok, msg = check_prerequisites(symbol, 'train')
        if not ok:
            print(f"   [{i}/{len(symbols)}] {symbol} — {msg}")
            failed.append(symbol)
            continue

        mp = model_path(symbol)
        if not recycle and os.path.exists(mp):
            print(f"   [{i}/{len(symbols)}] {symbol} — modelo existe, saltando")
            skipped += 1
            continue

        print(f"   [{i}/{len(symbols)}] {symbol} — entrenando...")
        try:
            df = trm.load_full_history(symbol)
            if df is None:
                print(f"      No se pudo cargar datos")
                failed.append(symbol)
                continue

            model_data = trm.train_symbol_model(symbol, df)
            if model_data is None:
                print(f"      Entrenamiento falló")
                failed.append(symbol)
                continue

            import joblib
            joblib.dump(model_data, mp)
            trained += 1
            n_clusters = model_data.get('n_clusters', '?')
            print(f"      {n_clusters} clusters → {mp}")
        except Exception as e:
            print(f"      FAILED: {e}")
            import traceback; traceback.print_exc()
            failed.append(symbol)

    print(f"\n   Resultado: {trained} entrenados, {skipped} saltados, {len(failed)} fallidos")
    if failed:
        print(f"   Fallidos: {', '.join(failed)}")

    # Verificar
    missing = [s for s in symbols if not os.path.exists(model_path(s))]
    if missing:
        print(f"\n   ⚠️  Símbolos sin modelo: {', '.join(missing)}")

    return failed


# ============================================
# PASO 3: GENERACIÓN DE PRESETS (LAB LITE)
# ============================================

def step_lite(symbols, recycle=False):
    """Genera presets usando lab_lite_zonas_v5e (solo fase lite, no lab_historico)."""
    print("\n" + "=" * 70)
    print("🧪 PASO 3: GENERACIÓN DE PRESETS (LAB LITE)")
    print("=" * 70)

    import lab_lite_zonas_v5e as lab_lite

    # Sobreescribir globals del módulo
    lab_lite.TOTAL_CANDLES = CONFIG['total_candles']
    lab_lite.TIMEFRAME = CONFIG['timeframe']
    lab_lite.COMMISSION_ROUND_TRIP = CONFIG['commission_round_trip']
    lab_lite.TOP_N_PRESETS = CONFIG['top_n_presets']
    lab_lite.DIVERSITY_MODE = CONFIG['diversity_mode']
    lab_lite.OUTPUT_DIR = CONFIG['output_dir']
    lab_lite.USE_CLUSTERING = True
    lab_lite.REGIME_MODEL_DIR = CONFIG['regime_models_dir']
    lab_lite.CLUSTER_MIN_TRADES = CONFIG['cluster_min_trades']
    lab_lite.CLUSTER_TOP_N_PER = CONFIG['cluster_top_n_per']
    lab_lite.CLUSTER_MIN_PF = CONFIG['cluster_min_pf']

    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # Warm up sklearn
    try:
        from sklearn.mixture import GaussianMixture
        import numpy as np
        GaussianMixture(n_components=2, n_init=1, max_iter=1, random_state=42).fit(
            np.random.randn(20, 3))
    except ImportError:
        pass

    skipped = 0
    generated = 0
    failed = []

    for i, symbol in enumerate(symbols, 1):
        # Verificar prerrequisitos
        ok, msg = check_prerequisites(symbol, 'lite')
        if not ok:
            print(f"   [{i}/{len(symbols)}] {symbol} — {msg}")
            failed.append(symbol)
            continue

        pp = presets_path(symbol)
        if not recycle and os.path.exists(pp):
            print(f"   [{i}/{len(symbols)}] {symbol} — presets existen, saltando")
            skipped += 1
            continue

        print(f"   [{i}/{len(symbols)}] {symbol} — generando presets...")
        try:
            result = lab_lite.process_symbol(symbol)
            if result:
                generated += 1
            else:
                print(f"      Sin resultados")
                failed.append(symbol)
        except Exception as e:
            print(f"      FAILED: {e}")
            import traceback; traceback.print_exc()
            failed.append(symbol)

    print(f"\n   Resultado: {generated} generados, {skipped} saltados, {len(failed)} fallidos")
    if failed:
        print(f"   Fallidos: {', '.join(failed)}")

    # Verificar
    missing = [s for s in symbols if not os.path.exists(presets_path(s))]
    if missing:
        print(f"\n   ⚠️  Símbolos sin presets: {', '.join(missing)}")

    return failed


# ============================================
# PASO 4: REGIME WALK-FORWARD
# ============================================

def step_regime_wf(symbols, recycle=False, cli_args=None):
    """Ejecuta regime walk-forward validation para cada símbolo."""
    print("\n" + "=" * 70)
    print("🔬 PASO 4: REGIME WALK-FORWARD")
    print("=" * 70)

    import regime_walk_forward as rwf
    import lab_historico_numba_v8_3 as lab

    # Sobreescribir globals de regime_walk_forward
    rwf.TRAIN_RATIO = CONFIG['train_ratio']
    rwf.MIN_EPISODE_BARS = CONFIG['min_episode_bars']
    rwf.MIN_TRADES_CLUSTER = CONFIG['min_trades_cluster']
    rwf.MIN_EPISODES_VALIDATE = CONFIG['min_episodes_validate']
    rwf.TOP_TRAIN = CONFIG['top_train']
    rwf.OUTPUT_DIR = CONFIG['regime_wf_dir']
    rwf.CACHE_DIR = CONFIG['data_cache_dir']
    rwf.MODEL_DIR = CONFIG['regime_models_dir']
    rwf.TIMEFRAME = CONFIG['timeframe']

    # Filtros internos
    rwf._TRAIN_MIN_TRADES = CONFIG['train_min_trades']
    rwf._TRAIN_MIN_PF = CONFIG['train_min_pf']
    rwf._FWD_MIN_TRADES = CONFIG['fwd_min_trades']
    rwf._FWD_MIN_PF = CONFIG['fwd_min_pf']
    rwf._TOP_KEEP_RAM = CONFIG['top_keep_ram']
    rwf._TOP_JSON = CONFIG['top_json']
    rwf._CROSS_CLUSTER_MIN_PF = CONFIG['cross_cluster_pf_min']
    rwf._CROSS_CLUSTER_MIN_TRANS_PROB = CONFIG['cross_cluster_p_min']

    # Sobreescribir globals de lab_historico (usado por regime_wf para simulación)
    lab.SL_PERCENT = CONFIG['sl_percent']
    lab.SL_EMERGENCY_PERCENT = CONFIG['sl_emergency_percent']
    lab.TS_PERCENT = CONFIG['ts_percent']
    lab.COOLDOWN_BARS = CONFIG['cooldown_bars']
    lab.COMMISSION_ROUND_TRIP = CONFIG['commission_round_trip']

    os.makedirs(CONFIG['regime_wf_dir'], exist_ok=True)

    # Warm up sklearn
    from sklearn.mixture import GaussianMixture
    import numpy as np
    GaussianMixture(n_components=2, n_init=1, max_iter=1, random_state=42).fit(
        np.random.randn(20, 3))

    # Construir args mock para process_symbol (simula argparse namespace)
    class Args:
        pass
    args = Args()
    args.train_ratio = CONFIG['train_ratio']
    args.min_episode_bars = CONFIG['min_episode_bars']
    args.output_dir = CONFIG['regime_wf_dir']
    args.presets_dir = CONFIG['output_dir']

    # Toxic tail
    if str(CONFIG['toxic_tail']).lower() == 'dynamic':
        args.toxic_tail_mode = 'dynamic'
        args.toxic_tail = 0
    else:
        args.toxic_tail_mode = 'fixed'
        args.toxic_tail = int(CONFIG['toxic_tail'])
    args.max_toxic_tail = CONFIG['max_toxic_tail']
    args.min_toxic_tail = CONFIG['min_toxic_tail']
    args.confirm_threshold = CONFIG['confirm_threshold']
    # R1 v18 — chunk_size pass-through (default 5M robust; cli_args overrides)
    args.chunk_size = getattr(cli_args, 'chunk_size', 5_000_000) if cli_args is not None else 5_000_000
    # M4 v18 — preset slice (default full range; worker mode injects via cli_args)
    args.preset_idx_start = getattr(cli_args, 'preset_idx_start', 0) if cli_args is not None else 0
    args.preset_idx_end = getattr(cli_args, 'preset_idx_end', None) if cli_args is not None else None

    import gc

    skipped = 0
    completed = 0
    failed = []
    results_summary = {}

    for i, symbol in enumerate(symbols, 1):
        # Verificar prerrequisitos
        ok, msg = check_prerequisites(symbol, 'regime-wf')
        if not ok:
            print(f"\n   [{i}/{len(symbols)}] {symbol} — {msg}")
            failed.append(symbol)
            results_summary[symbol] = {'status': 'prereq_fail', 'msg': msg}
            continue

        jp = specialist_json_path(symbol)
        if not recycle and os.path.exists(jp):
            # Leer resumen del JSON existente
            try:
                with open(jp, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                n_clusters = data.get('n_clusters', '?')
                clusters_info = data.get('clusters', {})
                specs_per = {k: len(v.get('specialists', [])) for k, v in clusters_info.items()}
                results_summary[symbol] = {'status': 'skipped', 'n_clusters': n_clusters, 'specs': specs_per}
            except (OSError, json.JSONDecodeError) as _e:
                # tanda tech-debt 2026-06-05: distinguir skip por error de lectura de
                # skip limpio — un JSON corrupto/ilegible NO debe pasar por 'skipped'
                # silencioso (riesgo de omitir un símbolo del output sin aviso).
                results_summary[symbol] = {'status': 'skip_read_error', 'error': f"{type(_e).__name__}: {_e}"}
                print(f"   ⚠️  {symbol}: JSON final existe pero ILEGIBLE ({type(_e).__name__}: {_e}) — marcado skip_read_error")
            print(f"\n   [{i}/{len(symbols)}] {symbol} — JSON final existe, saltando")
            skipped += 1
            continue

        print(f"\n{'=' * 70}")
        print(f"   [{i}/{len(symbols)}] {symbol}")
        print(f"{'=' * 70}")

        try:
            # Load data
            df = rwf.load_full_history(symbol)
            if df is None or len(df) < 5000:
                print(f"   ⚠️  Datos insuficientes")
                failed.append(symbol)
                results_summary[symbol] = {'status': 'fail', 'msg': 'datos insuficientes'}
                continue

            # Load model
            model_data = rwf.load_regime_model(symbol)
            if model_data is None:
                failed.append(symbol)
                results_summary[symbol] = {'status': 'fail', 'msg': 'no se pudo cargar modelo'}
                continue

            # Load presets
            presets = rwf.load_presets(symbol, args.presets_dir)
            if presets is None or len(presets) == 0:
                failed.append(symbol)
                results_summary[symbol] = {'status': 'fail', 'msg': 'no se pudieron cargar presets'}
                continue

            # Process
            result = rwf.process_symbol(symbol, presets, df, model_data, args)

            del df, model_data, presets
            gc.collect()

            if result is None:
                failed.append(symbol)
                results_summary[symbol] = {'status': 'fail', 'msg': 'process_symbol retornó None'}
                continue

            # Extract validated specialists
            rwf.extract_validated_specialists(result, args.output_dir)

            # Cleanup part files — Sub-Option A fix v18.2 (2026-05-11):
            # Skip in M4 worker mode (internal_subprocess_mode=True). Parent
            # _step_regime_wf_with_subprocess_respawn owns cleanup post-all-chunks
            # so last worker's extract_validated_specialists sees ALL parts (not
            # just its own slice). Without this skip, each worker deletes prior
            # workers' parts → final JSON reflects ONLY last chunk's data (6×
            # truncation empirically observed Run 1 BTC: 5,162 vs 29,534 validated
            # specialists + 5 vs 12 preset families + Brain≠Kernel fidelity edge).
            _skip_cleanup = (cli_args is not None and
                             getattr(cli_args, 'internal_subprocess_mode', False))
            if not _skip_cleanup:
                _parts_dir = result.get('parts_dir')
                if _parts_dir and os.path.isdir(_parts_dir):
                    import glob as _glob_cleanup
                    import shutil
                    _pfiles = _glob_cleanup.glob(os.path.join(_parts_dir, "*.parquet"))
                    _n_files = len(_pfiles)
                    _total_bytes = sum(os.path.getsize(f) for f in _pfiles)
                    shutil.rmtree(_parts_dir)
                    print(f"   🗑️ Limpieza: eliminados {_n_files} part files"
                          f" ({_total_bytes / 1e6:.1f} MB)")

            # Resumen
            n_clusters = result.get('n_clusters', '?')
            results_summary[symbol] = {'status': 'ok', 'n_clusters': n_clusters}
            completed += 1

        except Exception as e:
            print(f"   FAILED: {e}")
            import traceback; traceback.print_exc()
            failed.append(symbol)
            results_summary[symbol] = {'status': 'fail', 'msg': str(e)}

        gc.collect()
        # Mitigation v17 post-DIAG.11 cross-2-crashes TDR 0x116:
        # defensive flush deferred GPU dealloc queue inter-symbol.
        # NO cuda.close() — preservar singleton _cuda_sim válido en regime_walk_forward.
        try:
            from numba import cuda as _cuda_v17
            _cuda_v17.current_context().deallocations.clear()
        except (ImportError, AttributeError, RuntimeError) as _ce:
            # tanda tech-debt 2026-06-05: cleanup CUDA es best-effort, pero logear el
            # fallo (antes era silencioso — operador sin visibilidad si teardown falla).
            print(f"   ⚠️  CUDA dealloc flush inter-symbol falló (best-effort): {type(_ce).__name__}: {_ce}")

    print(f"\n   Resultado: {completed} completados, {skipped} saltados, {len(failed)} fallidos")
    if failed:
        print(f"   Fallidos: {', '.join(failed)}")

    return failed, results_summary


# ============================================
# RESUMEN FINAL
# ============================================

def print_summary(symbols, results_summary, total_time):
    """Imprime resumen final del pipeline."""
    print("\n" + "=" * 70)
    print("MASTER PIPELINE — Resumen")
    print("=" * 70)
    print(f"Símbolos procesados: {len(symbols)}")

    ok_count = 0
    warn_count = 0
    fail_count = 0
    skip_count = 0

    for symbol in symbols:
        info = results_summary.get(symbol, {})
        status = info.get('status', 'unknown')
        sc = sym_clean(symbol)

        if status == 'ok':
            n_c = info.get('n_clusters', '?')
            print(f"  ✅ {symbol}: {n_c} clusters")
            ok_count += 1
        elif status == 'skipped':
            n_c = info.get('n_clusters', '?')
            specs = info.get('specs', {})
            if specs:
                specs_str = ", ".join(f"{k}={v}" for k, v in specs.items())
                print(f"  ✅ {symbol}: {n_c} clusters, {specs_str} specialists (ya existía)")
            else:
                print(f"  ✅ {symbol}: ya existía")
            skip_count += 1
        elif status == 'prereq_fail':
            msg = info.get('msg', '')
            print(f"  ❌ {symbol}: {msg}")
            fail_count += 1
        elif status == 'fail':
            msg = info.get('msg', '')
            print(f"  ⚠️ {symbol}: {msg}")
            warn_count += 1
        else:
            print(f"  ❓ {symbol}: estado desconocido")

    print(f"\nOutput: {CONFIG['regime_wf_dir']}/*_specialist_configs.json")
    print(f"Tiempo total: {format_duration(total_time)}")

    # Contar JSONs finales
    json_count = sum(1 for s in symbols if os.path.exists(specialist_json_path(s)))
    print(f"JSONs finales: {json_count}/{len(symbols)}")
    print("=" * 70)


# ============================================
# CLI
# ============================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Master pipeline de producción: download → train → lite → regime-wf',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python master.py                                  # Todo, 45 símbolos
  python master.py --symbols BTC/USDT ETH/USDT      # Solo estos
  python master.py --recycle                         # Fuerza re-entrenar y re-generar
  python master.py --skip-download                   # Salta descarga
  python master.py --from-step regime-wf             # Desde regime walk-forward
  python master.py --from-step lite                  # Desde lab_lite
        """
    )
    parser.add_argument('--symbols', nargs='+', default=None,
                        help='Símbolos a procesar (default: todos los 45)')
    parser.add_argument('--recycle', action='store_true',
                        help='Fuerza re-entrenar modelos y re-generar todo')
    parser.add_argument('--skip-download', action='store_true',
                        help='Salta paso 1 (descarga de datos)')
    parser.add_argument('--from-step', type=str, default=None,
                        choices=STEP_NAMES,
                        help='Empieza desde este paso (download, train, lite, regime-wf)')
    # H_B isolation fix 2026-05-19 — Crash 12+13 cross-Sub-Sesiones precedent absoluto
    # cumulative refuted full-pipeline pattern. Grupo 1 ZERO crashes 42h con
    # `--from-step regime-wf` aislado vs Grupo 2 crashed full pipeline same process.
    # --to-step permite split del pipeline en subprocess separados: phase 1
    # (steps 1-3) + phase 2 (step 4 isolated GPU compute fresh process).
    parser.add_argument('--to-step', type=str, default=None,
                        choices=STEP_NAMES,
                        help='Termina en este paso inclusive (default regime-wf = last). '
                             'Permite split pipeline en subprocess separados para isolation '
                             'step 4 GPU (H_B fix Crash 12+13 cumulative).')
    # R1 v18 — config chunking (CHUNK_SIZE configurable runtime)
    parser.add_argument('--chunk-size', type=int, default=5_000_000,
                        help='R1 v18: max configs per kernel call clustered path '
                             '(default 5_000_000). Reduce peak VRAM allocation linearly. '
                             'Set 0 ó >= n_configs para single-call legacy behavior.')
    # M4 v18 — subprocess respawn (defensive Fenómeno 2 leak residual)
    parser.add_argument('--presets-per-subprocess', type=int, default=0,
                        help='M4 v18: spawn subprocess cada N presets per symbol '
                             '(default 0 = disabled, single process). Set 5 para '
                             'defensive context reset cumulative cross-presets.')
    parser.add_argument('--internal-subprocess-mode', action='store_true',
                        help='M4 v18 internal flag — worker mode invoked by orchestrator. '
                             'Do NOT pass directly.')
    parser.add_argument('--target-symbol', type=str, default=None,
                        help='M4 v18 internal — symbol for worker mode.')
    parser.add_argument('--preset-idx-start', type=int, default=0,
                        help='M4 v18 internal — preset slice start index.')
    parser.add_argument('--preset-idx-end', type=int, default=None,
                        help='M4 v18 internal — preset slice end index (None=len(presets)).')
    return parser.parse_args()


# ============================================
# MAIN
# ============================================

def _step_regime_wf_with_subprocess_respawn(symbols, cli_args):
    """M4 v18 — orchestrate per-symbol subprocess respawn cada N presets.

    Defensive Fenómeno 2 leak residual cumulative cross-presets: cada subprocess
    starts con CUDA context fresh, processes preset slice de N presets, exits
    cleanly → context destruido → próximo subprocess starts clean. Skip-if-exists
    en regime_walk_forward.py:546-553 handles resume natural cross-subprocess.
    """
    import subprocess
    import json
    import regime_walk_forward as rwf

    failed = []
    results_summary = {}
    presets_per_subprocess = max(1, int(cli_args.presets_per_subprocess))

    # B.1 fix v18.1 — defensive cleanup parent CUDA context inicial pre-loop.
    # rwf import (línea 723) crea CUDA context en parent process. Flush deferred
    # dealloc queue antes de spawnar primer child subprocess preserva consistency
    # con pattern legacy step_regime_wf:589-597 + Caveat #17 mitigation parent-side.
    try:
        import gc as _gc_b1
        _gc_b1.collect()
        from numba import cuda as _cuda_b1
        _cuda_b1.current_context().deallocations.clear()
    except Exception:
        pass

    for i, symbol in enumerate(symbols, 1):
        print(f"\n{'=' * 70}")
        print(f"   [{i}/{len(symbols)}] {symbol} — M4 subprocess respawn mode "
              f"({presets_per_subprocess} presets/subprocess)")
        print(f"{'=' * 70}")

        ok, msg = check_prerequisites(symbol, 'regime-wf')
        if not ok:
            print(f"   {msg}")
            failed.append(symbol)
            results_summary[symbol] = {'status': 'prereq_fail', 'msg': msg}
            continue

        jp = specialist_json_path(symbol)
        if not cli_args.recycle and os.path.exists(jp):
            print(f"   JSON final existe, saltando")
            results_summary[symbol] = {'status': 'skipped'}
            continue

        # Load presets count para chunking (subprocess re-loads presets internally)
        presets = rwf.load_presets(symbol, CONFIG['output_dir'])
        if presets is None or len(presets) == 0:
            failed.append(symbol)
            results_summary[symbol] = {'status': 'fail', 'msg': 'no presets'}
            continue
        n_presets = len(presets)
        del presets

        n_chunks = (n_presets + presets_per_subprocess - 1) // presets_per_subprocess
        master_path = os.path.abspath(__file__)
        any_chunk_failed = False

        for ci in range(n_chunks):
            ps = ci * presets_per_subprocess
            pe = min(ps + presets_per_subprocess, n_presets)
            print(f"\n   ─ Chunk {ci+1}/{n_chunks}: presets[{ps}:{pe}] ({pe-ps} presets)")
            cmd = [
                sys.executable, master_path,
                '--internal-subprocess-mode',
                '--target-symbol', symbol,
                '--preset-idx-start', str(ps),
                '--preset-idx-end', str(pe),
                '--chunk-size', str(cli_args.chunk_size),
                '--from-step', 'regime-wf',
            ]
            try:
                cp = subprocess.run(cmd, check=False)
                if cp.returncode != 0:
                    print(f"   ⚠️  Chunk {ci+1} exit code {cp.returncode} "
                          f"(skip-if-exists may resume next attempt)")
                    any_chunk_failed = True
            except Exception as e:
                print(f"   ⚠️  Chunk {ci+1} subprocess exception: {e}")
                any_chunk_failed = True

            # B.1 fix v18.1 — defensive cleanup parent context post-subprocess.run.
            # Pattern equivalente legacy step_regime_wf:589-597 aplicado inter-chunk.
            try:
                import gc as _gc_b1
                _gc_b1.collect()
                from numba import cuda as _cuda_b1
                _cuda_b1.current_context().deallocations.clear()
            except Exception:
                pass

        # Post-chunks: parent cleanup parts_dir — Sub-Option A fix v18.2 (2026-05-11).
        # Workers skip cleanup (internal_subprocess_mode=True) → parts persist
        # progressively across chunks. Last worker's extract_validated_specialists
        # sees ALL parts and produces correct final JSON. Parent now owns final
        # cleanup post-all-chunks completion.
        try:
            sc = sym_clean(symbol)
            parts_dir = os.path.join(CONFIG['regime_wf_dir'], f"_parts_{sc}")
            if os.path.isdir(parts_dir):
                import glob as _glob_cleanup
                import shutil
                _pfiles = _glob_cleanup.glob(os.path.join(parts_dir, "*.parquet"))
                _n_files = len(_pfiles)
                _total_bytes = sum(os.path.getsize(f) for f in _pfiles)
                shutil.rmtree(parts_dir)
                print(f"   🗑️ Parent post-all-chunks cleanup: {_n_files} part files"
                      f" ({_total_bytes / 1e6:.1f} MB)")
        except Exception as e:
            print(f"   ⚠️  Post-chunks parent cleanup exception: {e}")

        if any_chunk_failed:
            results_summary[symbol] = {'status': 'partial'}
        else:
            results_summary[symbol] = {'status': 'ok'}

        # B.1 fix v18.1 — defensive cleanup parent context post-symbol completion.
        # Pattern equivalente legacy step_regime_wf:589-597 aplicado inter-symbol.
        try:
            import gc as _gc_b1
            _gc_b1.collect()
            from numba import cuda as _cuda_b1
            _cuda_b1.current_context().deallocations.clear()
        except Exception:
            pass

    return failed, results_summary


def main():
    args = parse_args()
    t0 = time.time()

    # M4 v18 — worker mode: process single symbol + preset slice, exit
    if args.internal_subprocess_mode:
        if not args.target_symbol:
            print("ERROR: --internal-subprocess-mode requires --target-symbol")
            sys.exit(1)
        # Worker delegates to step_regime_wf con scope reducido (single symbol)
        # process_symbol slicing presets[preset_idx_start:preset_idx_end] vía args.
        failed, _ = step_regime_wf([args.target_symbol], recycle=False, cli_args=args)
        sys.exit(0 if not failed else 1)

    # Resolver símbolos
    symbols = args.symbols if args.symbols else CONFIG['symbols']
    # Validar símbolos
    valid_symbols = CONFIG['symbols']
    invalid = [s for s in symbols if s not in valid_symbols]
    if invalid:
        print(f"⚠️  Símbolos no reconocidos (ignorados): {', '.join(invalid)}")
        symbols = [s for s in symbols if s in valid_symbols]
    if not symbols:
        print("❌ No hay símbolos válidos para procesar")
        sys.exit(1)

    # Resolver paso inicial
    from_step = 0
    if args.from_step:
        from_step = STEP_ORDER[args.from_step]
    if args.skip_download and from_step == 0:
        from_step = 1  # saltar download

    # H_B isolation fix 2026-05-19 — Resolver paso final inclusive
    to_step = 3  # default = run through last step (regime-wf)
    if args.to_step:
        to_step = STEP_ORDER[args.to_step]
    if to_step < from_step:
        print(f"❌ ERROR: --to-step ({args.to_step}) < --from-step ({args.from_step})",
              file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("🚀 MASTER PIPELINE — Producción")
    print(f"   Símbolos: {len(symbols)}")
    print(f"   Desde paso: {STEP_NAMES[from_step]} ({from_step + 1}/4)")
    print(f"   Hasta paso: {STEP_NAMES[to_step]} ({to_step + 1}/4)")
    print(f"   Recycle: {'SI' if args.recycle else 'NO'}")
    print(f"   SL: {CONFIG['sl_percent']}% | TS: {CONFIG['ts_percent']}% | "
          f"Commission: {CONFIG['commission_round_trip']}%")
    print(f"   Toxic tail: {CONFIG['toxic_tail']} | "
          f"Train ratio: {CONFIG['train_ratio']}")
    print(f"   Output: {CONFIG['output_dir']} | WF: {CONFIG['regime_wf_dir']}")
    print("=" * 70)

    all_failed = []
    results_summary = {}

    # Paso 1: Download
    if from_step <= 0 and to_step >= 0:
        failed = step_download(symbols, recycle=args.recycle)
        all_failed.extend(failed)

    # Paso 2: Train
    if from_step <= 1 and to_step >= 1:
        failed = step_train(symbols, recycle=args.recycle)
        all_failed.extend(failed)

    # Paso 3: Lab Lite
    if from_step <= 2 and to_step >= 2:
        failed = step_lite(symbols, recycle=args.recycle)
        all_failed.extend(failed)

    # Paso 4: Regime Walk-Forward
    if from_step <= 3 and to_step >= 3:
        # M4 v18 — orchestrate subprocess respawn cada N presets si flag > 0
        if args.presets_per_subprocess and args.presets_per_subprocess > 0:
            failed, results_summary = _step_regime_wf_with_subprocess_respawn(
                symbols, args)
        else:
            failed, results_summary = step_regime_wf(symbols, recycle=args.recycle,
                                                    cli_args=args)
        all_failed.extend(failed)

    total_time = time.time() - t0

    # Resumen final
    print_summary(symbols, results_summary, total_time)

    if all_failed:
        # Deduplicate
        unique_failed = list(dict.fromkeys(all_failed))
        print(f"\n⚠️  Símbolos con al menos un fallo: {', '.join(unique_failed)}")


if __name__ == "__main__":
    main()
