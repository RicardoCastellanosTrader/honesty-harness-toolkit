# ============================================
# 🧪 LABORATORIO LITE v5e — CLUSTERING DE REGÍMENES
# ============================================
# Base: v5d (PnL relativo a EMAs, todo el período)
#
# CAMBIOS v5e vs v5d:
#   - Etiquetado de velas por régimen de mercado (GMM sobre Hurst, NATR, VolImbalance)
#   - Kernel Numba acumula PnL por cluster ADEMÁS del global (fidelidad intacta)
#   - Selección dual: global (backward compatible) + per-cluster (especialistas)
#   - Output: presets con cluster_id para pipeline downstream
#
# FIDELIDAD: El motor de simulación (zonas, entries, exits, equity) es IDÉNTICO a v5d.
#   Solo se añade contabilidad por cluster DESPUÉS de cada cierre de trade.
#   Los arrays globales (pnl_arr, trades_arr, raw_crosses_arr) son bit-identical a v5d.
# ============================================

import time
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import os
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')

# sklearn para GMM — se importa con fallback
try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  sklearn no disponible — clustering desactivado, modo v5d puro")

# Módulo compartido de features de régimen
from regime_features import compute_regime_features, FEATURE_NAMES

# joblib para cargar modelos pre-entrenados
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

# ============================================
# CONFIGURACIÓN
# ============================================

SYMBOLS = [
    "BNB/USDT", "BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT",
    "TRX/USDT", "DOGE/USDT", "ADA/USDT", "BCH/USDT", "LINK/USDT",
    "XLM/USDT", "SUI/USDT", "LTC/USDT", "AVAX/USDT", "HBAR/USDT",
    "SHIB/USDT", "DOT/USDT", "UNI/USDT", "TAO/USDT",
    "AAVE/USDT", "NEAR/USDT", "ICP/USDT", "ETC/USDT",
    "ONDO/USDT", "ENA/USDT", "POL/USDT", "WLD/USDT", "APT/USDT",
    "ATOM/USDT", "ALGO/USDT", "RENDER/USDT", "ARB/USDT", "FIL/USDT",
    "QNT/USDT", "VET/USDT", "SEI/USDT", "OP/USDT",
    "IMX/USDT", "INJ/USDT", "FET/USDT", "STX/USDT", "SAND/USDT",
    "MANA/USDT", "GRT/USDT", "THETA/USDT",
]


TOTAL_CANDLES = 5000
TIMEFRAME = "1h"
TRAIN_RATIO = 0.0  # v5d: medir PnL sobre TODO el período (no solo test)
TOP_SAVE = 5000

# EMA reference periods
EMA_FAST = 10
EMA_SLOW = 55
EMA_TREND = 200
COMMISSION_ROUND_TRIP = 0.10  # 0.1% round-trip

TOP_N_PRESETS = 7        # Presets a seleccionar por simbolo (global)
DIVERSITY_MODE = True    # Diversidad de familias en seleccion
OUTPUT_DIR = "."         # Directorio de salida (pipeline lo sobreescribe)

# --- Clustering config (v5e) ---
USE_CLUSTERING = True           # Activar clustering de regímenes (carga modelos pre-entrenados)
REGIME_MODEL_DIR = "regime_models"  # Directorio con modelos pre-entrenados
CLUSTER_MIN_TRADES = 30         # Mínimo trades por cluster para significancia
CLUSTER_TOP_N_PER = 3           # Top N presets por cluster
CLUSTER_MIN_PF = 1.2            # PF mínimo para ser especialista de un cluster

# ============================================
# EXCHANGE
# ============================================
exchange = ccxt.binance({'enableRateLimit': True})

def fetch_all_candles(symbol, timeframe, total_candles, max_retries=3):
    print(f"   📥 Descargando {total_candles} velas de {symbol}...")
    all_candles = []
    interval_ms = 3600000
    current_since = int(time.time() * 1000) - (total_candles + 200) * interval_ms
    while len(all_candles) < total_candles + 100:
        for attempt in range(max_retries):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
                if not ohlcv:
                    return all_candles[-total_candles:] if len(all_candles) >= total_candles else all_candles
                all_candles.extend(ohlcv)
                current_since = ohlcv[-1][0] + interval_ms
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"   ❌ Error: {e}")
                    return all_candles[-total_candles:] if len(all_candles) >= total_candles else all_candles
                time.sleep(2)
    return all_candles[-total_candles:]

# ============================================
# MOVING AVERAGE CALCULATIONS (16 familias)
# ============================================
# A12 dedup 2026-04-23: las 16 funciones MA ahora se importan desde
# lab_historico_numba_v8_3 (source-of-truth canónica, el mismo módulo
# desde el que live/brain_engine, mean_reversion_features y
# regime_walk_forward también importan).
#
# Pre-refactor verificación bit-a-bit (14 MAs × 3-4 periods = 52 tests,
# 52/52 PASS rtol=1e-10, max drift 3.3e-11 en calc_vwma por cumsum
# vectorizado). calc_wma era única local sin contraparte pública en
# lab_historico (solo usada internamente por calc_hma lite). Tras
# importar calc_hma canónico, calc_wma deja de tener usuario y se
# elimina. Ver §13.4 entrada A12 IMPLEMENTADO 2026-04-23.

from lab_historico_numba_v8_3 import (
    calc_ema, calc_sma, calc_hma, calc_alma, calc_zlema, calc_kama,
    calc_dema, calc_tema, calc_mcginley, calc_vidya, calc_frama,
    calc_t3, calc_ssmoother, calc_vwma, calc_tenkan, calc_jma,
)

# ============================================
# MA TYPE INDICES & CATALOG
# ============================================
MA_NAMES = {
    0: 'EMA', 1: 'SMA', 2: 'HMA', 3: 'ALMA', 4: 'ZLEMA', 5: 'KAMA',
    6: 'DEMA', 7: 'TEMA', 8: 'McGinley', 9: 'VIDYA', 10: 'FRAMA',
    11: 'T3', 12: 'SSmoother', 13: 'VWMA', 14: 'Tenkan', 15: 'JMA',
}
NONE_TYPE = 99
SIMPLE_TYPES = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14]

def build_catalog():
    fast_periods = list(range(8, 25, 2))
    fast = []
    for p in fast_periods:
        for t in SIMPLE_TYPES: fast.append((t, p, 0, 0))
        for off in [0.5, 0.85]:
            for sig in [4, 6]: fast.append((3, p, off, sig))
        for vf in [0.7, 0.8]: fast.append((11, p, vf, 0))
        fast.append((15, p, 0, 2))
    slow_periods = sorted(set(list(range(30, 76, 3)) + [49]))
    slow = []
    for p in slow_periods:
        for t in SIMPLE_TYPES: slow.append((t, p, 0, 0))
        for off in [0.5, 0.85]:
            for sig in [4, 6]: slow.append((3, p, off, sig))
        for vf in [0.7, 0.8]: slow.append((11, p, vf, 0))
        slow.append((15, p, 0, 2))
    # Trend NO es variable libre en el lite.
    # Se deriva como slow×4 al exportar presets (ver result_to_preset_tuple).
    # Internamente el lite simula sin filtro de tendencia (NONE).
    trend = [(NONE_TYPE, 0, 0, 0)]
    return fast, slow, trend

def calc_ma(close, high, low, volume, ma_type, period, p1=0, p2=0):
    if ma_type == 0:  return calc_ema(close, period)
    elif ma_type == 1:  return calc_sma(close, period)
    elif ma_type == 2:  return calc_hma(close, period)
    elif ma_type == 3:  return calc_alma(close, period, p1, p2)
    elif ma_type == 4:  return calc_zlema(close, period)
    elif ma_type == 5:  return calc_kama(close, period)
    elif ma_type == 6:  return calc_dema(close, period)
    elif ma_type == 7:  return calc_tema(close, period)
    elif ma_type == 8:  return calc_mcginley(close, period)
    elif ma_type == 9:  return calc_vidya(close, period)
    elif ma_type == 10: return calc_frama(close, high, low, period)
    elif ma_type == 11: return calc_t3(close, period, p1)
    elif ma_type == 12: return calc_ssmoother(close, period)
    elif ma_type == 13: return calc_vwma(close, volume, period)
    elif ma_type == 14: return calc_tenkan(high, low, period)
    elif ma_type == 15: return calc_jma(close, period, int(p1), p2)
    else:               return np.full(len(close), np.nan)

def precalculate_all_mas(df):
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)
    n = len(close)
    fast_cat, slow_cat, trend_cat = build_catalog()
    print(f"   🔧 Precalculando MAs: {len(fast_cat)}F + {len(slow_cat)}S + {len(trend_cat)}T...")
    fast_arr = np.full((len(fast_cat), n), np.nan, dtype=np.float64)
    for i, (t, p, p1, p2) in enumerate(fast_cat):
        fast_arr[i] = calc_ma(close, high, low, volume, t, p, p1, p2)
    slow_arr = np.full((len(slow_cat), n), np.nan, dtype=np.float64)
    for i, (t, p, p1, p2) in enumerate(slow_cat):
        slow_arr[i] = calc_ma(close, high, low, volume, t, p, p1, p2)
    trend_arr = np.full((len(trend_cat), n), np.nan, dtype=np.float64)
    for i, (t, p, p1, p2) in enumerate(trend_cat):
        if t == NONE_TYPE: trend_arr[i] = np.zeros(n)
        else: trend_arr[i] = calc_ma(close, high, low, volume, t, p, p1, p2)
    return close, fast_arr, slow_arr, trend_arr, fast_cat, slow_cat, trend_cat

def ma_name(cat_entry):
    t, p, p1, p2 = cat_entry
    if t == NONE_TYPE: return "NONE"
    name = MA_NAMES.get(t, f"T{t}")
    if t == 3: return f"{name}({p},{p1},{p2})"
    if t == 11: return f"{name}({p},v{p1})"
    return f"{name}({p})"

def ma_family(cat_entry):
    t = cat_entry[0]
    if t == NONE_TYPE: return "NONE"
    return MA_NAMES.get(t, f"T{t}")


# ============================================
# SELECTION FUNCTIONS
# ============================================

def select_top_diverse(top_results, top_n=7, period_diff_min=3):
    """Selecciona top N presets con diversidad de familias + 1 variante por familia.

    Estrategia:
    1. Mejor de cada familia (fast_type/slow_type)
    2. Si hay mas familias que slots: top N familias por score
    3. Si hay menos: rellenar con siguientes mejores por score
    4. Post-seleccion: para cada familia seleccionada, buscar la siguiente
       mejor combo de esa misma familia con slow_period diff>=period_diff_min.
       Resultado: hasta top_n*2 presets (en practica 9-12).
    """
    if not top_results or top_n <= 0:
        return top_results[:top_n]

    # Agrupar por familia (primera aparicion = mejor por score)
    seen_families = {}
    for i, res in enumerate(top_results):
        fam = (ma_family(res['fast']), ma_family(res['slow']))
        if fam not in seen_families:
            seen_families[fam] = i

    # Mejores por familia, en orden de score
    family_best_indices = sorted(seen_families.values())

    if len(family_best_indices) >= top_n:
        base_indices = family_best_indices[:top_n]
    else:
        # Menos familias que slots: tomar todos los family bests + rellenar
        selected = set(family_best_indices)
        base_indices = list(family_best_indices)

        for i in range(len(top_results)):
            if len(base_indices) >= top_n:
                break
            if i not in selected:
                base_indices.append(i)
                selected.add(i)

        base_indices = sorted(base_indices[:top_n])

    # --- Variantes de periodo ---
    # Para cada familia seleccionada, registrar el slow_period usado
    selected_set = set(base_indices)
    family_periods = {}  # fam -> slow_period del seleccionado
    for i in base_indices:
        res = top_results[i]
        fam = (ma_family(res['fast']), ma_family(res['slow']))
        family_periods[fam] = res['slow'][1]

    variant_indices = []
    # Para cada familia seleccionada, buscar 1 variante con slow_period diferente
    families_with_variant = set()

    for i, res in enumerate(top_results):
        if i in selected_set:
            continue
        fam = (ma_family(res['fast']), ma_family(res['slow']))
        if fam not in family_periods or fam in families_with_variant:
            continue
        slow_p = res['slow'][1]
        if abs(slow_p - family_periods[fam]) >= period_diff_min:
            variant_indices.append(i)
            selected_set.add(i)
            families_with_variant.add(fam)

    result_indices = sorted(base_indices + variant_indices)
    return [top_results[i] for i in result_indices]


def select_top_per_cluster(all_configs, cl_pnl, cl_trades, cl_gp, cl_gl,
                            n_clusters, fast_cat, slow_cat, trend_cat,
                            n_slow, n_trend,
                            min_trades=30, min_pf=1.2, top_n=3,
                            period_diff_min=3):
    """Selecciona top N presets ESPECIALISTAS por cluster.

    Para cada cluster:
    1. Filtrar configs con >= min_trades trades en ese cluster
    2. Calcular PF = gross_profit / gross_loss dentro del cluster
    3. Filtrar PF >= min_pf
    4. Rankear por PF (no por PnL: PF mide calidad estructural)
    5. Aplicar diversidad de familias
    6. Seleccionar top N

    Returns: dict {cluster_id: [list of result dicts with cluster_pf, cluster_pnl, etc.]}
    """
    n_configs = len(all_configs)
    cluster_presets = {}

    for k in range(n_clusters):
        # Filtrar por trades mínimos en este cluster
        valid_mask = cl_trades[:, k] >= min_trades
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            cluster_presets[k] = []
            continue

        # Calcular PF por cluster
        gp = cl_gp[valid_indices, k]
        gl = cl_gl[valid_indices, k]
        pf = np.where(gl > 0, gp / gl, np.where(gp > 0, 99.0, 0.0))

        # Filtrar por PF mínimo
        pf_mask = pf >= min_pf
        pf_valid = valid_indices[pf_mask]
        pf_values = pf[pf_mask]

        if len(pf_valid) == 0:
            cluster_presets[k] = []
            continue

        # Rankear por PF descendente
        sort_idx = np.argsort(-pf_values)
        ranked_indices = pf_valid[sort_idx]
        ranked_pf = pf_values[sort_idx]

        # Construir resultados para selección diversa
        results_for_cluster = []
        for rank_pos in range(min(len(ranked_indices), top_n * 20)):
            cfg_idx = ranked_indices[rank_pos]
            fi = cfg_idx // (n_slow * n_trend)
            remainder = cfg_idx % (n_slow * n_trend)
            si = remainder // n_trend
            ti = remainder % n_trend

            results_for_cluster.append({
                'fast': fast_cat[fi], 'slow': slow_cat[si], 'trend': trend_cat[ti],
                'cluster_pf': float(ranked_pf[rank_pos]),
                'cluster_pnl': float(cl_pnl[cfg_idx, k]),
                'cluster_trades': int(cl_trades[cfg_idx, k]),
                'cluster_gp': float(cl_gp[cfg_idx, k]),
                'cluster_gl': float(cl_gl[cfg_idx, k]),
                'global_pnl': float(all_configs[cfg_idx]) if all_configs is not None else 0.0,
                'score': float(ranked_pf[rank_pos]),  # use PF as score for diversity selector
            })

        # Aplicar diversidad de familias
        selected = select_top_diverse(results_for_cluster, top_n, period_diff_min)
        cluster_presets[k] = selected

    return cluster_presets


def result_to_preset_tuple(res):
    """Convierte resultado del lab lite a tupla de preset para lab historico.

    Trend se deriva siempre como slow×4: mismo tipo/params, periodo×4.
    Esto garantiza compatibilidad con el lab completo (TF5 necesita trend).

    Returns: (fast_type_str, fast_period, fast_p1, fast_p2,
              slow_type_str, slow_period, slow_p1, slow_p2,
              trend_type_str, trend_period, trend_p1, trend_p2)
    """
    def convert(entry):
        type_idx, period, p1, p2 = entry
        if type_idx == NONE_TYPE:
            return ("NONE", 0, 0.0, 0.0)
        return (MA_NAMES.get(type_idx, f"T{type_idx}"), int(period), float(p1), float(p2))

    f = convert(res['fast'])
    s = convert(res['slow'])
    # Trend = slow×4 (mismo tipo y params, periodo multiplicado)
    slow_type_idx, slow_period, slow_p1, slow_p2 = res['slow']
    t = convert((slow_type_idx, slow_period * 4, slow_p1, slow_p2))
    return (*f, *s, *t)


# ============================================
# REGIME MODEL LOADING (v5e — pre-trained models)
# ============================================

def load_and_apply_regime_model(symbol, df, model_dir="regime_models"):
    """Carga modelo pre-entrenado y etiqueta las velas del df.

    Returns:
        labels: array int64 (n_bars,), -1 para barras sin features válidas
        cluster_info: dict con metadata del modelo

    Si el modelo no existe, fallback a cluster_labels = 0 para todas las barras
    (comportamiento v5d, sin clustering).
    """
    n_bars = len(df)

    # Determinar nombre del archivo: BTC/USDT → BTC
    sym_key = symbol.replace("/USDT", "").replace("/", "")
    model_path = os.path.join(model_dir, f"{sym_key}_regime.joblib")

    if not os.path.exists(model_path) or not HAS_JOBLIB:
        if not HAS_JOBLIB:
            print(f"   ⚠️  joblib no disponible — fallback sin clustering")
        else:
            print(f"   ⚠️  Modelo no encontrado: {model_path} — fallback sin clustering")
        labels = np.zeros(n_bars, dtype=np.int32)
        return labels, {'n_clusters': 1, 'method': 'fallback_no_model',
                        'centroids': None, 'cluster_sizes': [n_bars],
                        'cluster_names': ['all'], 'feature_names': list(FEATURE_NAMES)}

    # Cargar modelo
    model_data = joblib.load(model_path)
    gmm = model_data['gmm']
    scaler = model_data['scaler']
    lookback = model_data['lookback']
    n_clusters = model_data['n_clusters']

    print(f"   🔬 Modelo cargado: {model_path} (k={n_clusters}, "
          f"trained on {model_data['training_bars']} bars, "
          f"{model_data['training_date_range'][0][:10]} → {model_data['training_date_range'][1][:10]})")

    # Calcular features con la MISMA función que usó el training
    features, valid_mask = compute_regime_features(df, lookback=lookback)

    # Aplicar scaler + predict del modelo pre-entrenado
    labels = np.full(n_bars, -1, dtype=np.int32)
    valid_features = features[valid_mask]
    n_valid = len(valid_features)

    if n_valid == 0:
        print(f"   ⚠️  Sin barras válidas para clasificar — fallback")
        labels[:] = 0
        return labels, {'n_clusters': 1, 'method': 'fallback_no_valid_bars',
                        'centroids': None, 'cluster_sizes': [n_bars],
                        'cluster_names': ['all'], 'feature_names': list(FEATURE_NAMES)}

    X = scaler.transform(valid_features)
    pred = gmm.predict(X)

    valid_indices = np.where(valid_mask)[0]
    for i, idx in enumerate(valid_indices):
        labels[idx] = pred[i]

    # Cluster sizes en esta ventana
    cluster_sizes = [int(np.sum(pred == k)) for k in range(n_clusters)]

    cluster_info = {
        'n_clusters': n_clusters,
        'method': f'pretrained_GMM(k={n_clusters})',
        'bic_values': model_data.get('bic_values', {}),
        'centroids': model_data['centroids'],
        'cluster_sizes': cluster_sizes,
        'cluster_names': model_data['cluster_names'],
        'feature_names': model_data['feature_names'],
        'training_bars': model_data['training_bars'],
        'training_date_range': model_data['training_date_range'],
    }

    return labels, cluster_info


# ============================================
# NUMBA ENGINE v5e — PnL COMPARISON + CLUSTER TRACKING
# ============================================

@jit(nopython=True)
def simulate_ema_pnl(close, ema_fast, ema_slow, ema_trend, commission_pct,
                     test_start, warmup, cluster_labels, n_clusters):
    """Simulate trades using EMA zones. Returns global + per-cluster stats."""
    n_bars = len(close)
    position = 0
    entry_price = 0.0
    entry_bar = 0
    equity = 100.0
    n_trades = 0
    n_raw_crosses = 0
    prev_side = 0

    # Per-cluster accumulators
    cl_pnl = np.zeros(n_clusters, dtype=np.float64)
    cl_trades = np.zeros(n_clusters, dtype=np.int32)

    for t in range(warmup, n_bars):
        ef = ema_fast[t]
        es = ema_slow[t]
        et = ema_trend[t]
        if ef != ef or es != es or et != et:
            continue

        cross_bull = ef > es
        cross_bear = ef < es

        # Raw crosses
        curr_side = 1 if cross_bull else -1
        if prev_side != 0 and curr_side != prev_side:
            if t >= test_start:
                n_raw_crosses += 1
        prev_side = curr_side

        zone_bull = cross_bull and close[t] > et
        zone_bear = cross_bear and close[t] < et

        if t < test_start:
            if position == 0:
                if zone_bull: position = 1; entry_price = close[t]; entry_bar = t
                elif zone_bear: position = -1; entry_price = close[t]; entry_bar = t
            elif position == 1 and (zone_bear or (not zone_bull and not zone_bear)):
                if zone_bear: position = -1; entry_price = close[t]; entry_bar = t
                else: position = 0
            elif position == -1 and (zone_bull or (not zone_bull and not zone_bear)):
                if zone_bull: position = 1; entry_price = close[t]; entry_bar = t
                else: position = 0
            continue

        # Test period
        if position == 0:
            if zone_bull: position = 1; entry_price = close[t]; entry_bar = t
            elif zone_bear: position = -1; entry_price = close[t]; entry_bar = t
        elif position == 1:
            if zone_bear or (not zone_bull and not zone_bear):
                pnl = (close[t] - entry_price) / entry_price * 100.0 - commission_pct
                equity *= (1 + pnl / 100.0)
                n_trades += 1
                # Cluster accounting
                cl_idx = cluster_labels[entry_bar]
                if 0 <= cl_idx < n_clusters:
                    cl_pnl[cl_idx] += pnl
                    cl_trades[cl_idx] += 1
                if zone_bear: position = -1; entry_price = close[t]; entry_bar = t
                else: position = 0
        elif position == -1:
            if zone_bull or (not zone_bull and not zone_bear):
                pnl = (entry_price - close[t]) / entry_price * 100.0 - commission_pct
                equity *= (1 + pnl / 100.0)
                n_trades += 1
                cl_idx = cluster_labels[entry_bar]
                if 0 <= cl_idx < n_clusters:
                    cl_pnl[cl_idx] += pnl
                    cl_trades[cl_idx] += 1
                if zone_bull: position = 1; entry_price = close[t]; entry_bar = t
                else: position = 0

    total_pnl = equity - 100.0
    return total_pnl, n_trades, n_raw_crosses, cl_pnl, cl_trades


@jit(nopython=True, parallel=True)
def run_simulation_v5e(close, fast_arr, slow_arr, trend_arr,
                       n_fast, n_slow, n_trend,
                       commission_pct, test_start, warmup,
                       cluster_labels, n_clusters):
    """Kernel v5e: IDENTICAL simulation logic to v5d + per-cluster accounting.

    The global arrays (pnl_arr, trades_arr, raw_crosses_arr) produce bit-identical
    results to run_simulation_v5d. The cluster arrays are purely additive tracking.
    """
    n_bars = len(close)
    n_configs = n_fast * n_slow * n_trend

    # Global arrays (identical to v5d)
    pnl_arr = np.zeros(n_configs, dtype=np.float64)
    trades_arr = np.zeros(n_configs, dtype=np.int32)
    raw_crosses_arr = np.zeros(n_configs, dtype=np.int32)

    # Per-cluster arrays (NEW — additive PnL decomposition)
    cl_pnl = np.zeros((n_configs, n_clusters), dtype=np.float64)
    cl_trades = np.zeros((n_configs, n_clusters), dtype=np.int32)
    cl_gp = np.zeros((n_configs, n_clusters), dtype=np.float64)  # gross profit
    cl_gl = np.zeros((n_configs, n_clusters), dtype=np.float64)  # gross loss (positive)

    for cfg_idx in prange(n_configs):
        fi = cfg_idx // (n_slow * n_trend)
        remainder = cfg_idx % (n_slow * n_trend)
        si = remainder // n_trend
        ti = remainder % n_trend

        position = 0
        entry_price = 0.0
        entry_bar = 0          # v5e: track entry bar for cluster assignment
        equity = 100.0
        n_trades = 0
        n_raw_crosses = 0
        prev_side = 0

        for t in range(warmup, n_bars):
            fv = fast_arr[fi, t]
            sv = slow_arr[si, t]
            if fv != fv or sv != sv:
                continue

            cross_bull = fv > sv
            cross_bear = fv < sv

            # Raw crosses (fast/slow without trend) — IDENTICAL to v5d
            curr_side = 0
            if cross_bull: curr_side = 1
            elif cross_bear: curr_side = -1
            if curr_side != 0 and prev_side != 0 and curr_side != prev_side:
                if t >= test_start:
                    n_raw_crosses += 1
            if curr_side != 0:
                prev_side = curr_side

            # Zone with trend filter — IDENTICAL to v5d
            if ti > 0:
                tv = trend_arr[ti, t]
                if tv != tv:
                    continue
                zone_bull = cross_bull and close[t] > tv
                zone_bear = cross_bear and close[t] < tv
            else:
                zone_bull = cross_bull
                zone_bear = cross_bear

            # Warmup / pre-test tracking — IDENTICAL to v5d + entry_bar
            if t < test_start:
                if position == 0:
                    if zone_bull: position = 1; entry_price = close[t]; entry_bar = t
                    elif zone_bear: position = -1; entry_price = close[t]; entry_bar = t
                elif position == 1:
                    if zone_bear or (ti > 0 and not zone_bull and not zone_bear):
                        if zone_bear: position = -1; entry_price = close[t]; entry_bar = t
                        else: position = 0
                elif position == -1:
                    if zone_bull or (ti > 0 and not zone_bull and not zone_bear):
                        if zone_bull: position = 1; entry_price = close[t]; entry_bar = t
                        else: position = 0
                continue

            # === TEST PERIOD — simulation logic IDENTICAL to v5d ===
            if position == 0:
                if zone_bull: position = 1; entry_price = close[t]; entry_bar = t
                elif zone_bear: position = -1; entry_price = close[t]; entry_bar = t
            elif position == 1:
                if zone_bear or (ti > 0 and not zone_bull and not zone_bear):
                    pnl = (close[t] - entry_price) / entry_price * 100.0 - commission_pct
                    equity *= (1 + pnl / 100.0)
                    n_trades += 1
                    # --- v5e cluster accounting (additive, no afecta equity) ---
                    cl_idx = cluster_labels[entry_bar]
                    if 0 <= cl_idx < n_clusters:
                        cl_pnl[cfg_idx, cl_idx] += pnl
                        cl_trades[cfg_idx, cl_idx] += 1
                        if pnl > 0:
                            cl_gp[cfg_idx, cl_idx] += pnl
                        else:
                            cl_gl[cfg_idx, cl_idx] += (-pnl)
                    # --- fin cluster accounting ---
                    if zone_bear: position = -1; entry_price = close[t]; entry_bar = t
                    else: position = 0
            elif position == -1:
                if zone_bull or (ti > 0 and not zone_bull and not zone_bear):
                    pnl = (entry_price - close[t]) / entry_price * 100.0 - commission_pct
                    equity *= (1 + pnl / 100.0)
                    n_trades += 1
                    # --- v5e cluster accounting (additive, no afecta equity) ---
                    cl_idx = cluster_labels[entry_bar]
                    if 0 <= cl_idx < n_clusters:
                        cl_pnl[cfg_idx, cl_idx] += pnl
                        cl_trades[cfg_idx, cl_idx] += 1
                        if pnl > 0:
                            cl_gp[cfg_idx, cl_idx] += pnl
                        else:
                            cl_gl[cfg_idx, cl_idx] += (-pnl)
                    # --- fin cluster accounting ---
                    if zone_bull: position = 1; entry_price = close[t]; entry_bar = t
                    else: position = 0

        pnl_arr[cfg_idx] = equity - 100.0
        trades_arr[cfg_idx] = n_trades
        raw_crosses_arr[cfg_idx] = n_raw_crosses

    return pnl_arr, trades_arr, raw_crosses_arr, cl_pnl, cl_trades, cl_gp, cl_gl


# ============================================
# PROCESS SYMBOL
# ============================================
def process_symbol(symbol):
    # Calcular warmup antes de descargar para que TOTAL_CANDLES = velas medidas
    fast_cat_pre, slow_cat_pre, trend_cat_pre = build_catalog()
    max_slow_period = max(p for _, p, _, _ in slow_cat_pre)
    max_trend_period = max((p for _, p, _, _ in trend_cat_pre if p > 0), default=0)
    max_period = max(max_slow_period, max_trend_period)
    warmup = max_period * 3
    total_download = TOTAL_CANDLES + warmup

    candles = fetch_all_candles(symbol, TIMEFRAME, total_download)
    if len(candles) < TOTAL_CANDLES:
        print(f"   ❌ Solo {len(candles)} velas")
        return None

    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    print(f"   📊 {len(df)} velas descargadas: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

    close, fast_arr, slow_arr, trend_arr, fast_cat, slow_cat, trend_cat = precalculate_all_mas(df)
    n_fast, n_slow, n_trend = len(fast_cat), len(slow_cat), len(trend_cat)
    n_configs = n_fast * n_slow * n_trend
    n_bars = len(close)
    n_measured = n_bars - warmup

    test_start = warmup
    print(f"   📐 Warmup={warmup} (max_period={max_period} × 3) | Velas medidas: {n_measured}")

    # --- v5e: Regime clustering (modelos pre-entrenados) ---
    use_clustering = USE_CLUSTERING and HAS_SKLEARN and HAS_JOBLIB
    cluster_info = None
    n_clusters_actual = 1  # fallback: todo en cluster 0

    if use_clustering:
        t0_cl = time.time()
        cluster_labels, cluster_info = load_and_apply_regime_model(
            symbol, df, model_dir=REGIME_MODEL_DIR)
        n_clusters_actual = cluster_info['n_clusters']
        elapsed_cl = time.time() - t0_cl
        print(f"   🔬 Clustering: {cluster_info['method']} → {n_clusters_actual} clusters ({elapsed_cl:.1f}s)")
        for k in range(n_clusters_actual):
            name = cluster_info['cluster_names'][k] if cluster_info.get('cluster_names') else f"C{k}"
            size = cluster_info['cluster_sizes'][k]
            pct = size / sum(cluster_info['cluster_sizes']) * 100
            centroid = cluster_info['centroids'][k] if cluster_info.get('centroids') is not None else None
            if centroid is not None:
                print(f"      Cluster {k} ({name}): {size} bars ({pct:.0f}%)"
                      f" — H={centroid[0]:.3f} NATR={centroid[1]:.4f} VI={centroid[2]:.3f}")
            else:
                print(f"      Cluster {k}: {size} bars ({pct:.0f}%)")
    else:
        # No clustering: all bars = cluster 0
        cluster_labels = np.zeros(n_bars, dtype=np.int32)

    cluster_labels_np = cluster_labels.astype(np.int64)

    # EMA reference PnL (with cluster tracking)
    print(f"   📏 Calculando referencia EMA({EMA_FAST}/{EMA_SLOW}/{EMA_TREND})...")
    ema_f = calc_ema(close, EMA_FAST)
    ema_s = calc_ema(close, EMA_SLOW)
    ema_t = calc_ema(close, EMA_TREND)
    ema_pnl, ema_trades, ema_raw_crosses, ema_cl_pnl, ema_cl_trades = simulate_ema_pnl(
        close, ema_f, ema_s, ema_t, COMMISSION_ROUND_TRIP, test_start, warmup,
        cluster_labels_np, n_clusters_actual)
    print(f"   📏 EMA ref: PnL={ema_pnl:+.2f}% | {ema_trades} trades | {ema_raw_crosses} raw crosses")
    if use_clustering:
        for k in range(n_clusters_actual):
            name = cluster_info['cluster_names'][k] if cluster_info.get('cluster_names') else f"C{k}"
            print(f"      EMA C{k}({name}): PnL={ema_cl_pnl[k]:+.2f}% | {ema_cl_trades[k]} trades")

    # Run all candidates
    print(f"   🚀 Simulando {n_configs:,} configs (v5e, {n_clusters_actual} clusters)...")
    t0 = time.time()
    cand_pnl, cand_trades, cand_raw_crosses, cl_pnl, cl_trades, cl_gp, cl_gl = \
        run_simulation_v5e(
            close, fast_arr, slow_arr, trend_arr,
            n_fast, n_slow, n_trend,
            COMMISSION_ROUND_TRIP, test_start, warmup,
            cluster_labels_np, n_clusters_actual)
    elapsed = time.time() - t0
    print(f"   ⏱️  {elapsed:.1f}s ({n_configs/elapsed:,.0f} configs/s)")

    # --- GLOBAL SCORING (identical to v5d) ---
    MIN_TRADES = 10
    valid = cand_trades >= MIN_TRADES

    excess_cross = np.where(ema_raw_crosses > 0,
                            np.maximum(0.0, (cand_raw_crosses.astype(np.float64) - ema_raw_crosses) / ema_raw_crosses),
                            0.0)
    excess_cross = np.minimum(excess_cross, 1.0)
    cross_penalty = 1.0 - 0.3 * excess_cross

    if ema_pnl > 0:
        pnl_ratio = cand_pnl / ema_pnl
    else:
        pnl_ratio = cand_pnl / 10.0

    score = pnl_ratio * cross_penalty
    score_valid = np.where(valid, score, -999.0)

    n_valid = int(np.sum(valid))
    n_beating_ema = int(np.sum((cand_pnl > ema_pnl) & valid))
    print(f"   📊 Válidas (≥{MIN_TRADES}t): {n_valid:,} | Superan EMA: {n_beating_ema:,} ({n_beating_ema/max(n_valid,1)*100:.1f}%)")

    sorted_idx = np.argsort(-score_valid)
    bnh = (close[-1] - close[warmup]) / close[warmup] * 100
    test_years = (n_bars - warmup) / 8760.0

    # --- CLUSTER STATS SUMMARY ---
    if use_clustering:
        print(f"   📊 Stats por cluster:")
        for k in range(n_clusters_actual):
            name = cluster_info['cluster_names'][k] if cluster_info.get('cluster_names') else f"C{k}"
            has_trades = cl_trades[:, k] >= CLUSTER_MIN_TRADES
            n_with_trades = int(np.sum(has_trades))
            if n_with_trades > 0:
                gp_k = cl_gp[has_trades, k]
                gl_k = cl_gl[has_trades, k]
                pf_k = np.where(gl_k > 0, gp_k / gl_k, 0.0)
                good_pf = np.sum(pf_k >= CLUSTER_MIN_PF)
                print(f"      C{k}({name}): {n_with_trades:,} configs con ≥{CLUSTER_MIN_TRADES}t"
                      f" | {int(good_pf):,} con PF≥{CLUSTER_MIN_PF}")
            else:
                print(f"      C{k}({name}): ⚠️ ninguna config con ≥{CLUSTER_MIN_TRADES} trades")

    # --- Ranking file ---
    sym_clean = symbol.replace("/", "")
    ranking_file = os.path.join(OUTPUT_DIR, f"ranking_lite_v5e_{sym_clean}.txt")
    with open(ranking_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*140}\n")
        f.write(f"RANKING LITE v5e — CLUSTERING DE REGÍMENES — {symbol}\n")
        f.write(f"Score = (PnL_cand/PnL_ema) × (1 - 0.3 × ExcessCrossRate)  [EMA<=0: PnL/10]\n")
        f.write(f"Referencia: EMA({EMA_FAST}/{EMA_SLOW}/{EMA_TREND}) → PnL={ema_pnl:+.2f}% | {ema_trades}t | {ema_raw_crosses} rawX\n")
        f.write(f"{'='*140}\n")
        f.write(f"Velas: {len(df)} | {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}\n")
        f.write(f"Warmup: {warmup} | Velas medidas: {n_bars - warmup} ({test_years:.2f}yr)\n")
        f.write(f"Configs: {n_configs:,} | Válidas (≥{MIN_TRADES}t): {n_valid:,} | Superan EMA: {n_beating_ema:,}\n")
        f.write(f"Comisiones: {COMMISSION_ROUND_TRIP}% | B&H: {bnh:+.1f}%\n")
        f.write(f"Tiempo: {elapsed:.1f}s\n")

        # Cluster info in ranking file
        if use_clustering and cluster_info:
            f.write(f"\n{'='*80}\nCLUSTERING DE REGÍMENES\n{'='*80}\n")
            f.write(f"Método: {cluster_info['method']}\n")
            f.write(f"Features: {', '.join(cluster_info['feature_names'])}\n")
            if cluster_info.get('training_bars'):
                f.write(f"Training bars: {cluster_info['training_bars']}\n")
            if cluster_info.get('training_date_range'):
                f.write(f"Training range: {cluster_info['training_date_range'][0][:10]}"
                        f" → {cluster_info['training_date_range'][1][:10]}\n")
            f.write(f"\n")
            for k in range(n_clusters_actual):
                name = cluster_info['cluster_names'][k] if cluster_info.get('cluster_names') else f"C{k}"
                size = cluster_info['cluster_sizes'][k]
                pct = size / sum(cluster_info['cluster_sizes']) * 100
                f.write(f"  Cluster {k} ({name}): {size} bars ({pct:.0f}%)\n")
                if cluster_info.get('centroids') is not None:
                    centroid = cluster_info['centroids'][k]
                    f.write(f"    Centroides: Hurst={centroid[0]:.3f} NATR={centroid[1]:.4f} VolImb={centroid[2]:.3f}\n")
                f.write(f"    EMA ref: PnL={ema_cl_pnl[k]:+.2f}% | {ema_cl_trades[k]} trades\n")

        f.write(f"\n{'='*140}\nRANKING GLOBAL (backward compatible con v5d)\n{'='*140}\n\n")

        count = 0
        for idx in sorted_idx:
            if not valid[idx]: continue
            if count >= TOP_SAVE: break
            fi = idx // (n_slow * n_trend)
            remainder = idx % (n_slow * n_trend)
            si = remainder // n_trend
            ti = remainder % n_trend
            rawX_ratio = cand_raw_crosses[idx] / ema_raw_crosses if ema_raw_crosses > 0 else 0
            tpy = cand_trades[idx] / test_years if test_years > 0 else 0
            pnl_vs = cand_pnl[idx] - ema_pnl
            if count < 200:
                f.write(f"#{count+1:4d} | Score={score[idx]:.4f}"
                        f" | PnL={cand_pnl[idx]:+.2f}% (vs EMA {pnl_vs:+.2f}%)"
                        f" | {int(cand_trades[idx])}t ({tpy:.0f}/yr)"
                        f" | RawX={int(cand_raw_crosses[idx])}({rawX_ratio:.2f}x)"
                        f" | XPen={excess_cross[idx]:.2f}\n")
                f.write(f"      Fast: {ma_name(fast_cat[fi])}"
                        f" | Slow: {ma_name(slow_cat[si])}"
                        f" | Trend: {ma_name(trend_cat[ti])}\n")
                # Cluster decomposition
                if use_clustering:
                    parts = []
                    for k in range(n_clusters_actual):
                        ct = cl_trades[idx, k]
                        cp = cl_pnl[idx, k]
                        if ct > 0:
                            gp_k = cl_gp[idx, k]
                            gl_k = cl_gl[idx, k]
                            pf_k = gp_k / gl_k if gl_k > 0 else (99.0 if gp_k > 0 else 0.0)
                            name = cluster_info['cluster_names'][k]
                            parts.append(f"C{k}({name}): PnL={cp:+.1f}% {ct}t PF={pf_k:.2f}")
                    if parts:
                        f.write(f"      Clusters: {' | '.join(parts)}\n")
                f.write("\n")
            count += 1

        # --- PER-CLUSTER RANKINGS ---
        if use_clustering:
            f.write(f"\n{'='*140}\nRANKING POR CLUSTER (especialistas)\n{'='*140}\n")
            for k in range(n_clusters_actual):
                name = cluster_info['cluster_names'][k]
                size = cluster_info['cluster_sizes'][k]
                f.write(f"\n{'='*80}\nCluster {k} ({name}) — {size} bars\n{'='*80}\n\n")

                # Configs con suficientes trades en este cluster
                has_trades_k = cl_trades[:, k] >= CLUSTER_MIN_TRADES
                indices_k = np.where(has_trades_k)[0]
                if len(indices_k) == 0:
                    f.write("  ⚠️ Ninguna config con trades suficientes en este cluster\n")
                    continue

                # PF por cluster
                gp_k = cl_gp[indices_k, k]
                gl_k = cl_gl[indices_k, k]
                pf_k = np.where(gl_k > 0, gp_k / gl_k, np.where(gp_k > 0, 99.0, 0.0))

                # Rankear por PF
                sort_pf = np.argsort(-pf_k)

                shown = 0
                for rank_pos in range(len(sort_pf)):
                    if shown >= 50: break
                    cfg_idx = indices_k[sort_pf[rank_pos]]
                    this_pf = pf_k[sort_pf[rank_pos]]
                    if this_pf < CLUSTER_MIN_PF: break

                    fi = cfg_idx // (n_slow * n_trend)
                    remainder = cfg_idx % (n_slow * n_trend)
                    si = remainder // n_trend
                    ti = remainder % n_trend

                    f.write(f"  #{shown+1:3d} | PF={this_pf:.2f}"
                            f" | PnL_C{k}={cl_pnl[cfg_idx, k]:+.2f}%"
                            f" | {cl_trades[cfg_idx, k]}t"
                            f" | PnL_global={cand_pnl[cfg_idx]:+.2f}%"
                            f" | {ma_name(fast_cat[fi])} / {ma_name(slow_cat[si])}"
                            f" / {ma_name(trend_cat[ti])}\n")
                    shown += 1

                f.write(f"\n  Total configs con ≥{CLUSTER_MIN_TRADES}t y PF≥{CLUSTER_MIN_PF}: "
                        f"{int(np.sum(pf_k >= CLUSTER_MIN_PF))}\n")

    print(f"   💾 {ranking_file}")

    # --- Top results (global, backward compatible) ---
    top_results = []
    count = 0
    for idx in sorted_idx:
        if not valid[idx]: continue
        if count >= TOP_SAVE: break
        fi = idx // (n_slow * n_trend)
        remainder = idx % (n_slow * n_trend)
        si = remainder // n_trend
        ti = remainder % n_trend
        top_results.append({
            'fast': fast_cat[fi], 'slow': slow_cat[si], 'trend': trend_cat[ti],
            'score': float(score[idx]), 'pnl': float(cand_pnl[idx]),
            'trades': int(cand_trades[idx]),
            'raw_crosses': int(cand_raw_crosses[idx]),
            'excess_cross': float(excess_cross[idx]),
        })
        count += 1

    # --- SELECCIÓN GLOBAL (backward compatible con v5d) ---
    if DIVERSITY_MODE:
        selected_global = select_top_diverse(top_results, TOP_N_PRESETS)
        mode_str = "DIVERSE"
    else:
        selected_global = top_results[:TOP_N_PRESETS]
        mode_str = "TOP"

    # --- SELECCIÓN PER-CLUSTER (v5e nuevo) ---
    cluster_selected = {}
    if use_clustering:
        cluster_selected = select_top_per_cluster(
            cand_pnl, cl_pnl, cl_trades, cl_gp, cl_gl,
            n_clusters_actual, fast_cat, slow_cat, trend_cat,
            n_slow, n_trend,
            min_trades=CLUSTER_MIN_TRADES, min_pf=CLUSTER_MIN_PF,
            top_n=CLUSTER_TOP_N_PER)

    # --- CSV de presets (formato extendido con cluster info) ---
    csv_file = os.path.join(OUTPUT_DIR, f"presets_{sym_clean}.csv")
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("rank,fast_type,fast_period,fast_p1,fast_p2,"
                "slow_type,slow_period,slow_p1,slow_p2,"
                "trend_type,trend_period,trend_p1,trend_p2,"
                "score,pnl,trades,raw_crosses,excess_cross,family,"
                "cluster_id,cluster_pf,cluster_pnl,cluster_trades\n")

        # Global presets (cluster_id = -1 means "global/generalist")
        for r, res in enumerate(selected_global, 1):
            pt = result_to_preset_tuple(res)
            fam = f"{ma_family(res['fast'])}/{ma_family(res['slow'])}"
            f.write(f"{r},{pt[0]},{pt[1]},{pt[2]},{pt[3]},"
                    f"{pt[4]},{pt[5]},{pt[6]},{pt[7]},"
                    f"{pt[8]},{pt[9]},{pt[10]},{pt[11]},"
                    f"{res['score']:.6f},{res['pnl']:.4f},{res['trades']},"
                    f"{res['raw_crosses']},{res['excess_cross']:.4f},{fam},"
                    f"-1,,,\n")

        # Per-cluster presets
        if use_clustering:
            for k in sorted(cluster_selected.keys()):
                for r, res in enumerate(cluster_selected[k], 1):
                    pt = result_to_preset_tuple(res)
                    fam = f"{ma_family(res['fast'])}/{ma_family(res['slow'])}"
                    f.write(f"C{k}_{r},{pt[0]},{pt[1]},{pt[2]},{pt[3]},"
                            f"{pt[4]},{pt[5]},{pt[6]},{pt[7]},"
                            f"{pt[8]},{pt[9]},{pt[10]},{pt[11]},"
                            f"{res.get('score', 0):.6f},{res.get('global_pnl', 0):.4f},"
                            f"{res.get('cluster_trades', 0)},"
                            f"0,0.0000,{fam},"
                            f"{k},{res.get('cluster_pf', 0):.4f},"
                            f"{res.get('cluster_pnl', 0):.4f},"
                            f"{res.get('cluster_trades', 0)}\n")

    print(f"   💾 {csv_file}")

    # --- Print summary ---
    selected_presets_global = [result_to_preset_tuple(res) for res in selected_global]

    families = set()
    for res in selected_global:
        families.add(f"{ma_family(res['fast'])}/{ma_family(res['slow'])}")

    print(f"\n   🏆 GLOBAL {mode_str} {len(selected_global)} PRESETS ({len(families)} familias)"
          f" (EMA ref: PnL={ema_pnl:+.2f}%, {ema_trades}t, {ema_raw_crosses}rawX):")
    for r, res in enumerate(selected_global, 1):
        pnl_vs = res['pnl'] - ema_pnl
        rx = res['raw_crosses'] / ema_raw_crosses if ema_raw_crosses > 0 else 0
        fam = f"{ma_family(res['fast'])}/{ma_family(res['slow'])}"
        print(f"      #{r} Score={res['score']:.4f} PnL={res['pnl']:+.2f}% ({pnl_vs:+.2f}vs)"
              f" {res['trades']}t RawX={res['raw_crosses']}({rx:.2f}x)"
              f" | {ma_name(res['fast'])} / {ma_name(res['slow'])} / {ma_name(res['trend'])}"
              f" [{fam}]")

    if use_clustering and cluster_selected:
        print(f"\n   🎯 ESPECIALISTAS POR CLUSTER:")
        for k in sorted(cluster_selected.keys()):
            name = cluster_info['cluster_names'][k] if cluster_info.get('cluster_names') else f"C{k}"
            presets_k = cluster_selected[k]
            if not presets_k:
                print(f"      C{k}({name}): ⚠️ sin especialistas (insuficientes trades o PF)")
                continue
            print(f"      C{k}({name}): {len(presets_k)} especialistas")
            for r, res in enumerate(presets_k, 1):
                fam = f"{ma_family(res['fast'])}/{ma_family(res['slow'])}"
                print(f"         #{r} PF={res['cluster_pf']:.2f}"
                      f" PnL_C{k}={res['cluster_pnl']:+.2f}%"
                      f" {res['cluster_trades']}t"
                      f" | {ma_name(res['fast'])} / {ma_name(res['slow'])}"
                      f" [{fam}]")

    # Combinar presets globales + cluster para devolver
    all_selected_presets = list(selected_presets_global)
    if use_clustering:
        for k in sorted(cluster_selected.keys()):
            for res in cluster_selected[k]:
                pt = result_to_preset_tuple(res)
                if pt not in all_selected_presets:
                    all_selected_presets.append(pt)

    return {'symbol': symbol, 'top': top_results, 'bnh': bnh,
            'ema_pnl': ema_pnl, 'ema_trades': ema_trades, 'ema_raw_crosses': ema_raw_crosses,
            'selected': selected_global, 'selected_presets': all_selected_presets,
            'cluster_selected': cluster_selected, 'cluster_info': cluster_info}


# ============================================
# CROSS-SYMBOL ANALYSIS v5e
# ============================================
def cross_symbol_analysis(all_results):
    print(f"\n{'='*140}")
    print(f"🔬 ANÁLISIS CROSS-SYMBOL v5e — PnL RELATIVO A EMAs + CLUSTERING")
    print(f"{'='*140}\n")
    from collections import Counter, defaultdict
    n_syms = len(all_results)
    TOP_N = TOP_SAVE

    outfile = os.path.join(OUTPUT_DIR, "cross_symbol_v5e.txt")
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write(f"{'='*140}\n")
        f.write(f"ANÁLISIS CROSS-SYMBOL v5e — PnL RELATIVO A EMAs + CLUSTERING\n")
        f.write(f"Símbolos: {n_syms} | Ref: EMA({EMA_FAST}/{EMA_SLOW}/{EMA_TREND})\n")
        f.write(f"{'='*140}\n\n")

        # EMA summary per symbol
        f.write(f"REFERENCIA EMA POR SÍMBOLO:\n")
        for result in sorted(all_results, key=lambda x: x['symbol']):
            sym = result['symbol'].replace('/USDT', '')
            f.write(f"  {sym:8s}: PnL={result['ema_pnl']:+.2f}% | {result['ema_trades']}t | {result['ema_raw_crosses']} rawX\n")

        # Cluster summary per symbol
        f.write(f"\n{'='*80}\nCLUSTERING POR SÍMBOLO\n{'='*80}\n\n")
        for result in sorted(all_results, key=lambda x: x['symbol']):
            sym = result['symbol'].replace('/USDT', '')
            ci = result.get('cluster_info')
            if ci and ci.get('centroids') is not None:
                f.write(f"  {sym:8s}: {ci['method']}\n")
                for k in range(ci['n_clusters']):
                    name = ci['cluster_names'][k]
                    size = ci['cluster_sizes'][k]
                    pct = size / sum(ci['cluster_sizes']) * 100
                    c = ci['centroids'][k]
                    f.write(f"    C{k}({name}): {size} bars ({pct:.0f}%)"
                            f" H={c[0]:.3f} NATR={c[1]:.4f} VI={c[2]:.3f}\n")
            else:
                f.write(f"  {sym:8s}: clustering no disponible\n")

        # Familia por rol (identical to v5d)
        for label, role in [("FAST", "fast"), ("SLOW", "slow"), ("TREND", "trend")]:
            f.write(f"\n{'='*80}\n{label}\n{'='*80}\n")
            ts = Counter(); t_syms = defaultdict(set)
            t_pnls = defaultdict(list)
            for result in all_results:
                sym = result['symbol']
                for cfg in result['top'][:TOP_N]:
                    k = ma_family(cfg[role])
                    ts[k] += cfg['score']; t_syms[k].add(sym)
                    t_pnls[k].append(cfg['pnl'])
            total = sum(ts.values()) or 1
            for t, sc in ts.most_common(16):
                ns = len(t_syms[t]); pct = sc / total * 100
                f.write(f"    {t:12s}: Score={sc:10.2f} ({pct:5.1f}%) | {ns:2d}/{n_syms} syms"
                        f" | AvgPnL={np.mean(t_pnls[t]):+.2f}%\n")

        # Top 30 familias
        f.write(f"\n{'='*140}\nTOP 30 FAMILIAS\n{'='*140}\n\n")
        fam_scores = Counter(); fam_syms = defaultdict(set)
        fam_pnls = defaultdict(list); fam_rawx = defaultdict(list)
        for result in all_results:
            sym = result['symbol']
            for cfg in result['top'][:TOP_N]:
                fam = f"{ma_family(cfg['fast'])}/{ma_family(cfg['slow'])}/{ma_family(cfg['trend'])}"
                fam_scores[fam] += cfg['score']; fam_syms[fam].add(sym)
                fam_pnls[fam].append(cfg['pnl']); fam_rawx[fam].append(cfg['raw_crosses'])
        for rank, (fam, sc) in enumerate(fam_scores.most_common(30), 1):
            ns = len(fam_syms[fam])
            f.write(f"  #{rank:2d} | Score={sc:8.2f} | {ns:2d}/{n_syms} syms"
                    f" | AvgPnL={np.mean(fam_pnls[fam]):+.2f}%"
                    f" | AvgRawX={np.mean(fam_rawx[fam]):.0f} | {fam}\n")

        # Top 30 combos exactos
        f.write(f"\n{'='*140}\nTOP 30 COMBINACIONES EXACTAS\n{'='*140}\n\n")
        cs = Counter(); c_syms = defaultdict(set)
        c_pnls = defaultdict(list); c_trades = defaultdict(list); c_rawx = defaultdict(list)
        for result in all_results:
            sym = result['symbol']
            for cfg in result['top'][:TOP_N]:
                key = (cfg['fast'], cfg['slow'], cfg['trend'])
                cs[key] += cfg['score']; c_syms[key].add(sym)
                c_pnls[key].append(cfg['pnl']); c_trades[key].append(cfg['trades'])
                c_rawx[key].append(cfg['raw_crosses'])
        for rank, (key, sc) in enumerate(cs.most_common(30), 1):
            fast, slow, trend = key; ns = len(c_syms[key])
            f.write(f"  #{rank:2d} | Score={sc:8.4f} | {ns:2d}/{n_syms} syms"
                    f" | PnL={np.mean(c_pnls[key]):+.2f}% Trd={np.mean(c_trades[key]):.0f}"
                    f" RawX={np.mean(c_rawx[key]):.0f}"
                    f" | {ma_name(fast)} / {ma_name(slow)} / {ma_name(trend)}\n")
            if rank <= 5:
                syms_list = sorted(s.replace('/USDT', '') for s in c_syms[key])
                f.write(f"       Símbolos: {', '.join(syms_list)}\n")

        # Convergencia períodos top 5
        f.write(f"\n{'='*140}\nCONVERGENCIA DE PERÍODOS — Top 5 familias\n{'='*140}\n")
        fp = defaultdict(lambda: {'fast': Counter(), 'slow': Counter(), 'trend': Counter()})
        for result in all_results:
            for cfg in result['top'][:TOP_N]:
                fam = f"{ma_family(cfg['fast'])}/{ma_family(cfg['slow'])}/{ma_family(cfg['trend'])}"
                fp[fam]['fast'][cfg['fast'][1]] += cfg['score']
                fp[fam]['slow'][cfg['slow'][1]] += cfg['score']
                fp[fam]['trend'][cfg['trend'][1]] += cfg['score']
        for rank, (fam, sc) in enumerate(fam_scores.most_common(5), 1):
            ns = len(fam_syms[fam])
            f.write(f"\n  #{rank} {fam} ({ns}/{n_syms} syms, Score={sc:.2f})\n")
            for role in ['fast', 'slow', 'trend']:
                periods = fp[fam][role]; total_p = sum(periods.values()) or 1
                top_p = periods.most_common(5)
                parts = [f"{p}({s/total_p*100:.0f}%)" for p, s in top_p]
                conv = ""
                if top_p:
                    pct = top_p[0][1] / total_p * 100
                    if pct > 30: conv = f"  ✅ converge en {top_p[0][0]}"
                    elif pct > 20: conv = f"  ~ tendencia hacia {top_p[0][0]}"
                    else: conv = f"  ⚠️ disperso"
                f.write(f"    {role.upper()}: {' > '.join(parts)}{conv}\n")

        # #1 por símbolo
        f.write(f"\n{'='*140}\n#1 POR SÍMBOLO (vs EMA ref)\n{'='*140}\n\n")
        for result in sorted(all_results, key=lambda x: x['symbol']):
            if result['top']:
                cfg = result['top'][0]; sym = result['symbol'].replace('/USDT', '')
                pnl_vs = cfg['pnl'] - result['ema_pnl']
                rx = cfg['raw_crosses'] / result['ema_raw_crosses'] if result['ema_raw_crosses'] > 0 else 0
                f.write(f"  {sym:8s}: Score={cfg['score']:.4f}"
                        f" PnL={cfg['pnl']:+.2f}% ({pnl_vs:+.2f}vs EMA={result['ema_pnl']:+.2f}%)"
                        f" {cfg['trades']}t RawX({rx:.2f}x)"
                        f" | {ma_name(cfg['fast'])} / {ma_name(cfg['slow'])} / {ma_name(cfg['trend'])}\n")

        # Cluster specialist overlap across symbols
        if any(r.get('cluster_selected') for r in all_results):
            f.write(f"\n{'='*140}\nESPECIALISTAS POR CLUSTER — CROSS-SYMBOL\n{'='*140}\n\n")
            for result in sorted(all_results, key=lambda x: x['symbol']):
                sym = result['symbol'].replace('/USDT', '')
                cs_sel = result.get('cluster_selected', {})
                ci = result.get('cluster_info')
                if not cs_sel or not ci:
                    continue
                f.write(f"  {sym}:\n")
                for k in sorted(cs_sel.keys()):
                    name = ci['cluster_names'][k] if ci.get('cluster_names') else f"C{k}"
                    presets = cs_sel[k]
                    if presets:
                        families_k = [f"{ma_family(p['fast'])}/{ma_family(p['slow'])}" for p in presets]
                        pfs = [f"PF={p['cluster_pf']:.2f}" for p in presets]
                        f.write(f"    C{k}({name}): {', '.join(f'{fam}({pf})' for fam, pf in zip(families_k, pfs))}\n")
                    else:
                        f.write(f"    C{k}({name}): sin especialistas\n")

    print(f"   💾 {outfile}")
    print(f"\n   🏆 TOP 10 FAMILIAS:")
    for rank, (fam, sc) in enumerate(fam_scores.most_common(10), 1):
        ns = len(fam_syms[fam])
        print(f"      #{rank} Score={sc:8.2f} ({ns:2d} syms) PnL={np.mean(fam_pnls[fam]):+.2f}% | {fam}")


# ============================================
# MAIN
# ============================================
def main():
    fast_cat, slow_cat, trend_cat = build_catalog()
    n_configs = len(fast_cat) * len(slow_cat) * len(trend_cat)
    print("="*70)
    print("🧪 LABORATORIO LITE v5e — CLUSTERING DE REGÍMENES")
    print(f"   Ref: EMA({EMA_FAST}/{EMA_SLOW}/{EMA_TREND})")
    print(f"   Score = PnL × (1 - 0.3×ExcessCrossRate)")
    print(f"   Total: {n_configs:,} configs × {len(SYMBOLS)} símbolos")
    print(f"   Comisiones: {COMMISSION_ROUND_TRIP}%")
    if USE_CLUSTERING and HAS_SKLEARN and HAS_JOBLIB:
        print(f"   Clustering: ON (pre-trained models from {REGIME_MODEL_DIR}/,"
              f" min_trades={CLUSTER_MIN_TRADES}, top_per_cluster={CLUSTER_TOP_N_PER})")
        # Warm up sklearn/OpenBLAS to avoid ~40s penalty on first symbol (Windows)
        GaussianMixture(n_components=2, n_init=1, max_iter=1, random_state=42).fit(
            np.random.randn(20, 3))
    else:
        reasons = []
        if not HAS_SKLEARN: reasons.append("sklearn no disponible")
        if not HAS_JOBLIB: reasons.append("joblib no disponible")
        if not USE_CLUSTERING: reasons.append("desactivado")
        print(f"   Clustering: OFF ({', '.join(reasons) if reasons else 'desactivado'})")
    print("="*70)
    all_results = []
    failed = []
    for i, symbol in enumerate(SYMBOLS, 1):
        print(f"\n[{i}/{len(SYMBOLS)}] {symbol}")
        try:
            result = process_symbol(symbol)
            if result: all_results.append(result)
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback; traceback.print_exc()
            failed.append(symbol)
    if len(all_results) >= 3:
        cross_symbol_analysis(all_results)
    print(f"\n{'='*70}")
    print(f"✅ Completado: {len(all_results)}/{len(SYMBOLS)} símbolos")
    if failed: print(f"❌ Fallidos: {', '.join(failed)}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
