#!/usr/bin/env python3
"""
train_regime_model.py - Entrena modelos de régimen GMM por símbolo sobre TODO el histórico.

Flujo:
  1. Para cada símbolo, carga todo el histórico 1h disponible (data_cache/)
  2. Calcula features rolling estacionarias (Hurst, Z_ATR, ER) con lookback=100/1000
  3. Entrena GMM con k=2 y k=3, selecciona por BIC
  4. Guarda modelo en regime_models/SYMBOL_regime.joblib
  5. Genera reporte regime_models/regime_report.txt

Uso:
  python train_regime_model.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Fix Windows cp1252 encoding for emoji output
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith('cp'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from regime_features import compute_regime_features, FEATURE_NAMES

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

LOOKBACK = 100
MAX_K = 3
MODEL_DIR = "regime_models"
TIMEFRAME = "1h"
CACHE_DIR = "data_cache"

# ============================================
# Frame 3.A meta-redesign Eje 1 — HMM CONDITIONAL γ.6 paradigm shift flag
# ============================================
# _HMM_ENABLED=False (default): preserve baseline GMM productive path absoluto backwards compat.
# _HMM_ENABLED=True: drop-in custom Numba HMM Gaussian (hmm_gaussian_numba.py)
#                    paradigm shift Markov regime persistence + transitions matrix.
# Validated robust split A 15/15 tests + split B identical-init atol 1e-7 cross-mature library.
# Activable per-call OR per-process via os.environ['HMM_ENABLED']='1'.
_HMM_ENABLED = os.environ.get('HMM_ENABLED', '0') == '1'


def _symbol_key(symbol):
    """BTC/USDT → BTC"""
    return symbol.replace("/USDT", "").replace("/", "")


def _cache_path(symbol):
    """Retorna path del parquet cacheado."""
    sc = symbol.replace("/", "")
    return os.path.join(CACHE_DIR, f"{sc}_{TIMEFRAME}.parquet")


def _download_full_history(symbol):
    """Descarga TODO el histórico 1h disponible desde Binance via ccxt.

    Itera con limit=1000 desde el timestamp más antiguo disponible hasta ahora.
    Guarda resultado en data_cache/ en formato parquet compatible con data_cache.py.

    Returns: DataFrame o None.
    """
    try:
        import ccxt
    except ImportError:
        print(f"   ❌ ccxt no disponible — no se puede descargar {symbol}")
        return None

    exchange = ccxt.binance({'enableRateLimit': True})
    interval_ms = 3600000  # 1h

    print(f"   📥 Descargando histórico completo de {symbol} desde Binance...")
    all_candles = []
    # Empezar desde 2017-01-01 para capturar todo el histórico posible
    current_since = int(pd.Timestamp('2017-01-01').timestamp() * 1000)
    requests_made = 0
    consecutive_errors = 0

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=current_since, limit=1000)
            if not ohlcv:
                break
            all_candles.extend(ohlcv)
            current_since = ohlcv[-1][0] + interval_ms
            requests_made += 1
            consecutive_errors = 0
            if requests_made % 50 == 0:
                print(f"      ... {len(all_candles)} velas descargadas ({requests_made} requests)")
            time.sleep(0.1)
        except Exception as e:
            consecutive_errors += 1
            if consecutive_errors >= 3:
                print(f"   ⚠️  Error tras 3 reintentos: {e}")
                break
            time.sleep(2 * consecutive_errors)

    if not all_candles:
        print(f"   ❌ Sin datos descargados para {symbol}")
        return None

    df = pd.DataFrame(all_candles, columns=['timestamp_ms', 'open', 'high', 'low', 'close', 'volume'])
    df = df.drop_duplicates(subset=['timestamp_ms']).sort_values('timestamp_ms').reset_index(drop=True)
    print(f"   📥 {len(df)} velas descargadas ({requests_made} requests)")

    # Guardar en cache (formato compatible con data_cache.py)
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(symbol)
    df.to_parquet(path, index=False)
    print(f"   💾 Cache guardado: {path}")

    return df


def load_full_history(symbol):
    """Carga TODO el histórico disponible en data_cache/ para un símbolo.

    Si no existe en cache, descarga desde Binance y guarda para futuras ejecuciones.

    Returns: DataFrame con columns [timestamp, open, high, low, close, volume] o None.
    """
    path = _cache_path(symbol)

    if os.path.exists(path):
        df = pd.read_parquet(path)
        if len(df) > 0:
            # Convertir a formato estándar
            result = df.copy()
            if 'timestamp_ms' in result.columns:
                result['timestamp'] = pd.to_datetime(result['timestamp_ms'], unit='ms')
            if 'timestamp' not in result.columns:
                print(f"   ⚠️  Sin columna timestamp para {symbol}")
                return None
            result = result[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            return result

    # Fallback: descargar desde Binance
    print(f"   ⚠️  Sin datos en cache para {symbol} — descargando desde Binance...")
    df = _download_full_history(symbol)
    if df is None:
        return None

    # Convertir a formato estándar
    result = df.copy()
    if 'timestamp_ms' in result.columns:
        result['timestamp'] = pd.to_datetime(result['timestamp_ms'], unit='ms')
    result = result[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    return result


def train_symbol_model(symbol, df):
    """Entrena modelo de régimen GMM para un símbolo sobre todo su histórico.

    Returns: dict con modelo y metadata, o None si falla.
    """
    sym_key = _symbol_key(symbol)
    n_bars = len(df)
    print(f"   📊 {n_bars} barras: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

    # Calcular features
    t0 = time.time()
    features, valid_mask = compute_regime_features(df, lookback=LOOKBACK)
    elapsed_feat = time.time() - t0

    valid_features = features[valid_mask]
    n_valid = len(valid_features)
    print(f"   🔬 Features calculadas: {n_valid}/{n_bars} válidas ({elapsed_feat:.1f}s)")

    if n_valid < MAX_K * 50:
        print(f"   ⚠️  Insuficientes barras válidas ({n_valid} < {MAX_K * 50})")
        return None

    # Estandarizar
    scaler = StandardScaler()
    X = scaler.fit_transform(valid_features)

    # Entrenar k=2..MAX_K, seleccionar por BIC
    # Branching _HMM_ENABLED:
    #   False (default): GMM productive path baseline absoluto.
    #   True: custom Numba HMM Gaussian paradigm shift (Frame 3.A meta-redesign Eje 1).
    best_bic = np.inf
    best_k = 2
    best_gmm = None
    bic_values = {}

    if _HMM_ENABLED:
        from hmm_gaussian_numba import GaussianHMM as CustomHMM
        model_label = "HMM"
    else:
        model_label = "GMM"

    for k in range(2, MAX_K + 1):
        if _HMM_ENABLED:
            gmm = CustomHMM(
                n_components=k, covariance_type='full',
                n_init=10, random_state=42, n_iter=300, reg_covar=1e-6
            )
        else:
            gmm = GaussianMixture(
                n_components=k, covariance_type='full',
                n_init=10, random_state=42, max_iter=300
            )
        gmm.fit(X)
        bic = gmm.bic(X)
        bic_values[k] = bic
        print(f"      k={k}: BIC={bic:.1f}")
        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_gmm = gmm

    print(f"   ✅ Seleccionado k={best_k} (BIC={best_bic:.1f}) [{model_label}]")

    # Predecir labels
    pred = best_gmm.predict(X)

    # Centroides en escala original
    centroids_scaled = best_gmm.means_
    centroids_original = scaler.inverse_transform(centroids_scaled)

    # Cluster sizes
    cluster_sizes = [int(np.sum(pred == k)) for k in range(best_k)]

    # Interpretar clusters (features: Hurst, Z_ATR, ER)
    cluster_names = []
    for k_idx in range(best_k):
        c = centroids_original[k_idx]
        h = c[0]       # Hurst
        z_atr = c[1]   # Z_ATR
        er = c[2]      # Efficiency Ratio
        parts = []
        # Hurst interpretation
        if h > 0.55:
            parts.append("trending")
        elif h < 0.45:
            parts.append("mean-rev")
        else:
            parts.append("neutral")
        # Z_ATR interpretation
        if z_atr > 0.5:
            parts.append("high-vol")
        elif z_atr < -0.5:
            parts.append("low-vol")
        else:
            parts.append("norm-vol")
        # ER interpretation
        if er > 0.3:
            parts.append("clean")
        elif er < 0.15:
            parts.append("choppy")
        cluster_names.append("/".join(parts))

    # Discriminación por feature
    discrimination = compute_discrimination(valid_features, pred, best_k)

    # Mapa temporal por año
    valid_indices = np.where(valid_mask)[0]
    temporal_map = compute_temporal_map(df, valid_indices, pred, best_k)

    # Date range
    date_range = (str(df['timestamp'].iloc[0]), str(df['timestamp'].iloc[-1]))

    model_data = {
        'gmm': best_gmm,
        'scaler': scaler,
        'n_clusters': best_k,
        'feature_names': list(FEATURE_NAMES),
        'lookback': LOOKBACK,
        'centroids': centroids_original,
        'cluster_names': cluster_names,
        'cluster_sizes': cluster_sizes,
        'training_bars': n_bars,
        'training_date_range': date_range,
        'bic_values': bic_values,
        'discrimination': discrimination,
        'temporal_map': temporal_map,
    }

    return model_data


def compute_discrimination(features, labels, n_clusters):
    """Calcula discriminación por feature: distancia inter-cluster / desviación intra-cluster.

    Para cada feature y par de clusters:
      ratio = |centroid_a - centroid_b| / mean(std_intra_a, std_intra_b)

    Si ratio < 0.5 en TODOS los pares, la feature no discrimina.

    Returns: dict con resultados por feature.
    """
    n_features = features.shape[1]
    results = {}

    for f_idx in range(n_features):
        fname = FEATURE_NAMES[f_idx]
        pair_ratios = []

        for a in range(n_clusters):
            mask_a = labels == a
            vals_a = features[mask_a, f_idx]
            std_a = np.std(vals_a) if len(vals_a) > 1 else 1e-10

            for b in range(a + 1, n_clusters):
                mask_b = labels == b
                vals_b = features[mask_b, f_idx]
                std_b = np.std(vals_b) if len(vals_b) > 1 else 1e-10

                centroid_dist = abs(np.mean(vals_a) - np.mean(vals_b))
                avg_std = (std_a + std_b) / 2
                if avg_std < 1e-10:
                    ratio = 0.0
                else:
                    ratio = centroid_dist / avg_std

                pair_ratios.append({
                    'pair': (a, b),
                    'ratio': ratio,
                    'centroid_dist': centroid_dist,
                    'avg_intra_std': avg_std,
                })

        discriminates = any(pr['ratio'] >= 0.5 for pr in pair_ratios)
        max_ratio = max(pr['ratio'] for pr in pair_ratios) if pair_ratios else 0.0

        results[fname] = {
            'pair_ratios': pair_ratios,
            'discriminates': discriminates,
            'max_ratio': max_ratio,
        }

    return results


def compute_temporal_map(df, valid_indices, labels, n_clusters):
    """Para cada año del histórico, porcentaje de barras en cada cluster.

    Returns: dict {year: {cluster_id: pct}}
    """
    timestamps = df['timestamp'].iloc[valid_indices]
    years = timestamps.dt.year.values

    temporal = {}
    for year in sorted(set(years)):
        year_mask = years == year
        year_labels = labels[year_mask]
        n_year = len(year_labels)
        if n_year == 0:
            continue
        temporal[int(year)] = {
            k: round(np.sum(year_labels == k) / n_year * 100, 1)
            for k in range(n_clusters)
        }

    return temporal


def generate_report(all_models, report_path):
    """Genera regime_report.txt con análisis de todos los modelos."""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*100}\n")
        f.write(f"REGIME MODEL REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"{'='*100}\n")
        f.write(f"Lookback: {LOOKBACK} | Max K: {MAX_K} | n_init: 10\n")
        f.write(f"Features: {', '.join(FEATURE_NAMES)}\n")
        f.write(f"Símbolos entrenados: {len(all_models)}\n\n")

        for sym_key, model_data in sorted(all_models.items()):
            f.write(f"\n{'='*80}\n")
            f.write(f"{sym_key}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Training bars: {model_data['training_bars']}\n")
            f.write(f"Date range: {model_data['training_date_range'][0]} → {model_data['training_date_range'][1]}\n")
            f.write(f"Selected k: {model_data['n_clusters']}\n")

            # BIC values
            f.write(f"\nBIC values:\n")
            for k, bic in sorted(model_data['bic_values'].items()):
                marker = " ← selected" if k == model_data['n_clusters'] else ""
                f.write(f"  k={k}: {bic:.1f}{marker}\n")

            # Centroids
            f.write(f"\nCentroides:\n")
            fnames = model_data['feature_names']
            header_parts = [f"{'Cluster':<25s}", f"{'Bars':>8s}", f"{'%':>6s}"]
            for fn in fnames:
                header_parts.append(f"{fn:>10s}")
            f.write(f"  {'  '.join(header_parts)}\n")
            f.write(f"  {'-'*80}\n")
            total_bars = sum(model_data['cluster_sizes'])
            for k in range(model_data['n_clusters']):
                name = model_data['cluster_names'][k]
                size = model_data['cluster_sizes'][k]
                pct = size / total_bars * 100 if total_bars > 0 else 0
                c = model_data['centroids'][k]
                vals = '  '.join(f"{c[j]:>10.4f}" for j in range(len(fnames)))
                f.write(f"  C{k} ({name:<19s}) {size:>8d} {pct:>5.1f}%  {vals}\n")

            # Discrimination analysis
            disc = model_data['discrimination']
            f.write(f"\nDiscriminación por feature:\n")
            for fname in FEATURE_NAMES:
                fd = disc[fname]
                status = "✅ DISCRIMINA" if fd['discriminates'] else "❌ NO discrimina"
                f.write(f"  {fname:<15s}: max_ratio={fd['max_ratio']:.3f}  {status}\n")
                for pr in fd['pair_ratios']:
                    a, b = pr['pair']
                    f.write(f"    C{a}-C{b}: ratio={pr['ratio']:.3f}"
                            f" (dist={pr['centroid_dist']:.5f}, intra_std={pr['avg_intra_std']:.5f})\n")

            non_disc = [fn for fn in FEATURE_NAMES if not disc[fn]['discriminates']]
            if non_disc:
                f.write(f"\n  ⚠️  Features sin discriminación (ratio < 0.5 en todos los pares): "
                        f"{', '.join(non_disc)}\n")
                f.write(f"     Considerar reemplazar estas features.\n")

            # Temporal map
            temporal = model_data['temporal_map']
            if temporal:
                f.write(f"\nDistribución temporal por año:\n")
                header_parts = [f"{'Año':>6s}"]
                for k in range(model_data['n_clusters']):
                    header_parts.append(f"C{k:>5d}%")
                f.write(f"  {'  '.join(header_parts)}\n")
                f.write(f"  {'-'*50}\n")
                for year in sorted(temporal.keys()):
                    parts = [f"{year:>6d}"]
                    for k in range(model_data['n_clusters']):
                        pct = temporal[year].get(k, 0)
                        parts.append(f"{pct:>6.1f}%")
                    f.write(f"  {'  '.join(parts)}\n")

            f.write("\n")

        # Summary section
        f.write(f"\n{'='*100}\n")
        f.write(f"RESUMEN DE DISCRIMINACIÓN\n")
        f.write(f"{'='*100}\n\n")

        for fname in FEATURE_NAMES:
            disc_count = sum(
                1 for md in all_models.values()
                if md['discrimination'][fname]['discriminates']
            )
            total = len(all_models)
            f.write(f"  {fname:<15s}: discrimina en {disc_count}/{total} símbolos"
                    f" ({disc_count/total*100:.0f}%)\n")

        f.write(f"\n{'='*100}\n")

    print(f"\n📄 Reporte generado: {report_path}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 70)
    print("🏋️ TRAIN REGIME MODELS — Pre-entrenamiento sobre histórico completo")
    print(f"   Features: {', '.join(FEATURE_NAMES)}")
    print(f"   Lookback: {LOOKBACK} | Max K: {MAX_K} | n_init: 10")
    print(f"   Output: {MODEL_DIR}/")
    print("=" * 70)

    # Warm up sklearn/OpenBLAS (Windows)
    GaussianMixture(n_components=2, n_init=1, max_iter=1, random_state=42).fit(
        np.random.randn(20, 3))

    all_models = {}
    failed = []

    for i, symbol in enumerate(SYMBOLS, 1):
        sym_key = _symbol_key(symbol)
        print(f"\n[{i}/{len(SYMBOLS)}] {symbol}")

        try:
            df = load_full_history(symbol)
            if df is None:
                failed.append(symbol)
                continue

            model_data = train_symbol_model(symbol, df)
            if model_data is None:
                failed.append(symbol)
                continue

            # Guardar modelo
            model_path = os.path.join(MODEL_DIR, f"{sym_key}_regime.joblib")
            joblib.dump(model_data, model_path)
            print(f"   💾 {model_path}")

            all_models[sym_key] = model_data

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            failed.append(symbol)

    # Generar reporte
    if all_models:
        report_path = os.path.join(MODEL_DIR, "regime_report.txt")
        generate_report(all_models, report_path)

    print(f"\n{'='*70}")
    print(f"✅ Completado: {len(all_models)}/{len(SYMBOLS)} modelos entrenados")
    if failed:
        print(f"❌ Fallidos: {', '.join(failed)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
