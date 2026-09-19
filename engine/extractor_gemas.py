#!/usr/bin/env python3
"""
extractor_gemas.py — Extrae gemas de los CSVs del lab completo v8.3

Uso:
    python extractor_gemas.py SYMBOL [--folder CARPETA] [--top N]

Ejemplos:
    python extractor_gemas.py BNBUSDT
    python extractor_gemas.py ETHUSDT --folder resultados_lab
    python extractor_gemas.py BTCUSDT --top 20

Si no se especifica --folder, busca en el directorio actual.
Si no se especifica --top, usa 15 por criterio.

Busca archivos:  full_SYMBOL_v??_H??.csv  y  ranking_SYMBOL_v??_H??.txt

Genera:
    gemas_SYMBOL.csv         — CSV consolidado con todas las gemas
    gemas_SYMBOL_resumen.txt — Resumen legible con familias y estadísticas
"""

import os
import sys
import re
import glob
import argparse
from collections import defaultdict

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────
# 1. PARSEO DE ARGUMENTOS
# ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Extractor de gemas del lab completo v8.3')
    parser.add_argument('symbol', help='Símbolo (ej: BNBUSDT, ETHUSDT)')
    parser.add_argument('--folder', default='.', help='Carpeta con CSVs y rankings (default: actual)')
    parser.add_argument('--top', type=int, default=15, help='Top N por criterio (default: 15)')
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────
# 2. CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────

def find_files(folder, symbol):
    """Encuentra CSVs full y rankings para el símbolo dado."""
    # Nuevo patrón con cluster_id (_C-1, _C0, _C1, _C2)
    csv_pattern = os.path.join(folder, f'full_{symbol}_v*_H*_C*.csv')
    rank_pattern = os.path.join(folder, f'ranking_{symbol}_v*_H*_C*.txt')

    csvs = sorted(glob.glob(csv_pattern))
    ranks = sorted(glob.glob(rank_pattern))

    # Fallback: patrón legacy sin _C*
    if not csvs:
        csv_pattern_legacy = os.path.join(folder, f'full_{symbol}_v*_H*.csv')
        rank_pattern_legacy = os.path.join(folder, f'ranking_{symbol}_v*_H*.txt')
        csvs = sorted(glob.glob(csv_pattern_legacy))
        ranks = sorted(glob.glob(rank_pattern_legacy))

    if not csvs:
        print(f"❌ No se encontraron CSVs: {csv_pattern}")
        sys.exit(1)

    return csvs, ranks


def extract_preset_from_filename(filepath):
    """Extrae 'v01_H00' o 'v01_H00_C-1' del nombre del archivo."""
    bn = os.path.basename(filepath)
    # Intentar con cluster_id primero
    m = re.search(r'(v\d+_H\d+_C-?\d+)', bn)
    if m:
        return m.group(1)
    # Fallback: sin cluster_id
    m = re.search(r'(v\d+_H\d+)', bn)
    return m.group(1) if m else 'unknown'


def extract_cluster_from_filename(filepath):
    """Extrae cluster_id del nombre. _C-1 = global, _C0/_C1/_C2 = cluster."""
    m = re.search(r'_C(-?\d+)', os.path.basename(filepath))
    return int(m.group(1)) if m else -1


def parse_ranking_header(filepath):
    """Extrae info de MAs del header del ranking.
    Formato: RANKING SYMBOL — FAST(len)/SLOW(len)/TREND_HYS - Lab...
    Retorna: 'KAMA(14)/T3(30) H00'
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if len(lines) >= 2:
            header = lines[1].strip()
            # Extraer "KAMA(14)/T3(30)/T3_H00"
            m = re.search(r'—\s*(.+?)\s*-\s*Lab', header)
            if m:
                ma_str = m.group(1).strip()
                # "KAMA(14)/T3(30)/T3_H00" → "KAMA(14)/T3(30) H00"
                parts = ma_str.split('/')
                if len(parts) >= 3:
                    fast = parts[0]
                    slow = parts[1]
                    trend_hys = parts[2]  # "T3_H00" or "ALMA_H05"
                    hys = 'H00' if 'H00' in trend_hys else 'H05'
                    return f"{fast}/{slow} {hys}"
            return ma_str
    except:
        pass
    return 'unknown'


def load_all_presets(csvs, ranks):
    """Carga todos los CSVs y les añade columnas de preset."""
    
    # Mapear preset → MA info desde rankings
    preset_ma = {}
    for rfile in ranks:
        preset = extract_preset_from_filename(rfile)
        ma_info = parse_ranking_header(rfile)
        preset_ma[preset] = ma_info
    
    all_dfs = []
    for csv_file in csvs:
        preset = extract_preset_from_filename(csv_file)
        # Para buscar ma_info, usar la parte sin _C* para compatibilidad con rankings
        preset_base = re.sub(r'_C-?\d+$', '', preset)
        ma_info = preset_ma.get(preset, preset_ma.get(preset_base, preset))

        df = pd.read_csv(csv_file)
        df['preset'] = preset
        df['ma_info'] = ma_info

        # cluster_id: from CSV column (if exists) or from filename
        if 'cluster_id' not in df.columns:
            df['cluster_id'] = extract_cluster_from_filename(csv_file)

        all_dfs.append(df)

        cid = df['cluster_id'].iloc[0] if len(df) > 0 else -1
        print(f"  📂 {os.path.basename(csv_file)}: {len(df)} configs — {ma_info} (C{cid})")

    combined = pd.concat(all_dfs, ignore_index=True)
    n_global = (combined['cluster_id'] == -1).sum()
    n_cluster = (combined['cluster_id'] >= 0).sum()
    print(f"\n  📊 Total: {len(combined)} filas en {len(all_dfs)} presets"
          f" (global: {n_global}, cluster: {n_cluster})")
    return combined


# ─────────────────────────────────────────────────────────────────
# 3. FILTROS Y CRITERIOS DE SELECCIÓN
# ─────────────────────────────────────────────────────────────────

# ─── 3b. FILTRO DE MESETA (ROBUSTEZ PARAMÉTRICA) ────────────────
# Estructura de bits del config_id (26 bits)
# Ref: lab_historico_numba_v8_3.py decode_config() / generate_valid_configs()
_PARAM_FIELDS = [
    #  (nombre,           shift, bits, is_bitmask)
    ('exit_mask',           0,    4,   True),
    ('entry_mask',          4,    5,   True),
    ('div_entry_mode',      9,    2,   False),
    ('div_exit',           11,    1,   False),
    ('div_type',           12,    2,   False),
    ('div_ind_mask',       14,    8,   True),
    ('cancel_tf',          22,    1,   False),
    ('use_ts',             23,    1,   False),
    ('reg_inv',            24,    1,   False),
    ('hid_inv',            25,    1,   False),
]


def _decode_config_id(config_id):
    """Descompone config_id en sus parámetros individuales."""
    params = {}
    for name, shift, n_bits, _ in _PARAM_FIELDS:
        params[name] = (config_id >> shift) & ((1 << n_bits) - 1)
    return params


def _encode_config_id(params):
    """Reconstruye config_id a partir de parámetros individuales."""
    cid = 0
    for name, shift, n_bits, _ in _PARAM_FIELDS:
        cid |= (params[name] & ((1 << n_bits) - 1)) << shift
    return cid


def _get_neighbors(config_id):
    """Genera vecinos paramétricos: configs que difieren en 1 parámetro por ±1 paso.

    Para campos bitmask (entry/exit/indicators): flip de cada bit individual.
    Para campos discretos/booleanos: ±1 dentro del rango válido.
    """
    params = _decode_config_id(config_id)
    neighbors = set()

    for name, shift, n_bits, is_bitmask in _PARAM_FIELDS:
        value = params[name]

        if is_bitmask:
            # Cada bit es un sub-parámetro independiente → flip individual
            for bit in range(n_bits):
                new_params = params.copy()
                new_params[name] = value ^ (1 << bit)
                neighbors.add(_encode_config_id(new_params))
        else:
            # Discreto/booleano → ±1 dentro del rango
            max_val = (1 << n_bits) - 1
            if value > 0:
                new_params = params.copy()
                new_params[name] = value - 1
                neighbors.add(_encode_config_id(new_params))
            if value < max_val:
                new_params = params.copy()
                new_params[name] = value + 1
                neighbors.add(_encode_config_id(new_params))

    neighbors.discard(config_id)
    return neighbors


def _evaluate_plateau(config_id, pf_lookup, extra_lookup=None,
                      min_neighbor_pf=1.2):
    """Evalúa si una config está en meseta de rentabilidad.

    CLAVE: El denominador del ratio es n_total (TODOS los vecinos generados),
    no solo los encontrados en el lookup. Los full_*.csv solo contienen el
    top ~22K de ~4M configs por preset. Un vecino ausente del lookup no pasó
    el corte de training → no es rentable → cuenta contra el ratio.

    Args:
        config_id: config candidata
        pf_lookup: dict {config_id: pf_fu} — top 22K del preset
        extra_lookup: dict {config_id: (pnl, maxdd, trades)} opcional
        min_neighbor_pf: PF mínimo para considerar vecino "rentable"

    Returns:
        dict con métricas de meseta
    """
    neighbors = _get_neighbors(config_id)
    n_total = len(neighbors)

    pf_values = []
    pnl_values = []
    dd_values = []
    trades_values = []
    n_profitable = 0
    n_pnl_positive = 0

    for nid in neighbors:
        if nid not in pf_lookup:
            # Vecino no está en top 22K → falló training → no rentable
            continue
        pf = pf_lookup[nid]
        pf_values.append(pf)
        if pf >= min_neighbor_pf:
            n_profitable += 1

        if extra_lookup and nid in extra_lookup:
            pnl, dd, trades = extra_lookup[nid]
            pnl_values.append(pnl)
            dd_values.append(dd)
            trades_values.append(trades)
            if pnl > 0:
                n_pnl_positive += 1

    n_found = len(pf_values)

    if n_total == 0:
        return {
            'plateau_ratio': 0.0,
            'plateau_pf_mean': 0.0,
            'plateau_pf_std': 0.0,
            'plateau_score': 0.0,
            'plateau_n_total': 0,
            'plateau_n_found': 0,
            'plateau_n_profitable': 0,
            'plateau_coverage': 0.0,
            'plateau_dd_consistency': 0.0,
            'plateau_trades_stability': 0.0,
            'plateau_direction_unanimity': 0.0,
        }

    # ratio sobre TODOS los vecinos (ausentes = no rentables)
    ratio = n_profitable / n_total
    coverage = n_found / n_total

    pf_mean = float(np.mean(pf_values)) if pf_values else 0.0
    pf_std = float(np.std(pf_values)) if pf_values else 0.0

    plateau_score = ratio * (1 - pf_std / pf_mean) if pf_mean > 0 else 0.0

    # Métricas adicionales (si hay datos)
    dd_consistency = 0.0
    trades_stability = 0.0
    direction_unanimity = 0.0

    if dd_values:
        dd_arr = np.array(dd_values)
        dd_mean = dd_arr.mean()
        dd_consistency = max(0.0, 1.0 - dd_arr.std() / dd_mean) if dd_mean > 0 else 0.0

    if trades_values:
        tr_arr = np.array(trades_values)
        tr_mean = tr_arr.mean()
        trades_stability = max(0.0, 1.0 - tr_arr.std() / tr_mean) if tr_mean > 0 else 0.0

    if pnl_values:
        direction_unanimity = n_pnl_positive / len(pnl_values)

    return {
        'plateau_ratio': ratio,
        'plateau_pf_mean': pf_mean,
        'plateau_pf_std': pf_std,
        'plateau_score': plateau_score,
        'plateau_n_total': n_total,
        'plateau_n_found': n_found,
        'plateau_n_profitable': n_profitable,
        'plateau_coverage': coverage,
        'plateau_dd_consistency': dd_consistency,
        'plateau_trades_stability': trades_stability,
        'plateau_direction_unanimity': direction_unanimity,
    }


def apply_plateau_filter(gems_best, combined, min_neighbors_ratio=0.6,
                         min_neighbor_pf=1.2):
    """Aplica filtro de meseta: descarta configs en picos aislados.

    Para cada gema, verifica que una fracción suficiente de sus vecinos
    paramétricos (configs con 1 parámetro diferente por ±1 paso) también
    son rentables dentro del mismo preset.
    """
    gem_presets = gems_best['preset'].unique()
    has_extra = all(c in combined.columns for c in ['pnl_ann_fu', 'maxdd_fu', 'trades_fu'])

    # Construir lookups solo para los presets usados por las gemas
    preset_pf = {}
    preset_extra = {}

    print(f"  Construyendo lookup de vecinos para {len(gem_presets)} presets...")
    for preset in gem_presets:
        pdf = combined[combined['preset'] == preset]
        ids = pdf['config_id'].astype(int).values
        preset_pf[preset] = dict(zip(ids, pdf['pf_fu'].values))

        if has_extra:
            preset_extra[preset] = dict(zip(
                ids,
                zip(pdf['pnl_ann_fu'].values, pdf['maxdd_fu'].values, pdf['trades_fu'].values)
            ))

        print(f"      {preset}: {len(ids)} configs en lookup")

    # Evaluar meseta para cada gema
    plateau_results = []
    for _, row in gems_best.iterrows():
        cid = int(row['config_id'])
        preset = row['preset']
        pf_lk = preset_pf.get(preset, {})
        ex_lk = preset_extra.get(preset) if has_extra else None

        result = _evaluate_plateau(cid, pf_lk, ex_lk, min_neighbor_pf)
        plateau_results.append(result)

    # Añadir columnas de meseta al DataFrame
    plateau_df = pd.DataFrame(plateau_results)
    gems_enriched = pd.concat(
        [gems_best.reset_index(drop=True), plateau_df.reset_index(drop=True)],
        axis=1
    )

    # Distribución antes de filtrar
    ratios = gems_enriched['plateau_ratio']
    coverage = gems_enriched['plateau_coverage']
    print(f"  Distribución plateau_ratio:    "
          f"min={ratios.min():.2f}, median={ratios.median():.2f}, max={ratios.max():.2f}")
    print(f"  Distribución plateau_coverage: "
          f"min={coverage.min():.2f}, median={coverage.median():.2f}, max={coverage.max():.2f}")
    print(f"  Vecinos: {gems_enriched['plateau_n_total'].iloc[0]} generados, "
          f"media {gems_enriched['plateau_n_found'].mean():.1f} encontrados en lookup, "
          f"media {gems_enriched['plateau_n_profitable'].mean():.1f} rentables (PF>={min_neighbor_pf})")

    before = len(gems_enriched)
    gems_filtered = gems_enriched[gems_enriched['plateau_ratio'] >= min_neighbors_ratio].copy()
    after = len(gems_filtered)
    print(f"  Filtro de meseta (ratio>={min_neighbors_ratio}): {before} -> {after} gemas "
          f"({before - after} descartadas como picos)")

    # Ajustar structural_score con plateau_ratio
    gems_filtered['structural_score'] = (
        gems_filtered['structural_score'] * gems_filtered['plateau_ratio']
    )
    gems_filtered = gems_filtered.sort_values(
        'structural_score', ascending=False
    ).reset_index(drop=True)

    return gems_filtered

# ─────────────────────────────────────────────────────────────────

def compute_structural_score(df):
    """Calcula structural_score basado en PF y robustness.

    structural_score = pf × sqrt(robustness)
    (plateau_ratio se multiplica después en apply_plateau_filter)

    Para especialistas de cluster (cluster_id >= 0), usa pf_cl_fu si disponible.
    """
    # Use cluster PF for specialists, global PF for generalists
    has_cl = 'pf_cl_fu' in df.columns and 'cluster_id' in df.columns
    if has_cl:
        pf = np.where(df['cluster_id'] >= 0, df['pf_cl_fu'], df['pf_fu'])
    else:
        pf = df['pf_fu'].values

    robustness = df['robustness'].values if 'robustness' in df.columns else np.ones(len(df))

    df['structural_score'] = pf * np.sqrt(np.maximum(robustness, 0.01))
    return df


def apply_validation_filter(df):
    """Filtro base: pf_te >= 1.2 AND trades_te >= 8.
    For cluster specialists (cluster_id >= 0), uses cluster-specific metrics
    (pf_cl_te, trades_cl_te) if available."""
    has_cl = 'pf_cl_te' in df.columns and 'trades_cl_te' in df.columns
    has_clusters = 'cluster_id' in df.columns and (df['cluster_id'] >= 0).any()

    if has_cl and has_clusters:
        # Global presets: use global metrics
        global_mask = df['cluster_id'] < 0
        valid_global = df[global_mask & (df['pf_te'] >= 1.2) & (df['trades_te'] >= 8)]
        # Cluster presets: use cluster-specific metrics
        cluster_mask = df['cluster_id'] >= 0
        valid_cluster = df[cluster_mask & (df['pf_cl_te'] >= 1.2) & (df['trades_cl_te'] >= 8)]
        valid = pd.concat([valid_global, valid_cluster], ignore_index=True)
        print(f"  ✅ Validadas: {len(valid)} de {len(df)} ({100*len(valid)/len(df):.1f}%)"
              f" (global: {len(valid_global)}, cluster: {len(valid_cluster)})")
    else:
        valid = df[(df['pf_te'] >= 1.2) & (df['trades_te'] >= 8)].copy()
        print(f"  ✅ Validadas: {len(valid)} de {len(df)} ({100*len(valid)/len(df):.1f}%)")
    return valid


def extract_gems(valid, top_n=15):
    """Aplica los 7 criterios y retorna set de config_ids seleccionados."""
    
    gem_ids = set()
    gem_criteria = defaultdict(list)  # config_id → [criterios que lo seleccionaron]
    
    # Auxiliares con filtros adicionales
    has_80 = valid[valid['trades_fu'] >= 80]
    has_80_calmar = has_80[has_80['maxdd_fu'] > 1].copy()
    if len(has_80_calmar) > 0:
        has_80_calmar['calmar'] = has_80_calmar['pnl_ann_fu'] / has_80_calmar['maxdd_fu']
    has_80_pf14 = has_80[has_80['pf_fu'] > 1.4]
    high_freq = valid[(valid['trades_fu'] > 250) & (valid['pf_fu'] > 1.4)]
    
    criteria = [
        ('structural_score', valid, 'structural_score', top_n),
        ('pf_fu', has_80, 'pf_fu', top_n),
        ('pnl_ann_fu', has_80, 'pnl_ann_fu', top_n),
        ('calmar', has_80_calmar, 'calmar', top_n),
        ('robustness', has_80_pf14, 'robustness', top_n),
        ('high_freq', high_freq, 'pnl_ann_fu', top_n),
        ('pf_tr', has_80, 'pf_tr', top_n),
    ]
    
    for name, pool, sort_col, n in criteria:
        if len(pool) == 0:
            print(f"  ⚠️  Criterio '{name}': pool vacío, saltando")
            continue
        top = pool.nlargest(n, sort_col)
        selected = set(top['config_id'].unique())
        for cid in selected:
            gem_criteria[cid].append(name)
        gem_ids.update(selected)
        print(f"  🔹 {name}: {len(selected)} configs seleccionadas")
    
    # Criterio 7: Cross-preset (≥3 presets)
    config_presets = defaultdict(set)
    presets = valid['preset'].unique()
    for preset in presets:
        preset_df = valid[valid['preset'] == preset]
        top200 = preset_df.nlargest(200, 'structural_score')
        for cid in top200['config_id'].unique():
            config_presets[cid].add(preset)
    
    cross = {cid for cid, ps in config_presets.items() if len(ps) >= 3}
    for cid in cross:
        gem_criteria[cid].append(f'cross_{len(config_presets[cid])}p')
    gem_ids.update(cross)
    print(f"  🔹 cross-preset (≥3): {len(cross)} configs seleccionadas")
    
    print(f"\n  💎 Total gemas únicas: {len(gem_ids)}")
    return gem_ids, gem_criteria, config_presets


# ─────────────────────────────────────────────────────────────────
# 4. DEDUPLICACIÓN Y ENRIQUECIMIENTO
# ─────────────────────────────────────────────────────────────────

def build_gem_table(valid, gem_ids, gem_criteria, config_presets):
    """Para cada gema, toma el preset con mejor structural_score."""

    gems = valid[valid['config_id'].isin(gem_ids)].copy()

    # Mejor preset por structural_score
    gems_best = gems.sort_values('structural_score', ascending=False).drop_duplicates('config_id').copy()
    
    # Añadir columnas extra
    gems_best['criterios'] = gems_best['config_id'].map(
        lambda cid: '|'.join(gem_criteria.get(cid, []))
    )
    gems_best['n_criterios'] = gems_best['config_id'].map(
        lambda cid: len(gem_criteria.get(cid, []))
    )
    gems_best['n_presets'] = gems_best['config_id'].map(
        lambda cid: len(config_presets.get(cid, set()))
    )
    gems_best['presets_list'] = gems_best['config_id'].map(
        lambda cid: '|'.join(sorted(config_presets.get(cid, set())))
    )
    
    # Calmar
    gems_best['calmar'] = np.where(
        gems_best['maxdd_fu'] > 0,
        gems_best['pnl_ann_fu'] / gems_best['maxdd_fu'],
        0
    )
    
    # Repintabilidad: TF4 = no repintable, TF1/TF2/TF3 = repintable
    def check_repaint(entry_tfs):
        s = str(entry_tfs)
        tfs = set(s.replace(' ', '').split('+')) if '+' in s else {s.strip()}
        return 'NO' if tfs == {'TF4'} else 'SI'
    
    gems_best['repintable'] = gems_best['entry_tfs'].apply(check_repaint)
    
    return gems_best.sort_values('structural_score', ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
# 5. AGRUPACIÓN EN FAMILIAS
# ─────────────────────────────────────────────────────────────────

def assign_families(gems_best):
    """Agrupa gemas en familias por similitud de configuración.
    Criterios: preset/MA, entry_tfs, div_type, indicators, exit_tfs, sl_type
    Configs que comparten ≥5 de 6 atributos van en la misma familia.
    """
    
    def get_signature(row):
        return (
            str(row.get('ma_info', '')),
            str(row.get('entry_tfs', '')),
            str(row.get('div_type', '')),
            str(row.get('indicators', '')),
            str(row.get('exit_tfs', '')),
            str(row.get('sl_type', '')),
        )
    
    def similarity(sig1, sig2):
        return sum(a == b for a, b in zip(sig1, sig2))
    
    signatures = [get_signature(row) for _, row in gems_best.iterrows()]
    n = len(signatures)
    family_ids = [-1] * n
    current_family = 0
    
    for i in range(n):
        if family_ids[i] >= 0:
            continue
        family_ids[i] = current_family
        for j in range(i + 1, n):
            if family_ids[j] >= 0:
                continue
            if similarity(signatures[i], signatures[j]) >= 5:
                family_ids[j] = current_family
        current_family += 1
    
    # Convertir a letras: 0→A, 1→B, etc.
    family_labels = []
    for fid in family_ids:
        if fid < 26:
            family_labels.append(chr(65 + fid))
        else:
            family_labels.append(f'Z{fid - 25}')
    
    gems_best['familia'] = family_labels
    
    # Reordenar familias por tamaño (más grande = A)
    family_sizes = gems_best['familia'].value_counts()
    size_rank = {fam: i for i, fam in enumerate(family_sizes.index)}
    new_labels = {old: chr(65 + rank) if rank < 26 else f'Z{rank-25}' 
                  for old, rank in size_rank.items()}
    gems_best['familia'] = gems_best['familia'].map(new_labels)
    
    return gems_best


# ─────────────────────────────────────────────────────────────────
# 6. GENERACIÓN DE RESUMEN
# ─────────────────────────────────────────────────────────────────

def generate_summary(gems_best, symbol, output_folder):
    """Genera resumen legible en texto."""

    # Guard: si no hay gemas, escribir resumen vacío y retornar
    if len(gems_best) == 0:
        lines = []
        lines.append(f"{'='*80}")
        lines.append(f"GEMAS {symbol} — 0 gemas encontradas")
        lines.append(f"{'='*80}\n")
        lines.append("No se encontraron gemas que cumplan los criterios de filtrado.")
        lines.append(f"\n{'='*80}")
        summary_text = '\n'.join(lines)
        summary_file = os.path.join(output_folder, f'gemas_{symbol}_resumen.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        print(f"\n{summary_text}")
        return summary_file

    lines = []
    lines.append(f"{'='*80}")
    lines.append(f"GEMAS {symbol} — {len(gems_best)} gemas en {gems_best['familia'].nunique()} familias")
    lines.append(f"{'='*80}\n")
    
    # Distribución por preset
    lines.append("DISTRIBUCIÓN POR PRESET:")
    preset_counts = gems_best['ma_info'].value_counts()
    for preset, count in preset_counts.items():
        pct = 100 * count / len(gems_best)
        lines.append(f"  {preset:<30s}: {count:>3d} gemas ({pct:.1f}%)")
    
    # Distribución H00 vs H05
    h00 = gems_best[gems_best['preset'].str.contains('H00')]
    h05 = gems_best[gems_best['preset'].str.contains('H05')]
    lines.append(f"\n  H00: {len(h00)} ({100*len(h00)/len(gems_best):.0f}%) | H05: {len(h05)} ({100*len(h05)/len(gems_best):.0f}%)")
    
    # Repintabilidad
    no_rep = gems_best[gems_best['repintable'] == 'NO']
    si_rep = gems_best[gems_best['repintable'] == 'SI']
    lines.append(f"  No repintable: {len(no_rep)} ({100*len(no_rep)/len(gems_best):.0f}%) | Repintable: {len(si_rep)} ({100*len(si_rep)/len(gems_best):.0f}%)")
    
    # Resumen por familia
    lines.append(f"\n{'─'*80}")
    lines.append("FAMILIAS:")
    lines.append(f"{'─'*80}\n")
    
    for fam in sorted(gems_best['familia'].unique()):
        fam_df = gems_best[gems_best['familia'] == fam]
        n = len(fam_df)
        
        # Representante: mejor structural_score
        rep = fam_df.iloc[0]
        
        lines.append(f"Familia {fam} — {n} gemas")
        lines.append(f"  Preset:     {rep['ma_info']}")
        lines.append(f"  Entry:      {rep['entry_tfs']} | Exit: {rep['exit_tfs']} | SL: {rep['sl_type']}")
        lines.append(f"  Div:        Entry={rep['div_entry']}, Type={rep['div_type']}, Variant={rep.get('variant','')}")
        lines.append(f"  Indicadores: {rep['indicators']}")
        lines.append(f"  Repintable: {rep['repintable']}")
        lines.append(f"  PF rango:   {fam_df['pf_fu'].min():.2f} — {fam_df['pf_fu'].max():.2f}")
        lines.append(f"  PnL/año:    {fam_df['pnl_ann_fu'].min():.1f}% — {fam_df['pnl_ann_fu'].max():.1f}%")
        lines.append(f"  MaxDD:      {fam_df['maxdd_fu'].min():.1f}% — {fam_df['maxdd_fu'].max():.1f}%")
        lines.append(f"  Trades:     {fam_df['trades_fu'].min():.0f} — {fam_df['trades_fu'].max():.0f}")
        lines.append(f"  Calmar:     {fam_df['calmar'].min():.1f} — {fam_df['calmar'].max():.1f}")
        lines.append(f"  Robustness: {fam_df['robustness'].min():.2f} — {fam_df['robustness'].max():.2f}")
        if 'plateau_ratio' in fam_df.columns:
            lines.append(f"  Plateau:    ratio={fam_df['plateau_ratio'].mean():.2f}, "
                         f"PF_mean={fam_df['plateau_pf_mean'].mean():.2f}, "
                         f"score={fam_df['plateau_score'].mean():.2f}")
        lines.append("")
    
    # Estadísticas de meseta
    if 'plateau_ratio' in gems_best.columns:
        lines.append(f"{'─'*80}")
        lines.append("ROBUSTEZ PARAMÉTRICA (MESETA):")
        lines.append(f"{'─'*80}")
        pr = gems_best['plateau_ratio']
        ps = gems_best['plateau_score']
        lines.append(f"  plateau_ratio:  min={pr.min():.2f}  median={pr.median():.2f}  "
                     f"mean={pr.mean():.2f}  max={pr.max():.2f}")
        lines.append(f"  plateau_score:  min={ps.min():.2f}  median={ps.median():.2f}  "
                     f"mean={ps.mean():.2f}  max={ps.max():.2f}")
        if 'plateau_dd_consistency' in gems_best.columns:
            ddc = gems_best['plateau_dd_consistency']
            ts = gems_best['plateau_trades_stability']
            du = gems_best['plateau_direction_unanimity']
            lines.append(f"  dd_consistency:  mean={ddc.mean():.2f}  |  "
                         f"trades_stability: mean={ts.mean():.2f}  |  "
                         f"direction_unanimity: mean={du.mean():.2f}")
        # Top 5 por plateau_score
        lines.append(f"\n  Top 5 por plateau_score:")
        lines.append(f"  {'ConfigID':>12s}  {'Fam':>3s}  {'PF':>5s}  {'PlRatio':>7s}  {'PlScore':>7s}  {'Total':>5s}  {'Found':>5s}  {'Rent':>4s}")
        for _, row in gems_best.nlargest(5, 'plateau_score').iterrows():
            lines.append(f"  {int(row['config_id']):>12d}  {row['familia']:>3s}  "
                         f"{row['pf_fu']:>5.2f}  {row['plateau_ratio']:>7.2f}  "
                         f"{row['plateau_score']:>7.3f}  "
                         f"{int(row['plateau_n_total']):>5d}  "
                         f"{int(row['plateau_n_found']):>5d}  "
                         f"{int(row['plateau_n_profitable']):>4d}")
        lines.append("")

    # Top 10 por PF
    lines.append(f"{'─'*80}")
    lines.append("TOP 10 POR PF (full):")
    lines.append(f"{'─'*80}")
    lines.append(f"  {'ConfigID':>12s}  {'Fam':>3s}  {'Preset':<25s}  {'PF':>5s}  {'PnL/y':>7s}  {'DD':>6s}  {'Trd':>4s}  {'Cal':>5s}  {'Rep':>3s}")
    for _, row in gems_best.nlargest(10, 'pf_fu').iterrows():
        lines.append(f"  {int(row['config_id']):>12d}  {row['familia']:>3s}  {row['ma_info']:<25s}  {row['pf_fu']:>5.2f}  {row['pnl_ann_fu']:>6.1f}%  {row['maxdd_fu']:>5.1f}%  {int(row['trades_fu']):>4d}  {row['calmar']:>5.1f}  {row['repintable']:>3s}")
    
    # Top 10 por Calmar
    lines.append(f"\n{'─'*80}")
    lines.append("TOP 10 POR CALMAR (PnL/DD):")
    lines.append(f"{'─'*80}")
    lines.append(f"  {'ConfigID':>12s}  {'Fam':>3s}  {'Preset':<25s}  {'PF':>5s}  {'PnL/y':>7s}  {'DD':>6s}  {'Trd':>4s}  {'Cal':>5s}  {'Rep':>3s}")
    for _, row in gems_best.nlargest(10, 'calmar').iterrows():
        lines.append(f"  {int(row['config_id']):>12d}  {row['familia']:>3s}  {row['ma_info']:<25s}  {row['pf_fu']:>5.2f}  {row['pnl_ann_fu']:>6.1f}%  {row['maxdd_fu']:>5.1f}%  {int(row['trades_fu']):>4d}  {row['calmar']:>5.1f}  {row['repintable']:>3s}")
    
    # Top 10 no repintables por PF
    no_rep_df = gems_best[gems_best['repintable'] == 'NO']
    if len(no_rep_df) > 0:
        lines.append(f"\n{'─'*80}")
        lines.append("TOP 10 NO REPINTABLES POR PF:")
        lines.append(f"{'─'*80}")
        lines.append(f"  {'ConfigID':>12s}  {'Fam':>3s}  {'Preset':<25s}  {'PF':>5s}  {'PnL/y':>7s}  {'DD':>6s}  {'Trd':>4s}  {'Cal':>5s}  {'Entry':<10s}")
        for _, row in no_rep_df.nlargest(10, 'pf_fu').iterrows():
            lines.append(f"  {int(row['config_id']):>12d}  {row['familia']:>3s}  {row['ma_info']:<25s}  {row['pf_fu']:>5.2f}  {row['pnl_ann_fu']:>6.1f}%  {row['maxdd_fu']:>5.1f}%  {int(row['trades_fu']):>4d}  {row['calmar']:>5.1f}  {row['entry_tfs']:<10s}")
    
    # Candidatas finales: PF > 2.0, DD < 15%, no repintables
    candidates = gems_best[(gems_best['pf_fu'] > 2.0) & (gems_best['maxdd_fu'] < 15)]
    candidates_nr = candidates[candidates['repintable'] == 'NO']
    
    lines.append(f"\n{'='*80}")
    lines.append(f"CANDIDATAS PARA VALIDADOR:")
    lines.append(f"  PF > 2.0 AND DD < 15%: {len(candidates)} gemas")
    lines.append(f"  + No repintables: {len(candidates_nr)} gemas")
    
    if len(candidates_nr) > 0:
        lines.append(f"\n  {'ConfigID':>12s}  {'Fam':>3s}  {'Preset':<25s}  {'PF':>5s}  {'PnL/y':>7s}  {'DD':>6s}  {'Trd':>4s}  {'Cal':>5s}")
        for _, row in candidates_nr.sort_values('pf_fu', ascending=False).iterrows():
            lines.append(f"  {int(row['config_id']):>12d}  {row['familia']:>3s}  {row['ma_info']:<25s}  {row['pf_fu']:>5.2f}  {row['pnl_ann_fu']:>6.1f}%  {row['maxdd_fu']:>5.1f}%  {int(row['trades_fu']):>4d}  {row['calmar']:>5.1f}")
    
    if len(candidates) > len(candidates_nr) and len(candidates_nr) < 5:
        lines.append(f"\n  (También repintables con PF > 2.0, DD < 15%:)")
        candidates_rep = candidates[candidates['repintable'] == 'SI']
        for _, row in candidates_rep.sort_values('pf_fu', ascending=False).head(10).iterrows():
            lines.append(f"  {int(row['config_id']):>12d}  {row['familia']:>3s}  {row['ma_info']:<25s}  {row['pf_fu']:>5.2f}  {row['pnl_ann_fu']:>6.1f}%  {row['maxdd_fu']:>5.1f}%  {int(row['trades_fu']):>4d}  {row['calmar']:>5.1f}  {row['entry_tfs']}")
    
    # Sección por cluster (si hay presets con cluster_id)
    if 'cluster_id' in gems_best.columns:
        cluster_ids_present = sorted(gems_best['cluster_id'].unique())
        has_clusters = any(cid >= 0 for cid in cluster_ids_present)
        if has_clusters:
            lines.append(f"\n{'='*80}")
            lines.append(f"GEMAS POR CLUSTER:")
            lines.append(f"{'='*80}")
            for cid in cluster_ids_present:
                cid_label = "GLOBAL" if cid == -1 else f"C{cid}"
                cid_df = gems_best[gems_best['cluster_id'] == cid]
                if len(cid_df) == 0:
                    continue
                n_fam = cid_df['familia'].nunique()
                lines.append(f"\n  {cid_label}: {len(cid_df)} gemas en {n_fam} familias")
                fams = sorted(cid_df['familia'].unique())
                for fam in fams[:5]:
                    fam_df = cid_df[cid_df['familia'] == fam]
                    rep = fam_df.iloc[0]
                    lines.append(f"    Fam {fam} ({len(fam_df)}): {rep['ma_info']}"
                                 f" PF={fam_df['pf_fu'].median():.2f}"
                                 f" PnL/y={fam_df['pnl_ann_fu'].median():.1f}%"
                                 f" DD={fam_df['maxdd_fu'].median():.1f}%")

    lines.append(f"\n{'='*80}")

    summary_text = '\n'.join(lines)

    summary_file = os.path.join(output_folder, f'gemas_{symbol}_resumen.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"\n{summary_text}")
    return summary_file


# ─────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    symbol = args.symbol.upper()
    folder = args.folder
    top_n = args.top
    
    print(f"\n{'='*60}")
    print(f"💎 EXTRACTOR DE GEMAS — {symbol}")
    print(f"{'='*60}\n")
    
    # 1. Encontrar archivos
    print("📁 Buscando archivos...")
    csvs, ranks = find_files(folder, symbol)
    
    # 2. Cargar datos
    print(f"\n📥 Cargando {len(csvs)} presets...")
    combined = load_all_presets(csvs, ranks)
    
    # 3. Filtro de validación
    print(f"\n🔍 Aplicando filtro de validación (pf_te≥1.2, trades_te≥8)...")
    valid = apply_validation_filter(combined)

    # 3b. Calcular structural_score
    valid = compute_structural_score(valid)

    # 4. Extraer gemas por criterios
    print(f"\n💎 Extrayendo gemas (top {top_n} por criterio)...")
    gem_ids, gem_criteria, config_presets = extract_gems(valid, top_n)
    
    # 5. Construir tabla deduplicada
    print(f"\n📊 Construyendo tabla de gemas...")
    gems_best = build_gem_table(valid, gem_ids, gem_criteria, config_presets)

    # 5b. Filtro de meseta (robustez paramétrica)
    print(f"\n🏔️  Aplicando filtro de meseta (robustez paramétrica)...")
    gems_best = apply_plateau_filter(gems_best, combined)

    # 6. Asignar familias
    print(f"\n👪 Agrupando en familias...")
    gems_best = assign_families(gems_best)
    
    # 7. Guardar CSV
    output_csv = os.path.join(folder, f'gemas_{symbol}.csv')
    gems_best.to_csv(output_csv, index=False)
    print(f"\n💾 CSV guardado: {output_csv}")
    
    # 8. Generar resumen
    print(f"\n📝 Generando resumen...")
    summary_file = generate_summary(gems_best, symbol, folder)
    print(f"💾 Resumen guardado: {summary_file}")
    
    print(f"\n✅ Extracción completada: {len(gems_best)} gemas en {gems_best['familia'].nunique()} familias\n")


if __name__ == '__main__':
    main()
