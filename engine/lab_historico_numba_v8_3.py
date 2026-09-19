# ============================================
# 🧪 LABORATORIO HISTÓRICO v8.3 UNIFICADO - NUMBA
# ============================================
# BASE: v8.2 con medias ultracortas (Lab LITE v5c)
#
# v8.3 CAMBIO: Período corto con split 50/50
#   TOTAL_CANDLES = 10000 (was 20000)
#   TRAIN_RATIO = 0.50 (was 0.75) → 5k train / 5k test
#   MIN_TRADES reducidos para período más corto (15/15)
#   Justificación: curva de degradación BNB muestra PF>1.8 en 5k,
#   colapso a PF<1 en 30k+. Optimizar sobre régimen reciente
#   con reciclaje periódico superior a optimizar sobre período largo.
#
# UNIFICA LAS 4 VARIANTES DE DIVERGENCIA EN UN SOLO LAB:
#   bit 24 = reg_inv (0=Tradicional, 1=Invertida)
#   bit 25 = hid_inv (0=Tradicional, 1=Invertida)
#
#   reg_inv=0, hid_inv=0 → FIX      (Reg trad + Hid trad)
#   reg_inv=0, hid_inv=1 → ORIGINAL (Reg trad + Hid inv)
#   reg_inv=1, hid_inv=1 → ALLINV   (Reg inv  + Hid inv)
#   reg_inv=1, hid_inv=0 → REGINV   (Reg inv  + Hid trad)
#
# Generación inteligente: solo genera combos relevantes por div_type
#   div_type=NONE    → ×1 (bits inv irrelevantes)
#   div_type=REGULAR → ×2 (solo reg_inv importa)
#   div_type=HIDDEN  → ×2 (solo hid_inv importa)
#   div_type=BOTH    → ×4 (ambos importan)
#
# Versión acelerada con Numba, FIDELIDAD ABSOLUTA al indicador Pine v10.0
# 
# v7.1 FIX: Div delay Pine-faithful (mantenido)
#   En Pine, div_raw se calcula en isconfirmed pero las condiciones de entrada
#   se evalúan fuera de isconfirmed con div_raw de la barra anterior.
#   Entradas usan div de barra N-1, salidas usan div de barra N.
#
# Estructura de config: 26 bits (24 base + 2 inversión)
# Comisiones: round-trip integrado (default 0.10%)
# ============================================

import sys
# v2.4.0: reconfigure stdout UTF-8 para prints con emojis (cp1252 compat en Windows).
# Preserva emojis en output en Linux/VPS. Alcance: módulo-wide (también al ser importado).
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import time
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import os
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN
# ============================================

SYMBOLS = [
    "BNB/USDT",
    "BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT", "TRX/USDT", "DOGE/USDT", 
    "ADA/USDT", "BCH/USDT", "LINK/USDT", "XLM/USDT", "SUI/USDT", "LTC/USDT", 
    "AVAX/USDT", "HBAR/USDT", "SHIB/USDT", "DOT/USDT", "UNI/USDT",
    "TAO/USDT", "AAVE/USDT", "NEAR/USDT", "ICP/USDT", "ETC/USDT", 
    "ONDO/USDT", "ENA/USDT", "POL/USDT", "WLD/USDT", "APT/USDT", "ATOM/USDT", 
    "KAS/USDT", "ALGO/USDT", "RENDER/USDT", "ARB/USDT", "FIL/USDT", "QNT/USDT", 
    "VET/USDT", "FLR/USDT", "SEI/USDT", "OP/USDT", "IMX/USDT", 
    "INJ/USDT", "FET/USDT", "STX/USDT", "SAND/USDT", "MANA/USDT", "GRT/USDT", 
    "THETA/USDT"
]

TOTAL_CANDLES = 10000
TIMEFRAME = "1h"

# ============================================
# PRESETS DE ZONAS v11.0 (Lab LITE — 45 symbols operacionales, ~20.9M configs/preset
# verificados empíricamente 2026-04-29 vía generate_valid_configs(); config_id ocupa
# bits 0-25 = 26 bits efectivos. Ver §0.0 CONTEXTO_PROYECTO_TRADING.md para el marco
# foundational del universo de candidatas. SYMBOLS list a 47 elementos arriba contiene
# residuos lab pendientes cleanup post-reciclaje; SYMBOL_ZONE_PRESETS abajo a 45
# alineado con master.py CONFIG['symbols'] operacional.)
# v8.3: Optimizadas sobre 5k recientes, reciclaje periódico
# ============================================
# Formato: lista de tuplas por símbolo
#   (fast_type, fast_len, fast_p1, fast_p2,
#    slow_type, slow_len, slow_p1, slow_p2,
#    trend_type, trend_len, trend_p1, trend_p2)
# Top 5 por score + agresivo (mejor PnL) si fuera del top 5
SYMBOL_ZONE_PRESETS = {
    "AAVE/USDT": [
        ("VIDYA",20,0.0,0.0, "VWMA",60,0.0,0.0, "VWMA",240,0.0,0.0),  # #1 VIDYA(20)/VWMA(60)/ft5:VWMA(240)
        ("VIDYA",16,0.0,0.0, "VWMA",54,0.0,0.0, "VWMA",216,0.0,0.0),  # #2 VIDYA(16)/VWMA(54)/ft5:VWMA(216)
        ("McGinley",12,0.0,0.0, "DEMA",72,0.0,0.0, "DEMA",288,0.0,0.0),  # #3 McGinley(12)/DEMA(72)/ft5:DEMA(288)
        ("VIDYA",20,0.0,0.0, "VWMA",54,0.0,0.0, "VWMA",216,0.0,0.0),  # #4 VIDYA(20)/VWMA(54)/ft5:VWMA(216)
        ("ALMA",22,0.5,4.0, "SSmoother",51,0.0,0.0, "SSmoother",204,0.0,0.0),  # #5 ALMA(22,0.5,4)/SSmoother(51)/ft5:SSmoother(204)
    ],
    "ADA/USDT": [
        ("McGinley",24,0.0,0.0, "KAMA",51,0.0,0.0, "KAMA",204,0.0,0.0),  # #1 McGinley(24)/KAMA(51)/ft5:KAMA(204)
        ("McGinley",22,0.0,0.0, "KAMA",51,0.0,0.0, "KAMA",204,0.0,0.0),  # #2 McGinley(22)/KAMA(51)/ft5:KAMA(204)
        ("VIDYA",12,0.0,0.0, "VWMA",69,0.0,0.0, "VWMA",276,0.0,0.0),  # #3 VIDYA(12)/VWMA(69)/ft5:VWMA(276)
        ("McGinley",22,0.0,0.0, "KAMA",49,0.0,0.0, "KAMA",196,0.0,0.0),  # #4 McGinley(22)/KAMA(49)/ft5:KAMA(196)
        ("McGinley",24,0.0,0.0, "KAMA",49,0.0,0.0, "KAMA",196,0.0,0.0),  # #5 McGinley(24)/KAMA(49)/ft5:KAMA(196)
    ],
    "ALGO/USDT": [
        ("VIDYA",24,0.0,0.0, "VWMA",57,0.0,0.0, "VWMA",228,0.0,0.0),  # #1 VIDYA(24)/VWMA(57)/ft5:VWMA(228)
        ("McGinley",24,0.0,0.0, "KAMA",51,0.0,0.0, "KAMA",204,0.0,0.0),  # #2 McGinley(24)/KAMA(51)/ft5:KAMA(204)
        ("VIDYA",24,0.0,0.0, "VWMA",54,0.0,0.0, "VWMA",216,0.0,0.0),  # #3 VIDYA(24)/VWMA(54)/ft5:VWMA(216)
        ("McGinley",16,0.0,0.0, "VWMA",30,0.0,0.0, "VWMA",120,0.0,0.0),  # #4 McGinley(16)/VWMA(30)/ft5:VWMA(120)
        ("VIDYA",22,0.0,0.0, "VWMA",51,0.0,0.0, "VWMA",204,0.0,0.0),  # #5 VIDYA(22)/VWMA(51)/ft5:VWMA(204)
    ],
    "APT/USDT": [
        ("Tenkan",24,0.0,0.0, "KAMA",72,0.0,0.0, "KAMA",288,0.0,0.0),  # #1 Tenkan(24)/KAMA(72)/ft5:KAMA(288)
        ("KAMA",24,0.0,0.0, "KAMA",72,0.0,0.0, "KAMA",288,0.0,0.0),  # #2 KAMA(24)/KAMA(72)/ft5:KAMA(288)
        ("Tenkan",22,0.0,0.0, "KAMA",72,0.0,0.0, "KAMA",288,0.0,0.0),  # #3 Tenkan(22)/KAMA(72)/ft5:KAMA(288)
        ("Tenkan",24,0.0,0.0, "KAMA",66,0.0,0.0, "KAMA",264,0.0,0.0),  # #4 Tenkan(24)/KAMA(66)/ft5:KAMA(264)
        ("Tenkan",22,0.0,0.0, "KAMA",66,0.0,0.0, "KAMA",264,0.0,0.0),  # #5 Tenkan(22)/KAMA(66)/ft5:KAMA(264)
    ],
    "ARB/USDT": [
        ("McGinley",24,0.0,0.0, "ALMA",33,0.5,4.0, "ALMA",132,0.5,4.0),  # #1 McGinley(24)/ALMA(33,0.5,4)/ft5:ALMA(132)
        ("McGinley",8,0.0,0.0, "TEMA",75,0.0,0.0, "TEMA",300,0.0,0.0),  # #2 McGinley(8)/TEMA(75)/ft5:TEMA(300)
        ("McGinley",8,0.0,0.0, "TEMA",75,0.0,0.0, "VIDYA",250,0.0,0.0),  # #3 McGinley(8)/TEMA(75)/VIDYA(250)
        ("McGinley",8,0.0,0.0, "TEMA",75,0.0,0.0, "VIDYA",240,0.0,0.0),  # #4 McGinley(8)/TEMA(75)/VIDYA(240)
        ("McGinley",8,0.0,0.0, "TEMA",75,0.0,0.0, "VIDYA",230,0.0,0.0),  # #5 McGinley(8)/TEMA(75)/VIDYA(230)
    ],
    "ATOM/USDT": [
        ("McGinley",20,0.0,0.0, "KAMA",36,0.0,0.0, "KAMA",144,0.0,0.0),  # #1 McGinley(20)/KAMA(36)/ft5:KAMA(144)
        ("KAMA",16,0.0,0.0, "SSmoother",63,0.0,0.0, "SSmoother",252,0.0,0.0),  # #2 KAMA(16)/SSmoother(63)/ft5:SSmoother(252)
        ("McGinley",20,0.0,0.0, "KAMA",36,0.0,0.0, "Tenkan",190,0.0,0.0),  # #3 McGinley(20)/KAMA(36)/Tenkan(190)
        ("McGinley",10,0.0,0.0, "ZLEMA",36,0.0,0.0, "ZLEMA",144,0.0,0.0),  # #4 McGinley(10)/ZLEMA(36)/ft5:ZLEMA(144)
        ("McGinley",20,0.0,0.0, "ALMA",36,0.85,6.0, "ALMA",144,0.85,6.0),  # #5 McGinley(20)/ALMA(36,0.85,6)/ft5:ALMA(144)
    ],
    "AVAX/USDT": [
        ("VIDYA",12,0.0,0.0, "T3",33,0.8,0.0, "T3",132,0.8,0.0),  # #1 VIDYA(12)/T3(33,v0.8)/ft5:T3(132)
        ("VIDYA",14,0.0,0.0, "ALMA",57,0.5,6.0, "ALMA",228,0.5,6.0),  # #2 VIDYA(14)/ALMA(57,0.5,6)/ft5:ALMA(228)
        ("KAMA",18,0.0,0.0, "DEMA",42,0.0,0.0, "DEMA",168,0.0,0.0),  # #3 KAMA(18)/DEMA(42)/ft5:DEMA(168)
        ("VIDYA",14,0.0,0.0, "T3",33,0.7,0.0, "T3",132,0.7,0.0),  # #4 VIDYA(14)/T3(33,v0.7)/ft5:T3(132)
        ("KAMA",14,0.0,0.0, "T3",36,0.8,0.0, "T3",144,0.8,0.0),  # #5 KAMA(14)/T3(36,v0.8)/ft5:T3(144)
    ],
    "BCH/USDT": [
        ("VIDYA",10,0.0,0.0, "DEMA",72,0.0,0.0, "DEMA",288,0.0,0.0),  # #1 VIDYA(10)/DEMA(72)/ft5:DEMA(288)
        ("VIDYA",10,0.0,0.0, "DEMA",75,0.0,0.0, "DEMA",300,0.0,0.0),  # #2 VIDYA(10)/DEMA(75)/ft5:DEMA(300)
        ("McGinley",24,0.0,0.0, "ZLEMA",51,0.0,0.0, "ZLEMA",204,0.0,0.0),  # #3 McGinley(24)/ZLEMA(51)/ft5:ZLEMA(204)
        ("VIDYA",22,0.0,0.0, "ZLEMA",48,0.0,0.0, "ZLEMA",192,0.0,0.0),  # #4 VIDYA(22)/ZLEMA(48)/ft5:ZLEMA(192)
        ("VIDYA",22,0.0,0.0, "ZLEMA",49,0.0,0.0, "ZLEMA",196,0.0,0.0),  # #5 VIDYA(22)/ZLEMA(49)/ft5:ZLEMA(196)
    ],
    "BNB/USDT": [
        ("KAMA",14,0.0,0.0, "T3",30,0.8,0.0, "T3",120,0.8,0.0),  # #1 KAMA(14)/T3(30,v0.8)/ft5:T3(120)
        ("KAMA",14,0.0,0.0, "ALMA",39,0.5,4.0, "ALMA",156,0.5,4.0),  # #2 KAMA(14)/ALMA(39,0.5,4)/ft5:ALMA(156)
        ("McGinley",14,0.0,0.0, "SMA",33,0.0,0.0, "SMA",132,0.0,0.0),  # #3 McGinley(14)/SMA(33)/ft5:SMA(132)
        ("KAMA",14,0.0,0.0, "ALMA",42,0.5,6.0, "ALMA",168,0.5,6.0),  # #4 KAMA(14)/ALMA(42,0.5,6)/ft5:ALMA(168)
        ("KAMA",16,0.0,0.0, "T3",30,0.8,0.0, "T3",120,0.8,0.0),  # #5 KAMA(16)/T3(30,v0.8)/ft5:T3(120)
    ],
    "BTC/USDT": [
        ("McGinley",20,0.0,0.0, "EMA",42,0.0,0.0, "EMA",168,0.0,0.0),  # #1 McGinley(20)/EMA(42)/ft5:EMA(168)
        ("SMA",22,0.0,0.0, "EMA",39,0.0,0.0, "EMA",156,0.0,0.0),  # #2 SMA(22)/EMA(39)/ft5:EMA(156)
        ("T3",20,0.8,0.0, "SMA",33,0.0,0.0, "SMA",132,0.0,0.0),  # #3 T3(20,v0.8)/SMA(33)/ft5:SMA(132)
        ("T3",18,0.7,0.0, "ALMA",66,0.85,4.0, "ALMA",264,0.85,4.0),  # #4 T3(18,v0.7)/ALMA(66,0.85,4)/ft5:ALMA(264)
        ("T3",20,0.8,0.0, "SMA",36,0.0,0.0, "SMA",144,0.0,0.0),  # #5 T3(20,v0.8)/SMA(36)/ft5:SMA(144)
    ],
    "DOGE/USDT": [
        ("KAMA",14,0.0,0.0, "TEMA",63,0.0,0.0, "TEMA",252,0.0,0.0),  # #1 KAMA(14)/TEMA(63)/ft5:TEMA(252)
        ("KAMA",14,0.0,0.0, "HMA",49,0.0,0.0, "HMA",196,0.0,0.0),  # #2 KAMA(14)/HMA(49)/ft5:HMA(196)
        ("KAMA",12,0.0,0.0, "TEMA",66,0.0,0.0, "TEMA",264,0.0,0.0),  # #3 KAMA(12)/TEMA(66)/ft5:TEMA(264)
        ("KAMA",16,0.0,0.0, "TEMA",63,0.0,0.0, "TEMA",252,0.0,0.0),  # #4 KAMA(16)/TEMA(63)/ft5:TEMA(252)
        ("KAMA",16,0.0,0.0, "HMA",57,0.0,0.0, "HMA",228,0.0,0.0),  # #5 KAMA(16)/HMA(57)/ft5:HMA(228)
    ],
    "DOT/USDT": [
        ("VIDYA",14,0.0,0.0, "KAMA",33,0.0,0.0, "KAMA",132,0.0,0.0),  # #1 VIDYA(14)/KAMA(33)/ft5:KAMA(132)
        ("VIDYA",14,0.0,0.0, "KAMA",36,0.0,0.0, "KAMA",144,0.0,0.0),  # #2 VIDYA(14)/KAMA(36)/ft5:KAMA(144)
        ("McGinley",24,0.0,0.0, "KAMA",49,0.0,0.0, "KAMA",196,0.0,0.0),  # #3 McGinley(24)/KAMA(49)/ft5:KAMA(196)
        ("VIDYA",18,0.0,0.0, "KAMA",36,0.0,0.0, "KAMA",144,0.0,0.0),  # #4 VIDYA(18)/KAMA(36)/ft5:KAMA(144)
        ("McGinley",24,0.0,0.0, "KAMA",51,0.0,0.0, "KAMA",204,0.0,0.0),  # #5 McGinley(24)/KAMA(51)/ft5:KAMA(204)
    ],
    "ENA/USDT": [
        ("McGinley",24,0.0,0.0, "VWMA",75,0.0,0.0, "VWMA",300,0.0,0.0),  # #1 McGinley(24)/VWMA(75)/ft5:VWMA(300)
        ("EMA",22,0.0,0.0, "KAMA",45,0.0,0.0, "KAMA",180,0.0,0.0),  # #2 EMA(22)/KAMA(45)/ft5:KAMA(180)
        ("McGinley",18,0.0,0.0, "VWMA",48,0.0,0.0, "VWMA",192,0.0,0.0),  # #3 McGinley(18)/VWMA(48)/ft5:VWMA(192)
        ("Tenkan",24,0.0,0.0, "KAMA",54,0.0,0.0, "KAMA",216,0.0,0.0),  # #4 Tenkan(24)/KAMA(54)/ft5:KAMA(216)
        ("Tenkan",20,0.0,0.0, "KAMA",72,0.0,0.0, "KAMA",288,0.0,0.0),  # #5 Tenkan(20)/KAMA(72)/ft5:KAMA(288)
    ],
    "ETC/USDT": [
        ("VIDYA",24,0.0,0.0, "KAMA",49,0.0,0.0, "KAMA",196,0.0,0.0),  # #1 VIDYA(24)/KAMA(49)/ft5:KAMA(196)
        ("KAMA",14,0.0,0.0, "TEMA",54,0.0,0.0, "TEMA",216,0.0,0.0),  # #2 KAMA(14)/TEMA(54)/ft5:TEMA(216)
        ("KAMA",12,0.0,0.0, "TEMA",54,0.0,0.0, "TEMA",216,0.0,0.0),  # #3 KAMA(12)/TEMA(54)/ft5:TEMA(216)
        ("VIDYA",24,0.0,0.0, "KAMA",48,0.0,0.0, "KAMA",192,0.0,0.0),  # #4 VIDYA(24)/KAMA(48)/ft5:KAMA(192)
        ("VIDYA",14,0.0,0.0, "SSmoother",60,0.0,0.0, "SSmoother",240,0.0,0.0),  # #5 VIDYA(14)/SSmoother(60)/ft5:SSmoother(240)
        ("VIDYA",14,0.0,0.0, "EMA",36,0.0,0.0, "EMA",144,0.0,0.0),  # #6 VIDYA(14)/EMA(36)/ft5:EMA(144)  # AGRESIVO
    ],
    "ETH/USDT": [
        ("McGinley",20,0.0,0.0, "KAMA",48,0.0,0.0, "KAMA",192,0.0,0.0),  # #1 McGinley(20)/KAMA(48)/ft5:KAMA(192)
        ("McGinley",16,0.0,0.0, "KAMA",49,0.0,0.0, "KAMA",196,0.0,0.0),  # #2 McGinley(16)/KAMA(49)/ft5:KAMA(196)
        ("Tenkan",22,0.0,0.0, "EMA",51,0.0,0.0, "EMA",204,0.0,0.0),  # #3 Tenkan(22)/EMA(51)/ft5:EMA(204)
        ("Tenkan",22,0.0,0.0, "SMA",57,0.0,0.0, "SMA",228,0.0,0.0),  # #4 Tenkan(22)/SMA(57)/ft5:SMA(228)
        ("Tenkan",22,0.0,0.0, "EMA",54,0.0,0.0, "EMA",216,0.0,0.0),  # #5 Tenkan(22)/EMA(54)/ft5:EMA(216)
        ("Tenkan",22,0.0,0.0, "EMA",45,0.0,0.0, "EMA",180,0.0,0.0),  # #6 Tenkan(22)/EMA(45)/ft5:EMA(180)  # AGRESIVO
    ],
    "FET/USDT": [
        ("ALMA",20,0.5,4.0, "T3",36,0.8,0.0, "T3",144,0.8,0.0),  # #1 ALMA(20,0.5,4)/T3(36,v0.8)/ft5:T3(144)
        ("ALMA",22,0.5,4.0, "T3",36,0.8,0.0, "T3",144,0.8,0.0),  # #2 ALMA(22,0.5,4)/T3(36,v0.8)/ft5:T3(144)
        ("McGinley",16,0.0,0.0, "T3",39,0.8,0.0, "T3",156,0.8,0.0),  # #3 McGinley(16)/T3(39,v0.8)/ft5:T3(156)
        ("McGinley",14,0.0,0.0, "T3",33,0.7,0.0, "T3",132,0.7,0.0),  # #4 McGinley(14)/T3(33,v0.7)/ft5:T3(132)
        ("McGinley",12,0.0,0.0, "ALMA",69,0.85,4.0, "ALMA",276,0.85,4.0),  # #5 McGinley(12)/ALMA(69,0.85,4)/ft5:ALMA(276)
    ],
    "FIL/USDT": [
        ("VIDYA",10,0.0,0.0, "DEMA",75,0.0,0.0, "DEMA",300,0.0,0.0),  # #1 VIDYA(10)/DEMA(75)/ft5:DEMA(300)
        ("VIDYA",12,0.0,0.0, "DEMA",75,0.0,0.0, "DEMA",300,0.0,0.0),  # #2 VIDYA(12)/DEMA(75)/ft5:DEMA(300)
        ("KAMA",14,0.0,0.0, "TEMA",69,0.0,0.0, "TEMA",276,0.0,0.0),  # #3 KAMA(14)/TEMA(69)/ft5:TEMA(276)
        ("KAMA",12,0.0,0.0, "DEMA",48,0.0,0.0, "DEMA",192,0.0,0.0),  # #4 KAMA(12)/DEMA(48)/ft5:DEMA(192)
        ("KAMA",14,0.0,0.0, "DEMA",49,0.0,0.0, "DEMA",196,0.0,0.0),  # #5 KAMA(14)/DEMA(49)/ft5:DEMA(196)
    ],
    "GRT/USDT": [
        ("VIDYA",24,0.0,0.0, "KAMA",39,0.0,0.0, "KAMA",156,0.0,0.0),  # #1 VIDYA(24)/KAMA(39)/ft5:KAMA(156)
        ("McGinley",10,0.0,0.0, "DEMA",39,0.0,0.0, "DEMA",156,0.0,0.0),  # #2 McGinley(10)/DEMA(39)/ft5:DEMA(156)
        ("VIDYA",24,0.0,0.0, "KAMA",36,0.0,0.0, "KAMA",144,0.0,0.0),  # #3 VIDYA(24)/KAMA(36)/ft5:KAMA(144)
        ("McGinley",10,0.0,0.0, "TEMA",60,0.0,0.0, "TEMA",240,0.0,0.0),  # #4 McGinley(10)/TEMA(60)/ft5:TEMA(240)
        ("EMA",12,0.0,0.0, "TEMA",60,0.0,0.0, "TEMA",240,0.0,0.0),  # #5 EMA(12)/TEMA(60)/ft5:TEMA(240)
        ("ALMA",18,0.5,6.0, "TEMA",63,0.0,0.0, "TEMA",252,0.0,0.0),  # #15 ALMA(18,0.5,6)/TEMA(63)/ft5:TEMA(252)  # AGRESIVO
    ],
    "HBAR/USDT": [
        ("McGinley",24,0.0,0.0, "KAMA",49,0.0,0.0, "KAMA",196,0.0,0.0),  # #1 McGinley(24)/KAMA(49)/ft5:KAMA(196)
        ("McGinley",18,0.0,0.0, "KAMA",42,0.0,0.0, "KAMA",168,0.0,0.0),  # #2 McGinley(18)/KAMA(42)/ft5:KAMA(168)
        ("McGinley",24,0.0,0.0, "KAMA",48,0.0,0.0, "KAMA",192,0.0,0.0),  # #3 McGinley(24)/KAMA(48)/ft5:KAMA(192)
        ("McGinley",22,0.0,0.0, "KAMA",45,0.0,0.0, "KAMA",180,0.0,0.0),  # #4 McGinley(22)/KAMA(45)/ft5:KAMA(180)
        ("VIDYA",16,0.0,0.0, "VWMA",54,0.0,0.0, "VWMA",216,0.0,0.0),  # #5 VIDYA(16)/VWMA(54)/ft5:VWMA(216)
    ],
    "ICP/USDT": [
        ("VIDYA",12,0.0,0.0, "KAMA",45,0.0,0.0, "KAMA",180,0.0,0.0),  # #1 VIDYA(12)/KAMA(45)/ft5:KAMA(180)
        ("VIDYA",14,0.0,0.0, "KAMA",48,0.0,0.0, "KAMA",192,0.0,0.0),  # #2 VIDYA(14)/KAMA(48)/ft5:KAMA(192)
        ("VIDYA",12,0.0,0.0, "KAMA",48,0.0,0.0, "KAMA",192,0.0,0.0),  # #3 VIDYA(12)/KAMA(48)/ft5:KAMA(192)
        ("VIDYA",10,0.0,0.0, "KAMA",42,0.0,0.0, "KAMA",168,0.0,0.0),  # #4 VIDYA(10)/KAMA(42)/ft5:KAMA(168)
        ("T3",18,0.8,0.0, "SMA",48,0.0,0.0, "SMA",192,0.0,0.0),  # #5 T3(18,v0.8)/SMA(48)/ft5:SMA(192)
    ],
    "IMX/USDT": [
        ("KAMA",14,0.0,0.0, "TEMA",66,0.0,0.0, "TEMA",264,0.0,0.0),  # #1 KAMA(14)/TEMA(66)/ft5:TEMA(264)
        ("SMA",22,0.0,0.0, "DEMA",39,0.0,0.0, "DEMA",156,0.0,0.0),  # #2 SMA(22)/DEMA(39)/ft5:DEMA(156)
        ("ALMA",24,0.5,4.0, "TEMA",54,0.0,0.0, "TEMA",216,0.0,0.0),  # #3 ALMA(24,0.5,4)/TEMA(54)/ft5:TEMA(216)
        ("KAMA",14,0.0,0.0, "TEMA",63,0.0,0.0, "TEMA",252,0.0,0.0),  # #4 KAMA(14)/TEMA(63)/ft5:TEMA(252)
        ("SMA",22,0.0,0.0, "JMA",72,0.0,0.0, "JMA",288,0.0,0.0),  # #5 SMA(22)/JMA(72)/ft5:JMA(288)
    ],
    "INJ/USDT": [
        ("McGinley",24,0.0,0.0, "KAMA",42,0.0,0.0, "KAMA",168,0.0,0.0),  # #1 McGinley(24)/KAMA(42)/ft5:KAMA(168)
        ("McGinley",24,0.0,0.0, "KAMA",49,0.0,0.0, "KAMA",196,0.0,0.0),  # #2 McGinley(24)/KAMA(49)/ft5:KAMA(196)
        ("SMA",14,0.0,0.0, "TEMA",60,0.0,0.0, "TEMA",240,0.0,0.0),  # #3 SMA(14)/TEMA(60)/ft5:TEMA(240)
        ("SMA",14,0.0,0.0, "TEMA",57,0.0,0.0, "TEMA",228,0.0,0.0),  # #4 SMA(14)/TEMA(57)/ft5:TEMA(228)
        ("McGinley",14,0.0,0.0, "KAMA",39,0.0,0.0, "KAMA",156,0.0,0.0),  # #5 McGinley(14)/KAMA(39)/ft5:KAMA(156)
        ("T3",12,0.8,0.0, "TEMA",49,0.0,0.0, "TEMA",196,0.0,0.0),  # #7 T3(12,v0.8)/TEMA(49)/ft5:TEMA(196)  # AGRESIVO
    ],
    "LINK/USDT": [
        ("McGinley",20,0.0,0.0, "ALMA",33,0.85,4.0, "ALMA",132,0.85,4.0),  # #1 McGinley(20)/ALMA(33,0.85,4)/ft5:ALMA(132)
        ("McGinley",18,0.0,0.0, "ALMA",30,0.85,6.0, "ALMA",120,0.85,6.0),  # #2 McGinley(18)/ALMA(30,0.85,6)/ft5:ALMA(120)
        ("McGinley",12,0.0,0.0, "HMA",30,0.0,0.0, "HMA",120,0.0,0.0),  # #3 McGinley(12)/HMA(30)/ft5:HMA(120)
        ("McGinley",20,0.0,0.0, "ALMA",48,0.85,6.0, "ALMA",192,0.85,6.0),  # #4 McGinley(20)/ALMA(48,0.85,6)/ft5:ALMA(192)
        ("McGinley",22,0.0,0.0, "SSmoother",33,0.0,0.0, "SSmoother",132,0.0,0.0),  # #5 McGinley(22)/SSmoother(33)/ft5:SSmoother(132)
        ("KAMA",14,0.0,0.0, "ALMA",36,0.85,6.0, "ALMA",144,0.85,6.0),  # #7 KAMA(14)/ALMA(36,0.85,6)/ft5:ALMA(144)  # AGRESIVO
    ],
    "LTC/USDT": [
        ("SMA",16,0.0,0.0, "TEMA",63,0.0,0.0, "TEMA",252,0.0,0.0),  # #1 SMA(16)/TEMA(63)/ft5:TEMA(252)
        ("EMA",14,0.0,0.0, "TEMA",69,0.0,0.0, "TEMA",276,0.0,0.0),  # #2 EMA(14)/TEMA(69)/ft5:TEMA(276)
        ("VWMA",24,0.0,0.0, "DEMA",45,0.0,0.0, "DEMA",180,0.0,0.0),  # #3 VWMA(24)/DEMA(45)/ft5:DEMA(180)
        ("KAMA",12,0.0,0.0, "DEMA",33,0.0,0.0, "DEMA",132,0.0,0.0),  # #4 KAMA(12)/DEMA(33)/ft5:DEMA(132)
        ("KAMA",12,0.0,0.0, "JMA",72,0.0,0.0, "JMA",288,0.0,0.0),  # #5 KAMA(12)/JMA(72)/ft5:JMA(288)
    ],
    "MANA/USDT": [
        ("EMA",14,0.0,0.0, "ZLEMA",30,0.0,0.0, "ZLEMA",120,0.0,0.0),  # #1 EMA(14)/ZLEMA(30)/ft5:ZLEMA(120)
        ("McGinley",22,0.0,0.0, "KAMA",45,0.0,0.0, "KAMA",180,0.0,0.0),  # #2 McGinley(22)/KAMA(45)/ft5:KAMA(180)
        ("McGinley",22,0.0,0.0, "KAMA",42,0.0,0.0, "KAMA",168,0.0,0.0),  # #3 McGinley(22)/KAMA(42)/ft5:KAMA(168)
        ("McGinley",24,0.0,0.0, "KAMA",42,0.0,0.0, "KAMA",168,0.0,0.0),  # #4 McGinley(24)/KAMA(42)/ft5:KAMA(168)
        ("McGinley",22,0.0,0.0, "KAMA",39,0.0,0.0, "KAMA",156,0.0,0.0),  # #5 McGinley(22)/KAMA(39)/ft5:KAMA(156)
    ],
    "NEAR/USDT": [
        ("KAMA",22,0.0,0.0, "ZLEMA",72,0.0,0.0, "ZLEMA",288,0.0,0.0),  # #1 KAMA(22)/ZLEMA(72)/ft5:ZLEMA(288)
        ("KAMA",14,0.0,0.0, "DEMA",45,0.0,0.0, "DEMA",180,0.0,0.0),  # #2 KAMA(14)/DEMA(45)/ft5:DEMA(180)
        ("KAMA",16,0.0,0.0, "DEMA",42,0.0,0.0, "DEMA",168,0.0,0.0),  # #3 KAMA(16)/DEMA(42)/ft5:DEMA(168)
        ("KAMA",22,0.0,0.0, "ZLEMA",69,0.0,0.0, "ZLEMA",276,0.0,0.0),  # #4 KAMA(22)/ZLEMA(69)/ft5:ZLEMA(276)
        ("KAMA",24,0.0,0.0, "HMA",60,0.0,0.0, "HMA",240,0.0,0.0),  # #5 KAMA(24)/HMA(60)/ft5:HMA(240)
    ],
    "ONDO/USDT": [
        ("T3",10,0.8,0.0, "EMA",36,0.0,0.0, "EMA",144,0.0,0.0),  # #1 T3(10,v0.8)/EMA(36)/ft5:EMA(144)
        ("VIDYA",24,0.0,0.0, "T3",33,0.7,0.0, "T3",132,0.7,0.0),  # #2 VIDYA(24)/T3(33,v0.7)/ft5:T3(132)
        ("T3",10,0.7,0.0, "EMA",36,0.0,0.0, "EMA",144,0.0,0.0),  # #3 T3(10,v0.7)/EMA(36)/ft5:EMA(144)
        ("T3",10,0.7,0.0, "EMA",33,0.0,0.0, "EMA",132,0.0,0.0),  # #4 T3(10,v0.7)/EMA(33)/ft5:EMA(132)
        ("VIDYA",18,0.0,0.0, "T3",33,0.8,0.0, "T3",132,0.8,0.0),  # #5 VIDYA(18)/T3(33,v0.8)/ft5:T3(132)
    ],
    "OP/USDT": [
        ("T3",20,0.8,0.0, "SMA",63,0.0,0.0, "SMA",252,0.0,0.0),  # #1 T3(20,v0.8)/SMA(63)/ft5:SMA(252)
        ("T3",20,0.8,0.0, "SMA",60,0.0,0.0, "SMA",240,0.0,0.0),  # #2 T3(20,v0.8)/SMA(60)/ft5:SMA(240)
        ("T3",20,0.7,0.0, "SMA",63,0.0,0.0, "SMA",252,0.0,0.0),  # #3 T3(20,v0.7)/SMA(63)/ft5:SMA(252)
        ("KAMA",22,0.0,0.0, "HMA",66,0.0,0.0, "VIDYA",210,0.0,0.0),  # #4 KAMA(22)/HMA(66)/VIDYA(210)
        ("KAMA",22,0.0,0.0, "HMA",66,0.0,0.0, "VIDYA",180,0.0,0.0),  # #5 KAMA(22)/HMA(66)/VIDYA(180)
    ],
    "POL/USDT": [
        ("T3",16,0.7,0.0, "VWMA",51,0.0,0.0, "VWMA",204,0.0,0.0),  # #1 T3(16,v0.7)/VWMA(51)/ft5:VWMA(204)
        ("T3",16,0.7,0.0, "KAMA",48,0.0,0.0, "KAMA",192,0.0,0.0),  # #2 T3(16,v0.7)/KAMA(48)/ft5:KAMA(192)
        ("T3",16,0.8,0.0, "KAMA",54,0.0,0.0, "KAMA",216,0.0,0.0),  # #3 T3(16,v0.8)/KAMA(54)/ft5:KAMA(216)
        ("T3",16,0.8,0.0, "KAMA",49,0.0,0.0, "KAMA",196,0.0,0.0),  # #4 T3(16,v0.8)/KAMA(49)/ft5:KAMA(196)
        ("McGinley",22,0.0,0.0, "KAMA",42,0.0,0.0, "KAMA",168,0.0,0.0),  # #5 McGinley(22)/KAMA(42)/ft5:KAMA(168)
    ],
    "QNT/USDT": [
        ("VWMA",12,0.0,0.0, "JMA",39,0.0,0.0, "JMA",156,0.0,0.0),  # #1 VWMA(12)/JMA(39)/ft5:JMA(156)
        ("VWMA",12,0.0,0.0, "JMA",36,0.0,0.0, "JMA",144,0.0,0.0),  # #2 VWMA(12)/JMA(36)/ft5:JMA(144)
        ("VWMA",12,0.0,0.0, "JMA",33,0.0,0.0, "JMA",132,0.0,0.0),  # #3 VWMA(12)/JMA(33)/ft5:JMA(132)
        ("VIDYA",24,0.0,0.0, "KAMA",36,0.0,0.0, "KAMA",144,0.0,0.0),  # #4 VIDYA(24)/KAMA(36)/ft5:KAMA(144)
        ("ALMA",18,0.5,6.0, "ZLEMA",45,0.0,0.0, "ZLEMA",180,0.0,0.0),  # #5 ALMA(18,0.5,6)/ZLEMA(45)/ft5:ZLEMA(180)
    ],
    "RENDER/USDT": [
        ("VIDYA",12,0.0,0.0, "VWMA",54,0.0,0.0, "VWMA",216,0.0,0.0),  # #1 VIDYA(12)/VWMA(54)/ft5:VWMA(216)
        ("VIDYA",20,0.0,0.0, "VWMA",66,0.0,0.0, "VWMA",264,0.0,0.0),  # #2 VIDYA(20)/VWMA(66)/ft5:VWMA(264)
        ("VIDYA",16,0.0,0.0, "VWMA",54,0.0,0.0, "VWMA",216,0.0,0.0),  # #3 VIDYA(16)/VWMA(54)/ft5:VWMA(216)
        ("VIDYA",20,0.0,0.0, "VWMA",72,0.0,0.0, "VWMA",288,0.0,0.0),  # #4 VIDYA(20)/VWMA(72)/ft5:VWMA(288)
        ("VIDYA",10,0.0,0.0, "VWMA",54,0.0,0.0, "VWMA",216,0.0,0.0),  # #5 VIDYA(10)/VWMA(54)/ft5:VWMA(216)
    ],
    "SAND/USDT": [
        ("KAMA",22,0.0,0.0, "KAMA",33,0.0,0.0, "KAMA",132,0.0,0.0),  # #1 KAMA(22)/KAMA(33)/ft5:KAMA(132)
        ("KAMA",24,0.0,0.0, "KAMA",33,0.0,0.0, "KAMA",132,0.0,0.0),  # #2 KAMA(24)/KAMA(33)/ft5:KAMA(132)
        ("KAMA",18,0.0,0.0, "KAMA",33,0.0,0.0, "KAMA",132,0.0,0.0),  # #3 KAMA(18)/KAMA(33)/ft5:KAMA(132)
        ("KAMA",20,0.0,0.0, "KAMA",33,0.0,0.0, "KAMA",132,0.0,0.0),  # #4 KAMA(20)/KAMA(33)/ft5:KAMA(132)
        ("KAMA",20,0.0,0.0, "KAMA",36,0.0,0.0, "KAMA",144,0.0,0.0),  # #5 KAMA(20)/KAMA(36)/ft5:KAMA(144)
    ],
    "SEI/USDT": [
        ("McGinley",20,0.0,0.0, "KAMA",45,0.0,0.0, "KAMA",180,0.0,0.0),  # #1 McGinley(20)/KAMA(45)/ft5:KAMA(180)
        ("McGinley",20,0.0,0.0, "KAMA",42,0.0,0.0, "KAMA",168,0.0,0.0),  # #2 McGinley(20)/KAMA(42)/ft5:KAMA(168)
        ("McGinley",22,0.0,0.0, "KAMA",48,0.0,0.0, "KAMA",192,0.0,0.0),  # #3 McGinley(22)/KAMA(48)/ft5:KAMA(192)
        ("McGinley",22,0.0,0.0, "KAMA",49,0.0,0.0, "KAMA",196,0.0,0.0),  # #4 McGinley(22)/KAMA(49)/ft5:KAMA(196)
        ("McGinley",24,0.0,0.0, "KAMA",51,0.0,0.0, "KAMA",204,0.0,0.0),  # #5 McGinley(24)/KAMA(51)/ft5:KAMA(204)
    ],
    "SHIB/USDT": [
        ("McGinley",24,0.0,0.0, "KAMA",49,0.0,0.0, "KAMA",196,0.0,0.0),  # #1 McGinley(24)/KAMA(49)/ft5:KAMA(196)
        ("McGinley",24,0.0,0.0, "KAMA",42,0.0,0.0, "KAMA",168,0.0,0.0),  # #2 McGinley(24)/KAMA(42)/ft5:KAMA(168)
        ("T3",16,0.7,0.0, "EMA",45,0.0,0.0, "EMA",180,0.0,0.0),  # #3 T3(16,v0.7)/EMA(45)/ft5:EMA(180)
        ("McGinley",22,0.0,0.0, "KAMA",42,0.0,0.0, "KAMA",168,0.0,0.0),  # #4 McGinley(22)/KAMA(42)/ft5:KAMA(168)
        ("T3",20,0.8,0.0, "KAMA",51,0.0,0.0, "KAMA",204,0.0,0.0),  # #5 T3(20,v0.8)/KAMA(51)/ft5:KAMA(204)
    ],
    "SOL/USDT": [
        ("KAMA",22,0.0,0.0, "HMA",69,0.0,0.0, "HMA",276,0.0,0.0),  # #1 KAMA(22)/HMA(69)/ft5:HMA(276)
        ("McGinley",18,0.0,0.0, "KAMA",48,0.0,0.0, "KAMA",192,0.0,0.0),  # #2 McGinley(18)/KAMA(48)/ft5:KAMA(192)
        ("KAMA",20,0.0,0.0, "HMA",69,0.0,0.0, "HMA",276,0.0,0.0),  # #3 KAMA(20)/HMA(69)/ft5:HMA(276)
        ("VIDYA",14,0.0,0.0, "EMA",49,0.0,0.0, "EMA",196,0.0,0.0),  # #4 VIDYA(14)/EMA(49)/ft5:EMA(196)
        ("KAMA",20,0.0,0.0, "HMA",72,0.0,0.0, "HMA",288,0.0,0.0),  # #5 KAMA(20)/HMA(72)/ft5:HMA(288)
    ],
    "STX/USDT": [
        ("VIDYA",20,0.0,0.0, "VWMA",54,0.0,0.0, "VWMA",216,0.0,0.0),  # #1 VIDYA(20)/VWMA(54)/ft5:VWMA(216)
        ("VIDYA",22,0.0,0.0, "ALMA",63,0.5,6.0, "ALMA",252,0.5,6.0),  # #2 VIDYA(22)/ALMA(63,0.5,6)/ft5:ALMA(252)
        ("VIDYA",24,0.0,0.0, "ALMA",66,0.5,6.0, "ALMA",264,0.5,6.0),  # #3 VIDYA(24)/ALMA(66,0.5,6)/ft5:ALMA(264)
        ("VIDYA",24,0.0,0.0, "ALMA",69,0.5,6.0, "ALMA",276,0.5,6.0),  # #4 VIDYA(24)/ALMA(69,0.5,6)/ft5:ALMA(276)
        ("VIDYA",22,0.0,0.0, "VWMA",57,0.0,0.0, "VWMA",228,0.0,0.0),  # #5 VIDYA(22)/VWMA(57)/ft5:VWMA(228)
    ],
    "SUI/USDT": [
        ("VIDYA",22,0.0,0.0, "McGinley",36,0.0,0.0, "McGinley",144,0.0,0.0),  # #1 VIDYA(22)/McGinley(36)/ft5:McGinley(144)
        ("SSmoother",24,0.0,0.0, "KAMA",36,0.0,0.0, "KAMA",144,0.0,0.0),  # #2 SSmoother(24)/KAMA(36)/ft5:KAMA(144)
        ("VIDYA",20,0.0,0.0, "KAMA",33,0.0,0.0, "KAMA",132,0.0,0.0),  # #3 VIDYA(20)/KAMA(33)/ft5:KAMA(132)
        ("McGinley",18,0.0,0.0, "KAMA",39,0.0,0.0, "KAMA",156,0.0,0.0),  # #4 McGinley(18)/KAMA(39)/ft5:KAMA(156)
        ("McGinley",20,0.0,0.0, "KAMA",45,0.0,0.0, "KAMA",180,0.0,0.0),  # #5 McGinley(20)/KAMA(45)/ft5:KAMA(180)
        ("SSmoother",20,0.0,0.0, "KAMA",33,0.0,0.0, "KAMA",132,0.0,0.0),  # #6 SSmoother(20)/KAMA(33)/ft5:KAMA(132)  # AGRESIVO
    ],
    "TAO/USDT": [
        ("VIDYA",24,0.0,0.0, "SMA",45,0.0,0.0, "SMA",180,0.0,0.0),  # #1 VIDYA(24)/SMA(45)/ft5:SMA(180)
        ("VIDYA",24,0.0,0.0, "SMA",45,0.0,0.0, "T3",250,0.8,0.0),  # #2 VIDYA(24)/SMA(45)/T3(250,v0.8)
        ("VIDYA",20,0.0,0.0, "KAMA",51,0.0,0.0, "KAMA",204,0.0,0.0),  # #3 VIDYA(20)/KAMA(51)/ft5:KAMA(204)
        ("VIDYA",12,0.0,0.0, "McGinley",33,0.0,0.0, "McGinley",132,0.0,0.0),  # #4 VIDYA(12)/McGinley(33)/ft5:McGinley(132)
        ("VIDYA",14,0.0,0.0, "ALMA",75,0.85,4.0, "ALMA",300,0.85,4.0),  # #5 VIDYA(14)/ALMA(75,0.85,4)/ft5:ALMA(300)
    ],
    "THETA/USDT": [
        ("McGinley",22,0.0,0.0, "KAMA",42,0.0,0.0, "KAMA",168,0.0,0.0),  # #1 McGinley(22)/KAMA(42)/ft5:KAMA(168)
        ("VIDYA",24,0.0,0.0, "KAMA",48,0.0,0.0, "KAMA",192,0.0,0.0),  # #2 VIDYA(24)/KAMA(48)/ft5:KAMA(192)
        ("VIDYA",24,0.0,0.0, "KAMA",49,0.0,0.0, "KAMA",196,0.0,0.0),  # #3 VIDYA(24)/KAMA(49)/ft5:KAMA(196)
        ("McGinley",22,0.0,0.0, "KAMA",45,0.0,0.0, "KAMA",180,0.0,0.0),  # #4 McGinley(22)/KAMA(45)/ft5:KAMA(180)
        ("McGinley",22,0.0,0.0, "KAMA",39,0.0,0.0, "KAMA",156,0.0,0.0),  # #5 McGinley(22)/KAMA(39)/ft5:KAMA(156)
    ],
    "TRX/USDT": [
        ("McGinley",24,0.0,0.0, "KAMA",69,0.0,0.0, "KAMA",276,0.0,0.0),  # #1 McGinley(24)/KAMA(69)/ft5:KAMA(276)
        ("McGinley",22,0.0,0.0, "KAMA",66,0.0,0.0, "KAMA",264,0.0,0.0),  # #2 McGinley(22)/KAMA(66)/ft5:KAMA(264)
        ("VIDYA",18,0.0,0.0, "KAMA",57,0.0,0.0, "KAMA",228,0.0,0.0),  # #3 VIDYA(18)/KAMA(57)/ft5:KAMA(228)
        ("McGinley",22,0.0,0.0, "KAMA",63,0.0,0.0, "KAMA",252,0.0,0.0),  # #4 McGinley(22)/KAMA(63)/ft5:KAMA(252)
        ("McGinley",24,0.0,0.0, "KAMA",57,0.0,0.0, "KAMA",228,0.0,0.0),  # #5 McGinley(24)/KAMA(57)/ft5:KAMA(228)
    ],
    "UNI/USDT": [
        ("KAMA",16,0.0,0.0, "HMA",54,0.0,0.0, "HMA",216,0.0,0.0),  # #1 KAMA(16)/HMA(54)/ft5:HMA(216)
        ("KAMA",16,0.0,0.0, "HMA",57,0.0,0.0, "HMA",228,0.0,0.0),  # #2 KAMA(16)/HMA(57)/ft5:HMA(228)
        ("KAMA",16,0.0,0.0, "HMA",51,0.0,0.0, "HMA",204,0.0,0.0),  # #3 KAMA(16)/HMA(51)/ft5:HMA(204)
        ("McGinley",14,0.0,0.0, "TEMA",45,0.0,0.0, "TEMA",180,0.0,0.0),  # #4 McGinley(14)/TEMA(45)/ft5:TEMA(180)
        ("KAMA",14,0.0,0.0, "DEMA",39,0.0,0.0, "DEMA",156,0.0,0.0),  # #5 KAMA(14)/DEMA(39)/ft5:DEMA(156)
    ],
    "VET/USDT": [
        ("McGinley",22,0.0,0.0, "KAMA",42,0.0,0.0, "KAMA",168,0.0,0.0),  # #1 McGinley(22)/KAMA(42)/ft5:KAMA(168)
        ("McGinley",20,0.0,0.0, "KAMA",45,0.0,0.0, "KAMA",180,0.0,0.0),  # #2 McGinley(20)/KAMA(45)/ft5:KAMA(180)
        ("McGinley",22,0.0,0.0, "KAMA",45,0.0,0.0, "KAMA",180,0.0,0.0),  # #3 McGinley(22)/KAMA(45)/ft5:KAMA(180)
        ("McGinley",16,0.0,0.0, "KAMA",39,0.0,0.0, "KAMA",156,0.0,0.0),  # #4 McGinley(16)/KAMA(39)/ft5:KAMA(156)
        ("McGinley",20,0.0,0.0, "KAMA",42,0.0,0.0, "KAMA",168,0.0,0.0),  # #5 McGinley(20)/KAMA(42)/ft5:KAMA(168)
    ],
    "WLD/USDT": [
        ("VIDYA",18,0.0,0.0, "T3",39,0.8,0.0, "T3",156,0.8,0.0),  # #1 VIDYA(18)/T3(39,v0.8)/ft5:T3(156)
        ("SMA",16,0.0,0.0, "HMA",69,0.0,0.0, "HMA",276,0.0,0.0),  # #2 SMA(16)/HMA(69)/ft5:HMA(276)
        ("VIDYA",18,0.0,0.0, "T3",39,0.7,0.0, "T3",156,0.7,0.0),  # #3 VIDYA(18)/T3(39,v0.7)/ft5:T3(156)
        ("McGinley",8,0.0,0.0, "DEMA",66,0.0,0.0, "DEMA",264,0.0,0.0),  # #4 McGinley(8)/DEMA(66)/ft5:DEMA(264)
        ("McGinley",8,0.0,0.0, "DEMA",69,0.0,0.0, "DEMA",276,0.0,0.0),  # #5 McGinley(8)/DEMA(69)/ft5:DEMA(276)
    ],
    "XLM/USDT": [
        ("VIDYA",14,0.0,0.0, "KAMA",48,0.0,0.0, "KAMA",192,0.0,0.0),  # #1 VIDYA(14)/KAMA(48)/ft5:KAMA(192)
        ("VIDYA",18,0.0,0.0, "KAMA",51,0.0,0.0, "KAMA",204,0.0,0.0),  # #2 VIDYA(18)/KAMA(51)/ft5:KAMA(204)
        ("VIDYA",24,0.0,0.0, "KAMA",30,0.0,0.0, "KAMA",120,0.0,0.0),  # #3 VIDYA(24)/KAMA(30)/ft5:KAMA(120)
        ("VIDYA",12,0.0,0.0, "KAMA",54,0.0,0.0, "KAMA",216,0.0,0.0),  # #4 VIDYA(12)/KAMA(54)/ft5:KAMA(216)
        ("VIDYA",14,0.0,0.0, "KAMA",49,0.0,0.0, "KAMA",196,0.0,0.0),  # #5 VIDYA(14)/KAMA(49)/ft5:KAMA(196)
    ],
    "XRP/USDT": [
        ("McGinley",14,0.0,0.0, "EMA",33,0.0,0.0, "EMA",132,0.0,0.0),  # #1 McGinley(14)/EMA(33)/ft5:EMA(132)
        ("McGinley",16,0.0,0.0, "EMA",39,0.0,0.0, "EMA",156,0.0,0.0),  # #2 McGinley(16)/EMA(39)/ft5:EMA(156)
        ("KAMA",12,0.0,0.0, "HMA",69,0.0,0.0, "HMA",276,0.0,0.0),  # #3 KAMA(12)/HMA(69)/ft5:HMA(276)
        ("McGinley",24,0.0,0.0, "KAMA",66,0.0,0.0, "KAMA",264,0.0,0.0),  # #4 McGinley(24)/KAMA(66)/ft5:KAMA(264)
        ("McGinley",14,0.0,0.0, "EMA",30,0.0,0.0, "EMA",120,0.0,0.0),  # #5 McGinley(14)/EMA(30)/ft5:EMA(120)
        ("KAMA",14,0.0,0.0, "SSmoother",33,0.0,0.0, "SSmoother",132,0.0,0.0),  # #8 KAMA(14)/SSmoother(33)/ft5:SSmoother(132)  # AGRESIVO
    ],
}

# Cluster IDs por preset (pipeline lo llena con cluster_id de lab_lite v5e)
PRESET_CLUSTER_IDS = {}  # symbol → [cluster_id per preset]

# Histéresis ATR: probar ambas variantes
HYST_VALUES = [0.0, 0.50]
HYST_ATR_LEN = 14


SL_PERCENT = 3.0
SL_EMERGENCY_PERCENT = 5.0
TS_PERCENT = 0.5

COOLDOWN_BARS = 1

# Comisiones round-trip (entry + exit)
# BingX Futuros: ~0.05% por operación → 0.10% round-trip
COMMISSION_ROUND_TRIP = 0.10

OUTPUT_DIR = "resultados_numba_v8_3"  # Pipeline lo sobreescribe

# Divergencias
DIV_RSI_LEN = 14
DIV_MACD_FAST = 12
DIV_MACD_SLOW = 26
DIV_MACD_SIGNAL = 9
DIV_STOCH_K = 14
DIV_STOCH_D = 3
DIV_STOCH_SMOOTH = 3
DIV_VWMACD_FAST = 12
DIV_VWMACD_SLOW = 26
DIV_CMF_LEN = 21
DIV_CCI_LEN = 10
DIV_MOM_LEN = 10
DIV_PIVOT_PERIOD = 5
DIV_MAX_PIVOTS = 10
DIV_MAX_BARS = 100

# ============================================
# EXCHANGE
# ============================================
exchange = ccxt.binance({'enableRateLimit': True})

# ============================================
# DESCARGA DE DATOS
# ============================================

def fetch_all_candles(symbol, timeframe, total_candles, max_retries=3):
    print(f"   [DOWN] Descargando {total_candles} velas de {symbol} ({timeframe})...")
    
    all_candles = []
    limit_per_request = 1000
    
    tf_ms = {'1h': 3600000, '4h': 14400000, '1d': 86400000}
    interval_ms = tf_ms.get(timeframe, 3600000)
    current_since = int(time.time() * 1000) - (total_candles + 200) * interval_ms
    
    requests_made = 0
    consecutive_errors = 0
    
    while len(all_candles) < total_candles + 100:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit_per_request)
            if not ohlcv:
                break
            all_candles.extend(ohlcv)
            current_since = ohlcv[-1][0] + interval_ms
            requests_made += 1
            consecutive_errors = 0
            time.sleep(0.1)
        except Exception as e:
            consecutive_errors += 1
            if consecutive_errors >= max_retries:
                print(f"   [ERROR] Error tras {max_retries} reintentos: {e}")
                return None
            print(f"   [WARN] Error (reintento {consecutive_errors}/{max_retries}): {e}")
            time.sleep(2 * consecutive_errors)
    
    if len(all_candles) < 100:
        print(f"   [ERROR] Solo se obtuvieron {len(all_candles)} velas")
        return None
    
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # UTC internally
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    
    if len(df) > total_candles:
        df = df.tail(total_candles).reset_index(drop=True)
    
    print(f"   [OK] {len(df)} velas descargadas ({requests_made} requests)")
    return df

# ============================================
# CÁLCULO DE INDICADORES (sin cambios vs v6.0)
# ============================================

def calc_heikin_ashi(df):
    n = len(df)
    ha_open = np.zeros(n)
    ha_close = np.zeros(n)
    ha_high = np.zeros(n)
    ha_low = np.zeros(n)
    
    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    
    ha_close[0] = (o[0] + h[0] + l[0] + c[0]) / 4
    ha_open[0] = (o[0] + c[0]) / 2
    ha_high[0] = max(h[0], ha_open[0], ha_close[0])
    ha_low[0] = min(l[0], ha_open[0], ha_close[0])
    
    for i in range(1, n):
        ha_close[i] = (o[i] + h[i] + l[i] + c[i]) / 4
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
        ha_high[i] = max(h[i], ha_open[i], ha_close[i])
        ha_low[i] = min(l[i], ha_open[i], ha_close[i])
    
    ha = pd.DataFrame({
        'open': ha_open,
        'high': ha_high,
        'low': ha_low,
        'close': ha_close
    }, index=df.index)
    return ha

def calc_tenkan(df, length):
    high_max = df['high'].rolling(window=length, min_periods=1).max()
    low_min = df['low'].rolling(window=length, min_periods=1).min()
    return (high_max + low_min) / 2

def calc_ha_trend(df):
    avg_price = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    avg_oc = (df['open'] + df['close'].shift(1)) / 2
    return avg_price > avg_oc

def get_slow_line(df_slow, use_smoothing=False, smoothing_len=10, shift=0):
    ha = calc_heikin_ashi(df_slow)
    hlc3 = (ha['high'] + ha['low'] + ha['close']) / 3
    if use_smoothing:
        hlc3 = hlc3.rolling(window=smoothing_len, min_periods=1).mean()
    if shift > 0:
        hlc3 = hlc3.shift(shift)
    return hlc3

def calc_rsi(close, length=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    avg_gain = gain.ewm(alpha=1.0/length, adjust=False, min_periods=1).mean()
    avg_loss = loss.ewm(alpha=1.0/length, adjust=False, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=1).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=1).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=1).mean()
    histogram = macd_line - signal_line
    return macd_line, histogram

def calc_stochastic(high, low, close, k=14, d=3, smooth=3):
    lowest_low = low.rolling(window=k, min_periods=1).min()
    highest_high = high.rolling(window=k, min_periods=1).max()
    stoch_range = highest_high - lowest_low
    stoch_raw = np.where(stoch_range > 0, 100 * (close - lowest_low) / stoch_range, 0.0)
    stoch_k = pd.Series(stoch_raw, index=close.index).rolling(window=smooth, min_periods=1).mean()
    return stoch_k

def calc_vwmacd(close, volume, fast=12, slow=26):
    def vwma(price, vol, length):
        return (price * vol).rolling(window=length, min_periods=1).sum() / vol.rolling(window=length, min_periods=1).sum()
    return vwma(close, volume, fast) - vwma(close, volume, slow)

def calc_cmf(high, low, close, volume, length=21):
    h_l = high - low
    ad = ((close - low) - (high - close)) / h_l.replace(0, np.nan) * volume
    ad = ad.fillna(0.0)
    sma_ad = ad.rolling(window=length, min_periods=1).mean()
    sma_vol = volume.rolling(window=length, min_periods=1).mean()
    result = sma_ad / sma_vol.replace(0, np.nan)
    return result.fillna(0.0)

def calc_cci(high, low, close, length=10):
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=length, min_periods=1).mean()
    mad = tp.rolling(window=length, min_periods=1).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mad)

def calc_momentum(close, length=10):
    return close - close.shift(length)

def resample_to_timeframe(df, target_tf):
    df_indexed = df.set_index('timestamp')
    if target_tf == "4h":
        rule = '4h'
    elif target_tf == "1d":
        rule = '1D'
    else:
        return df_indexed
    
    return df_indexed.resample(rule, closed='left', label='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()

# ============================================
# DIVERGENCIAS (sin cambios vs v6.0)
# ============================================

def precalculate_divergences_pine_faithful(close_arr, high_arr, low_arr, indicators_dict, 
                                            n_bars, period=5, max_pivots=10, max_bars=100):
    n_indicators = len(indicators_dict)
    div_bits_arr = np.zeros((n_bars, n_indicators), dtype=np.uint8)
    
    pl_positions = []
    pl_vals = []
    ph_positions = []
    ph_vals = []
    
    indicator_arrays = list(indicators_dict.values())
    
    startpoint = 1
    
    for t in range(period, n_bars):
        pivot_bar = t - period
        
        if pivot_bar >= period:
            is_pivot_low = True
            for j in range(1, period + 1):
                if close_arr[pivot_bar] > close_arr[pivot_bar - j] or close_arr[pivot_bar] > close_arr[pivot_bar + j]:
                    is_pivot_low = False
                    break
            if is_pivot_low:
                pl_positions.insert(0, t)
                pl_vals.insert(0, close_arr[pivot_bar])
                if len(pl_positions) > 20:
                    pl_positions.pop()
                    pl_vals.pop()
            
            is_pivot_high = True
            for j in range(1, period + 1):
                if close_arr[pivot_bar] < close_arr[pivot_bar - j] or close_arr[pivot_bar] < close_arr[pivot_bar + j]:
                    is_pivot_high = False
                    break
            if is_pivot_high:
                ph_positions.insert(0, t)
                ph_vals.insert(0, close_arr[pivot_bar])
                if len(ph_positions) > 20:
                    ph_positions.pop()
                    ph_vals.pop()
        
        if t < 1:
            continue
        
        for ind_idx in range(n_indicators):
            src = indicator_arrays[ind_idx]
            if np.isnan(src[t]):
                continue
            
            bull_reg = False
            bull_hid = False
            bear_reg = False
            bear_hid = False
            
            can_check_pos = t >= 1 and ((src[t] > src[t-1]) or (close_arr[t] > close_arr[t-1]))
            if can_check_pos and len(pl_positions) > 0:
                for x in range(min(max_pivots, len(pl_positions))):
                    pos = pl_positions[x]
                    if pos == 0: break
                    length = t - pos + period
                    if length > max_bars: break
                    if length > 5 and t >= length:
                        ss = src[t - startpoint]; se = src[t - length] if (t - length) >= 0 else np.nan
                        ps = close_arr[t - startpoint]; pe = pl_vals[x]
                        if np.isnan(ss) or np.isnan(se): continue
                        is_reg = (ss > se) and (ps < pe)
                        is_hid = (ss < se) and (ps > pe)
                        if is_reg or is_hid:
                            span = length - startpoint
                            if span <= 0: continue
                            sl_s = (ss - se) / span; sl_p = (ps - pe) / span
                            vs = ss - sl_s; vp = ps - sl_p
                            ok = True
                            for y in range(1 + startpoint, length):
                                bb = t - y
                                if bb < 0 or src[bb] < vs or close_arr[bb] < vp: ok = False; break
                                vs -= sl_s; vp -= sl_p
                            if ok:
                                if is_reg: bull_reg = True
                                if is_hid: bear_hid = True
                                break
            
            can_check_neg = t >= 1 and ((src[t] < src[t-1]) or (close_arr[t] < close_arr[t-1]))
            if can_check_neg and len(ph_positions) > 0:
                for x in range(min(max_pivots, len(ph_positions))):
                    pos = ph_positions[x]
                    if pos == 0: break
                    length = t - pos + period
                    if length > max_bars: break
                    if length > 5 and t >= length:
                        ss = src[t - startpoint]; se = src[t - length] if (t - length) >= 0 else np.nan
                        ps = close_arr[t - startpoint]; pe = ph_vals[x]
                        if np.isnan(ss) or np.isnan(se): continue
                        is_reg = (ss < se) and (ps > pe)
                        is_hid = (ss > se) and (ps < pe)
                        if is_reg or is_hid:
                            span = length - startpoint
                            if span <= 0: continue
                            sl_s = (ss - se) / span; sl_p = (ps - pe) / span
                            vs = ss - sl_s; vp = ps - sl_p
                            ok = True
                            for y in range(1 + startpoint, length):
                                bb = t - y
                                if bb < 0 or src[bb] > vs or close_arr[bb] > vp: ok = False; break
                                vs -= sl_s; vp -= sl_p
                            if ok:
                                if is_reg: bear_reg = True
                                if is_hid: bull_hid = True
                                break
            
            bits = 0
            if bull_reg: bits |= 1
            if bull_hid: bits |= 2
            if bear_reg: bits |= 4
            if bear_hid: bits |= 8
            div_bits_arr[t, ind_idx] = bits
    
    return div_bits_arr


# ============================================
# CÁLCULO MEDIAS MÓVILES v8.1 — 16 FAMILIAS (Pine-faithful)
# ============================================

def calc_ema(src, period):
    """EMA — fiel a Pine Script ta.ema()"""
    n = len(src)
    result = np.full(n, np.nan)
    k = 2.0 / (period + 1)
    result[0] = src[0]
    for i in range(1, n):
        if np.isnan(result[i-1]):
            result[i] = src[i]
        else:
            result[i] = src[i] * k + result[i-1] * (1 - k)
    return result

def calc_sma(src, period):
    """SMA"""
    n = len(src)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        result[i] = np.mean(src[i - period + 1:i + 1])
    return result

def calc_kama(close_arr, period, fast_sc=2, slow_sc=30):
    """KAMA — fiel a Pine Script"""
    n = len(close_arr)
    result = np.full(n, np.nan)
    if period >= n: return result
    result[period - 1] = close_arr[period - 1]
    fast_k = 2.0 / (fast_sc + 1)
    slow_k = 2.0 / (slow_sc + 1)
    for i in range(period, n):
        direction = abs(close_arr[i] - close_arr[i - period])
        volatility = np.sum(np.abs(np.diff(close_arr[i - period:i + 1])))
        er = direction / volatility if volatility != 0 else 0.0
        sc = (er * (fast_k - slow_k) + slow_k) ** 2
        result[i] = result[i - 1] + sc * (close_arr[i] - result[i - 1])
    return result

def calc_zlema(close_arr, period):
    """ZLEMA — fiel a Pine Script"""
    n = len(close_arr)
    lag = (period - 1) // 2
    adjusted = np.full(n, np.nan)
    for i in range(lag, n): adjusted[i] = 2.0 * close_arr[i] - close_arr[i - lag]
    result = np.full(n, np.nan)
    k = 2.0 / (period + 1)
    start = lag; end = start + period
    if end > n: return result
    result[end - 1] = np.mean(adjusted[start:end])
    for i in range(end, n): result[i] = adjusted[i] * k + result[i - 1] * (1 - k)
    return result

def calc_alma(close_arr, period, offset=0.85, sigma=6):
    """ALMA — fiel a Pine Script ta.alma()"""
    n = len(close_arr)
    result = np.full(n, np.nan)
    if period < 1: return result
    m = offset * (period - 1)
    s = period / sigma
    weights = np.array([np.exp(-((i - m) ** 2) / (2 * s * s)) for i in range(period)])
    w_sum = np.sum(weights)
    if w_sum == 0: return result
    weights = weights / w_sum
    for i in range(period - 1, n): result[i] = np.sum(close_arr[i - period + 1:i + 1] * weights)
    return result

def calc_dema(src, period):
    """DEMA — fiel a Pine Script"""
    e1 = calc_ema(src, period)
    n = len(src)
    e2 = np.full(n, np.nan)
    k = 2.0 / (period + 1)
    for i in range(n):
        if not np.isnan(e1[i]):
            e2[i] = e1[i] if np.isnan(e2[i-1] if i > 0 else np.nan) else e1[i] * k + e2[i-1] * (1 - k)
    return 2 * e1 - e2

def calc_tema(src, period):
    """TEMA — fiel a Pine Script"""
    e1 = calc_ema(src, period)
    n = len(src)
    k = 2.0 / (period + 1)
    e2 = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(e1[i]):
            e2[i] = e1[i] if (i == 0 or np.isnan(e2[i-1])) else e1[i] * k + e2[i-1] * (1 - k)
    e3 = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(e2[i]):
            e3[i] = e2[i] if (i == 0 or np.isnan(e3[i-1])) else e2[i] * k + e3[i-1] * (1 - k)
    return 3 * e1 - 3 * e2 + e3

def calc_jma(src, period, phase=0.0, power=2.0):
    """JMA (Jurik approximation) — fiel a Pine Script"""
    n = len(src)
    phase_ratio = 0.5 if phase < -100 else (2.5 if phase > 100 else phase / 100.0 + 1.5)
    beta = 0.45 * (period - 1) / (0.45 * (period - 1) + 2)
    alpha = beta ** power
    e0 = np.zeros(n); e1 = np.zeros(n); e2 = np.zeros(n)
    result = np.full(n, np.nan)
    e0[0] = src[0]; result[0] = src[0]
    for i in range(1, n):
        e0[i] = (1 - alpha) * src[i] + alpha * e0[i-1]
        e1[i] = (src[i] - e0[i]) * (1 - beta) + beta * e1[i-1]
        e2[i] = (e0[i] + phase_ratio * e1[i] - result[i-1]) * (1 - alpha)**2 + alpha**2 * e2[i-1]
        result[i] = result[i-1] + e2[i]
    return result

def calc_mcginley(src, period):
    """McGinley Dynamic — fiel a Pine Script"""
    n = len(src)
    result = np.full(n, np.nan); result[0] = src[0]
    for i in range(1, n):
        prev = result[i-1] if not np.isnan(result[i-1]) else src[i]
        ratio = src[i] / prev if prev != 0 else 1.0
        denom = 0.6 * period * (ratio ** 4)
        result[i] = prev + (src[i] - prev) / denom if denom != 0 else prev
    return result

def calc_vidya(src, period):
    """VIDYA — fiel a Pine Script"""
    n = len(src)
    sc = 2.0 / (period + 1)
    result = np.full(n, np.nan); result[0] = src[0]
    for i in range(period, n):
        gains = np.sum(np.maximum(np.diff(src[i-period:i+1]), 0))
        losses = np.sum(np.maximum(-np.diff(src[i-period:i+1]), 0))
        total = gains + losses
        cmo = abs((gains - losses) / total) if total != 0 else 0.0
        prev = result[i-1] if not np.isnan(result[i-1]) else src[i]
        result[i] = src[i] * sc * cmo + prev * (1 - sc * cmo)
    return result

def calc_frama(src, high_arr, low_arr, period):
    """FRAMA — fiel a Pine Script"""
    n = len(src); half = period // 2
    result = np.full(n, np.nan); result[0] = src[0]
    for i in range(period, n):
        n1 = (np.max(high_arr[i-half+1:i+1]) - np.min(low_arr[i-half+1:i+1])) / half if half > 0 else 1
        n2 = (np.max(high_arr[i-period+1:i-half+1]) - np.min(low_arr[i-period+1:i-half+1])) / half if half > 0 else 1
        n3 = (np.max(high_arr[i-period+1:i+1]) - np.min(low_arr[i-period+1:i+1])) / period if period > 0 else 1
        d = (np.log(n1 + n2) - np.log(n3)) / np.log(2) if (n1+n2) > 0 and n3 > 0 else 1.0
        alpha_c = max(0.01, min(np.exp(-4.6 * (d - 1)), 1.0))
        prev = result[i-1] if not np.isnan(result[i-1]) else src[i]
        result[i] = alpha_c * src[i] + (1 - alpha_c) * prev
    return result

def calc_t3(src, period, v=0.7):
    """T3 (Tillson) — fiel a Pine Script"""
    c1 = -(v**3); c2 = 3*v**2 + 3*v**3; c3 = -6*v**2 - 3*v - 3*v**3; c4 = 1 + 3*v + v**3 + 3*v**2
    e1 = calc_ema(src, period); e2 = calc_ema(e1, period); e3 = calc_ema(e2, period)
    e4 = calc_ema(e3, period); e5 = calc_ema(e4, period); e6 = calc_ema(e5, period)
    return c1*e6 + c2*e5 + c3*e4 + c4*e3

def calc_ssmoother(src, period):
    """SuperSmoother (Ehlers) — fiel a Pine Script"""
    n = len(src)
    a = np.exp(-np.sqrt(2) * np.pi / period)
    b = 2 * a * np.cos(np.sqrt(2) * np.pi / period)
    c2 = b; c3 = -(a * a); c1 = 1 - c2 - c3
    result = np.full(n, np.nan); result[0] = src[0]
    if n > 1: result[1] = src[1]
    for i in range(2, n):
        result[i] = c1 * (src[i] + src[i-1]) / 2 + c2 * result[i-1] + c3 * result[i-2]
    return result

def calc_hma(src, period):
    """HMA — fiel a Pine Script"""
    half_len = max(period // 2, 1); sqrt_len = max(round(np.sqrt(period)), 1)
    def wma(x, p):
        s = pd.Series(x)
        return s.rolling(p).apply(lambda x: np.sum(x * np.arange(1,len(x)+1)) / np.sum(np.arange(1,len(x)+1)), raw=True).values
    diff = 2 * wma(src, half_len) - wma(src, period)
    return wma(diff, sqrt_len)

def calc_vwma(src, volume, period):
    """VWMA — fiel a Pine Script"""
    n = len(src); result = np.full(n, np.nan)
    for i in range(period - 1, n):
        sv = np.sum(src[i-period+1:i+1] * volume[i-period+1:i+1])
        v = np.sum(volume[i-period+1:i+1])
        result[i] = sv / v if v != 0 else src[i]
    return result

def calc_tenkan(high_arr, low_arr, period):
    """Tenkan — fiel a Pine Script"""
    n = len(high_arr); result = np.full(n, np.nan)
    for i in range(period - 1, n):
        result[i] = (np.max(high_arr[i-period+1:i+1]) + np.min(low_arr[i-period+1:i+1])) / 2
    return result

def calc_atr(high_arr, low_arr, close_arr, period):
    """ATR — fiel a Pine Script ta.atr()"""
    n = len(close_arr); tr = np.zeros(n)
    tr[0] = high_arr[0] - low_arr[0]
    for i in range(1, n):
        tr[i] = max(high_arr[i]-low_arr[i], abs(high_arr[i]-close_arr[i-1]), abs(low_arr[i]-close_arr[i-1]))
    atr = np.full(n, np.nan); atr[period-1] = np.mean(tr[:period])
    for i in range(period, n): atr[i] = atr[i-1] + (tr[i] - atr[i-1]) / period
    return atr

# ============================================
# MA DISPATCHER (idéntico a f_calc_ma de Pine v10.0)
# ============================================
def calc_ma(ma_type, close_arr, high_arr, low_arr, volume_arr, period, p1=0.0, p2=0.0):
    if ma_type == "NONE": return np.full(len(close_arr), np.nan)
    elif ma_type == "EMA": return calc_ema(close_arr, period)
    elif ma_type == "SMA": return calc_sma(close_arr, period)
    elif ma_type == "HMA": return calc_hma(close_arr, period)
    elif ma_type == "ALMA": return calc_alma(close_arr, period, offset=p1 if p1>0 else 0.85, sigma=p2 if p2>0 else 6)
    elif ma_type == "KAMA": return calc_kama(close_arr, period)
    elif ma_type == "ZLEMA": return calc_zlema(close_arr, period)
    elif ma_type == "DEMA": return calc_dema(close_arr, period)
    elif ma_type == "TEMA": return calc_tema(close_arr, period)
    elif ma_type == "JMA": return calc_jma(close_arr, period, phase=p1, power=p2 if p2 > 0 else 2.0)
    elif ma_type == "McGinley": return calc_mcginley(close_arr, period)
    elif ma_type == "VIDYA": return calc_vidya(close_arr, period)
    elif ma_type == "FRAMA": return calc_frama(close_arr, high_arr, low_arr, period)
    elif ma_type == "T3": return calc_t3(close_arr, period, v=p1 if p1>0 else 0.7)
    elif ma_type in ("SuperSmoother", "SSmoother"): return calc_ssmoother(close_arr, period)
    elif ma_type == "VWMA": return calc_vwma(close_arr, volume_arr, period)
    elif ma_type == "Tenkan": return calc_tenkan(high_arr, low_arr, period)
    else: return calc_ema(close_arr, period)

# ============================================
# HISTÉRESIS ATR (idéntico a Pine v10.0)
# ============================================
def calc_zone_with_hysteresis(ma_fast_arr, ma_slow_arr, atr_arr, hyst_mult):
    n = len(ma_fast_arr)
    zone_bull = np.zeros(n, dtype=np.bool_)
    zone_bear = np.zeros(n, dtype=np.bool_)
    hyst_zone = 0
    for i in range(n):
        if np.isnan(ma_fast_arr[i]) or np.isnan(ma_slow_arr[i]): continue
        hyst_band = hyst_mult * atr_arr[i] if not np.isnan(atr_arr[i]) else 0.0
        if hyst_mult == 0.0:
            hyst_zone = 1 if ma_fast_arr[i] > ma_slow_arr[i] else (-1 if ma_fast_arr[i] < ma_slow_arr[i] else hyst_zone)
        else:
            if hyst_zone <= 0 and ma_fast_arr[i] > ma_slow_arr[i] + hyst_band: hyst_zone = 1
            if hyst_zone >= 0 and ma_fast_arr[i] < ma_slow_arr[i] - hyst_band: hyst_zone = -1
            if hyst_zone == 0:
                hyst_zone = 1 if ma_fast_arr[i] > ma_slow_arr[i] + hyst_band else (-1 if ma_fast_arr[i] < ma_slow_arr[i] - hyst_band else 0)
        zone_bull[i] = (hyst_zone == 1)
        zone_bear[i] = (hyst_zone == -1)
    return zone_bull, zone_bear

# ============================================
# PRE-CÁLCULO COMPLETO (v8.1: Multi-MA + Histéresis)
# ============================================
def precalculate_all_data(df_1h, preset=None, hyst_mult=0.0, symbol=None):
    n = len(df_1h)
    print(f"   [CALC] Pre-calculando indicadores para {n} velas...")
    
    fast_type, fast_len, fast_p1, fast_p2 = preset[0], preset[1], preset[2], preset[3]
    slow_type, slow_len, slow_p1, slow_p2 = preset[4], preset[5], preset[6], preset[7]
    trend_type, trend_len, trend_p1, trend_p2 = preset[8], preset[9], preset[10], preset[11]
    
    print(f"   [STATS] Fase 1: MAs")
    print(f"      Fast: {fast_type}({fast_len}) | Slow: {slow_type}({slow_len}) | Trend: {trend_type}({trend_len}) | Hyst: {hyst_mult}×ATR({HYST_ATR_LEN})")
    
    df_1h_indexed = df_1h.set_index('timestamp')
    
    df_4h_full = df_1h_indexed.resample('4h', closed='left', label='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    df_1d_full = df_1h_indexed.resample('1D', closed='left', label='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    
    tf2_trend_resolved = calc_ha_trend(df_4h_full)
    tf2_bull_resolved = tf2_trend_resolved.shift(1).reindex(df_1h_indexed.index).ffill().fillna(True).values
    tf3_trend_resolved = calc_ha_trend(df_1d_full)
    tf3_bull_resolved = tf3_trend_resolved.shift(1).reindex(df_1h_indexed.index).ffill().fillna(True).values
    
    close_arr = df_1h['close'].values.astype(np.float64)
    high_arr = df_1h['high'].values.astype(np.float64)
    low_arr = df_1h['low'].values.astype(np.float64)
    volume_arr = df_1h['volume'].values.astype(np.float64)
    
    # v8.1: Multi-MA por dispatcher (idéntico a Pine v10.0)
    ma_fast_arr = calc_ma(fast_type, close_arr, high_arr, low_arr, volume_arr, fast_len, fast_p1, fast_p2)
    ma_slow_arr = calc_ma(slow_type, close_arr, high_arr, low_arr, volume_arr, slow_len, slow_p1, slow_p2)
    
    # v8.1: Histéresis ATR
    atr_arr = calc_atr(high_arr, low_arr, close_arr, HYST_ATR_LEN)
    zone_bull_arr, zone_bear_arr = calc_zone_with_hysteresis(ma_fast_arr, ma_slow_arr, atr_arr, hyst_mult)
    
    # TF4: close vs Slow MA
    tf4_bull_arr = close_arr > ma_slow_arr
    tf4_bear_arr = close_arr < ma_slow_arr
    
    # TF5: Trend MA
    ma_trend_arr = calc_ma(trend_type, close_arr, high_arr, low_arr, volume_arr, trend_len, trend_p1, trend_p2)
    tf5_bull_resolved = np.where(np.isnan(ma_trend_arr), False, close_arr > ma_trend_arr)
    tf5_bear_resolved = np.where(np.isnan(ma_trend_arr), False, close_arr < ma_trend_arr)
    
    print(f"   [OK] Fase 1 completada")
    print(f"   [CALC] Fase 2: Simulando forward testing (forming para TF2/TF3)...")
    
    filters_forming_arr = np.zeros(n, dtype=np.uint32)
    filters_resolved_arr = np.zeros(n, dtype=np.uint32)
    
    for bar_idx in range(100, n):
        df_until_now = df_1h.iloc[:bar_idx+1].copy()
        df_4h = resample_to_timeframe(df_until_now, "4h")
        df_1d = resample_to_timeframe(df_until_now, "1d")
        tf1_bull = calc_ha_trend(df_until_now).iloc[-1] if len(df_until_now) > 1 else True
        tf2_trend = calc_ha_trend(df_4h)
        tf2_bull_forming = tf2_trend.iloc[-1] if len(tf2_trend) > 0 else True
        tf3_trend = calc_ha_trend(df_1d)
        tf3_bull_forming = tf3_trend.iloc[-1] if len(tf3_trend) > 0 else True

        
        tf4_bull = tf4_bull_arr[bar_idx]
        tf4_bear = tf4_bear_arr[bar_idx]
        
        tf5_bull = tf5_bull_resolved[bar_idx]
        tf5_bear = tf5_bear_resolved[bar_idx]
        
        f_forming = 0
        if tf1_bull: f_forming |= (1 << 0)
        if tf2_bull_forming: f_forming |= (1 << 1)
        if tf3_bull_forming: f_forming |= (1 << 2)
        if tf4_bull: f_forming |= (1 << 3)
        if tf5_bull: f_forming |= (1 << 4)
        if tf4_bear: f_forming |= (1 << 11)
        if tf5_bear: f_forming |= (1 << 12)
        filters_forming_arr[bar_idx] = f_forming
        
        f_resolved = 0
        if tf1_bull: f_resolved |= (1 << 0)
        if tf2_bull_resolved[bar_idx]: f_resolved |= (1 << 1)
        if tf3_bull_resolved[bar_idx]: f_resolved |= (1 << 2)
        if tf4_bull: f_resolved |= (1 << 3)
        if tf5_bull: f_resolved |= (1 << 4)
        if tf4_bear: f_resolved |= (1 << 11)
        if tf5_bear: f_resolved |= (1 << 12)
        filters_resolved_arr[bar_idx] = f_resolved
        
        if bar_idx % 500 == 0:
            print(f"      Vela {bar_idx}/{n} pre-calculada...")
    
    print(f"   [OK] Fase 2 completada")
    
    print(f"   [CALC] Fase 3: Calculando divergencias (Pine-faithful)...")
    
    rsi_full = calc_rsi(df_1h['close'], DIV_RSI_LEN).values
    macd_line_full, macd_hist_full = calc_macd(df_1h['close'], DIV_MACD_FAST, DIV_MACD_SLOW, DIV_MACD_SIGNAL)
    macd_line_full = macd_line_full.values
    macd_hist_full = macd_hist_full.values
    stoch_full = calc_stochastic(df_1h['high'], df_1h['low'], df_1h['close'], DIV_STOCH_K, DIV_STOCH_D, DIV_STOCH_SMOOTH).values
    vwmacd_full = calc_vwmacd(df_1h['close'], df_1h['volume'], DIV_VWMACD_FAST, DIV_VWMACD_SLOW).values
    cmf_full = calc_cmf(df_1h['high'], df_1h['low'], df_1h['close'], df_1h['volume'], DIV_CMF_LEN).values
    cci_full = calc_cci(df_1h['high'], df_1h['low'], df_1h['close'], DIV_CCI_LEN).values
    mom_full = calc_momentum(df_1h['close'], DIV_MOM_LEN).values
    
    indicators = {
        'rsi': rsi_full,
        'macd_hist': macd_hist_full,
        'macd_line': macd_line_full,
        'stoch': stoch_full,
        'vwmacd': vwmacd_full,
        'cmf': cmf_full,
        'cci': cci_full,
        'mom': mom_full
    }
    
    div_bits_arr = precalculate_divergences_pine_faithful(
        close_arr, high_arr, low_arr, indicators,
        n, DIV_PIVOT_PERIOD, DIV_MAX_PIVOTS, DIV_MAX_BARS
    )
    
    ind_names = ["RSI", "MACD_H", "MACD_L", "STOCH", "VWMACD", "CMF", "CCI", "MOM"]
    print(f"   [STATS] Divergencias detectadas:")
    print(f"      {'Ind':<8} {'BullR':>6} {'BullH':>6} {'BearR':>6} {'BearH':>6}")
    total_hidden = 0
    for ii, name in enumerate(ind_names):
        br = int(np.sum((div_bits_arr[100:, ii] & 1) > 0))
        bh = int(np.sum((div_bits_arr[100:, ii] & 2) > 0))
        ber = int(np.sum((div_bits_arr[100:, ii] & 4) > 0))
        beh = int(np.sum((div_bits_arr[100:, ii] & 8) > 0))
        total_hidden += bh + beh
        print(f"      {name:<8} {br:>6} {bh:>6} {ber:>6} {beh:>6}")
    print(f"      Total hidden: {total_hidden}")
    
    print(f"   [OK] Fase 3 completada")
    
    return {
        'close': close_arr,
        'high': high_arr,
        'low': low_arr,
        'timestamps': df_1h['timestamp'].values,
        'zone_bull': zone_bull_arr,
        'zone_bear': zone_bear_arr,
        'filters_forming': filters_forming_arr,
        'filters_resolved': filters_resolved_arr,
        'div_bits': div_bits_arr,
    }

# ============================================
# GENERACIÓN DE CONFIGS (sin cambios vs v6.0)
# ============================================

def generate_valid_configs():
    valid_configs = []
    
    # Configs SIN divergencias (div_entry=OFF, div_exit=OFF, div_type=NONE)
    # reg_inv y hid_inv son irrelevantes → solo 1 combo (ambos=0)
    for use_ts in range(2):
        for cancel_tf in range(2):
            for exit_bits in range(16):
                for entry_bits in range(32):
                    config_id = (
                        exit_bits |
                        (entry_bits << 4) |
                        (cancel_tf << 22) |
                        (use_ts << 23)
                        # bits 24-25 = 0 (irrelevante, no se generan combos)
                    )
                    valid_configs.append(config_id)
    
    # Configs CON divergencias
    for use_ts in range(2):
        for cancel_tf in range(2):
            for exit_bits in range(16):
                for entry_bits in range(32):
                    for div_entry_mode in range(3):
                        for div_exit in range(2):
                            if div_entry_mode == 0 and div_exit == 0:
                                continue
                            for div_type in range(1, 4):
                                # Determinar qué combos de inversión son relevantes
                                if div_type == 1:    # REGULAR only → solo reg_inv importa
                                    inv_combos = [(0, 0), (1, 0)]
                                elif div_type == 2:  # HIDDEN only → solo hid_inv importa
                                    inv_combos = [(0, 0), (0, 1)]
                                else:                # BOTH → ambos importan
                                    inv_combos = [(0, 0), (0, 1), (1, 0), (1, 1)]
                                
                                for div_ind in range(1, 256):
                                    for reg_inv, hid_inv in inv_combos:
                                        config_id = (
                                            exit_bits |
                                            (entry_bits << 4) |
                                            (div_entry_mode << 9) |
                                            (div_exit << 11) |
                                            (div_type << 12) |
                                            (div_ind << 14) |
                                            (cancel_tf << 22) |
                                            (use_ts << 23) |
                                            (reg_inv << 24) |
                                            (hid_inv << 25)
                                        )
                                        valid_configs.append(config_id)
    
    return np.array(valid_configs, dtype=np.uint32)

def decode_config(config_id):
    exit_mask = config_id & 0xF
    entry_mask = (config_id >> 4) & 0x1F
    div_entry_mode = (config_id >> 9) & 0x3
    div_exit = (config_id >> 11) & 0x1
    div_type = (config_id >> 12) & 0x3
    div_ind_mask = (config_id >> 14) & 0xFF
    cancel_tf = (config_id >> 22) & 0x1
    use_ts = (config_id >> 23) & 0x1
    reg_inv = (config_id >> 24) & 0x1
    hid_inv = (config_id >> 25) & 0x1
    
    entry_tfs = []
    if entry_mask & 1: entry_tfs.append("TF1")
    if entry_mask & 2: entry_tfs.append("TF2")
    if entry_mask & 4: entry_tfs.append("TF3")
    if entry_mask & 8: entry_tfs.append("TF4")
    if entry_mask & 16: entry_tfs.append("TF5")
    
    exit_tfs = []
    if exit_mask & 1: exit_tfs.append("TF1")
    if exit_mask & 2: exit_tfs.append("TF2")
    if exit_mask & 4: exit_tfs.append("TF3")
    if exit_mask & 8: exit_tfs.append("TF4")
    if exit_mask == 0: exit_tfs.append("ZONA")
    
    div_mode_str = ["OFF", "CONTEXTUAL", "OR"][div_entry_mode]
    div_type_str = ["NONE", "REGULAR", "HIDDEN", "BOTH"][div_type]
    
    indicators = []
    ind_names = ["RSI", "MACD_H", "MACD_L", "STOCH", "VWMACD", "CMF", "CCI", "MOM"]
    for i, name in enumerate(ind_names):
        if div_ind_mask & (1 << i):
            indicators.append(name)
    
    cancel_str = "TF" if cancel_tf else "OFF"
    ts_str = "TS" if use_ts else "SL_FIJO"
    
    # Variant name from inversion bits
    variant_map = {(0,0): "FIX", (0,1): "ORIGINAL", (1,1): "ALLINV", (1,0): "REGINV"}
    variant_str = variant_map.get((reg_inv, hid_inv), "?")
    
    reg_str = "Invertida" if reg_inv else "Tradicional"
    hid_str = "Invertida" if hid_inv else "Tradicional"
    
    n_entry_tfs = bin(entry_mask).count('1')
    
    return {
        'entry_tfs': entry_tfs,
        'exit_tfs': exit_tfs,
        'div_entry_mode': div_mode_str,
        'div_exit': "ON" if div_exit else "OFF",
        'div_type': div_type_str,
        'div_indicators': indicators,
        'cancel_tf': cancel_tf,
        'use_ts': use_ts,
        'cancel_str': cancel_str,
        'ts_str': ts_str,
        'n_entry_tfs': n_entry_tfs,
        'reg_inv': reg_inv,
        'hid_inv': hid_inv,
        'variant': variant_str,
        'reg_str': reg_str,
        'hid_str': hid_str,
    }

# ============================================
# MOTOR NUMBA (sin cambios vs v6.0)
# ============================================

# G1.1 Tier 0 I1 Path γ — per-trade arrays opcionales — 2026-04-29 Sesión 2 Frame 2.
# Granular enum reason_exit TF kernel (sustituye Path α reduced enum Sesión 1B per
# decisión P1 Sesión 1: Path γ SUSTITUYE Path α + Path α' supplement, NO amend).
# Path γ split normal_exit_signal en tf_exit_signal + zone_exit_signal kernel-side
# para preservar info granular cross-strategy decomposition (Sesión 3 R4 Bloque 2c).
# sl_hit vs sl_emergency también separados (reuse sl_emergency_signal flag existente).
# Commits 9282e79 + f8205fa preserved git history trazabilidad.
# Ver §13.4 entrada Sesión 2 Frame 2 R3 Path γ + §12 L38 disciplinada.
REASON_TF_SL_HIT       = 0  # sl_exit_signal=True, sl_emergency_signal=False (on-close 3% TS)
REASON_TF_SL_EMERGENCY = 1  # sl_exit_signal=True, sl_emergency_signal=True (intrabar 5%)
REASON_TF_DIV_EXIT     = 2  # div_exit_signal: salida por divergencia (bit div_exit + div_type>0)
REASON_TF_TF_EXIT      = 3  # tf_exit_signal: TF filters reverse (split de normal_exit_signal)
REASON_TF_ZONE_EXIT    = 4  # zone_exit_signal: zone z_bull/z_bear (split de normal_exit_signal)
REASON_TF_CANCEL_TF    = 5  # cancel_signal: cancel_tf bit 22

# Sentinel arrays para defaults kwargs (pattern Numba @njit con array defaults).
# Backward compat 100%: existing callers no pasan kwargs per-trade → arrays sentinel
# (1,1) reciben writes condicionales que nunca se ejecutan (return_per_trade=False
# default). Flag-driven: solo callers que pasan return_per_trade=True allocaron arrays
# útiles. Memory protection inherente vía dispatch en run_on_slice wrapper Python.
_PT_SENTINEL_INT32 = np.zeros((1, 1), dtype=np.int32)
_PT_SENTINEL_INT8 = np.zeros((1, 1), dtype=np.int8)
_PT_SENTINEL_FLOAT64 = np.zeros((1, 1), dtype=np.float64)
_PT_SENTINEL_COUNT = np.zeros(1, dtype=np.int32)

@jit(nopython=True, parallel=True, cache=True)
def run_simulation_numba(
    configs,
    close_arr, high_arr, low_arr,
    timestamps_i64,
    zone_bull_arr, zone_bear_arr,
    filters_forming_arr, filters_resolved_arr,
    div_bits_arr,
    sl_pct, sl_emergency_pct, ts_pct, cooldown_bars,
    commission_pct,
    accounting_start=100,
    cluster_labels=np.zeros(1, dtype=np.int64),
    n_clusters=1,
    return_per_trade=False,
    pt_entry_bar=_PT_SENTINEL_INT32,
    pt_exit_bar=_PT_SENTINEL_INT32,
    pt_side=_PT_SENTINEL_INT8,
    pt_pnl=_PT_SENTINEL_FLOAT64,
    pt_reason=_PT_SENTINEL_INT8,
    pt_cluster=_PT_SENTINEL_INT8,
    pt_count=_PT_SENTINEL_COUNT,
    # Sesión 1B amendment Path α' supplement 2026-04-28 — entry_price + exit_price
    # output arrays para audit refactor Opción A clean (Sesión 1A.2 prereq).
    # Reduced enum collapsa sl_emergency vs sl_hit; exit_price kernel-output preserva
    # info necesaria audit (entry_price = close[entry_bar] siempre; exit_price =
    # emerg_level si sl_emergency intrabar O close[exit_bar] otherwise).
    pt_entry_price=_PT_SENTINEL_FLOAT64,
    pt_exit_price=_PT_SENTINEL_FLOAT64
):
    n_configs = len(configs)
    n_bars = len(close_arr)

    results = np.zeros((n_configs, 7), dtype=np.float64)

    # Per-cluster result arrays
    cl_pnl_out = np.zeros((n_configs, n_clusters), dtype=np.float64)
    cl_trades_out = np.zeros((n_configs, n_clusters), dtype=np.float64)
    cl_wins_out = np.zeros((n_configs, n_clusters), dtype=np.float64)
    cl_maxdd_out = np.zeros((n_configs, n_clusters), dtype=np.float64)
    cl_gp_out = np.zeros((n_configs, n_clusters), dtype=np.float64)
    cl_gl_out = np.zeros((n_configs, n_clusters), dtype=np.float64)

    has_clusters = len(cluster_labels) > 1

    for c in prange(n_configs):
        cfg = configs[c]
        
        exit_mask = cfg & 0xF
        entry_mask = (cfg >> 4) & 0x1F
        div_entry_mode = (cfg >> 9) & 0x3
        div_exit = (cfg >> 11) & 0x1
        div_type = (cfg >> 12) & 0x3
        div_ind_mask = (cfg >> 14) & 0xFF
        cancel_tf = (cfg >> 22) & 0x1
        use_ts = (cfg >> 23) & 0x1
        reg_inv = (cfg >> 24) & 0x1
        hid_inv = (cfg >> 25) & 0x1
        
        position = 0
        entry_price = 0.0
        entry_bar = 0
        entry_filters_forming = 0
        
        pnl = 0.0
        trades = 0
        wins = 0
        cancels = 0
        peak_pnl = 0.0
        max_dd = 0.0
        gross_profit = 0.0
        gross_loss = 0.0

        # Per-cluster accumulators
        cl_pnl = np.zeros(n_clusters, dtype=np.float64)
        cl_trades = np.zeros(n_clusters, dtype=np.int64)
        cl_wins = np.zeros(n_clusters, dtype=np.int64)
        cl_maxdd = np.zeros(n_clusters, dtype=np.float64)
        cl_gp = np.zeros(n_clusters, dtype=np.float64)
        cl_gl = np.zeros(n_clusters, dtype=np.float64)
        cl_peak = np.zeros(n_clusters, dtype=np.float64)

        div_ctx_bull = False
        div_ctx_bear = False
        last_zone_bull = False
        
        # Pine-faithful: entries use div from PREVIOUS bar
        # (in Pine, div_raw is calculated inside isconfirmed but
        #  entry conditions are evaluated outside using prev bar's values)
        prev_div_bull_now = False
        prev_div_bear_now = False
        prev_div_ctx_bull = False
        prev_div_ctx_bear = False
        
        # Saved div from previous bar (for next bar's entry)
        div_bull_now_saved = False
        div_bear_now_saved = False
        
        cooldown_until = 0
        
        sl_level = 0.0
        
        entry_tf_count = 0
        for bit in range(5):
            if (entry_mask >> bit) & 1:
                entry_tf_count += 1
        
        exit_tf_count = 0
        for bit in range(4):
            if (exit_mask >> bit) & 1:
                exit_tf_count += 1
        
        # Warmup: iterate from bar 1 to build state (last_zone_bull, div_ctx, etc.)
        # but only open trades and accumulate stats from accounting_start onwards.
        acct_start = accounting_start

        for t in range(1, n_bars):
            z_bull = zone_bull_arr[t]
            z_bear = zone_bear_arr[t]
            f_forming = filters_forming_arr[t]
            f_resolved = filters_resolved_arr[t]

            close_p = close_arr[t]
            high_p = high_arr[t]
            low_p = low_arr[t]

            # Pine-faithful: save previous bar's div state for entry evaluation
            # In Pine, entry conditions are evaluated OUTSIDE isconfirmed using
            # div_raw from bar N-1, then div_ctx is updated with zone changes
            # from bar N but div_raw from N-1
            prev_div_bull_now = div_bull_now_saved
            prev_div_bear_now = div_bear_now_saved

            # Phase 1: Zone change resets (uses current bar's zone - same as Pine)
            zone_changed_to_bear = z_bear and last_zone_bull
            zone_changed_to_bull = z_bull and not last_zone_bull
            
            if zone_changed_to_bear:
                div_ctx_bull = False
            if zone_changed_to_bull:
                div_ctx_bear = False
            
            # Phase 2: div_ctx update from PREVIOUS bar's div_raw (Pine L.447-452)
            # This happens outside isconfirmed, so uses N-1 values
            if prev_div_bull_now:
                div_ctx_bull = True
                div_ctx_bear = False
            if prev_div_bear_now:
                div_ctx_bear = True
                div_ctx_bull = False
            
            # Snapshot div_ctx for entry evaluation (before current bar's div changes it)
            entry_div_ctx_bull = div_ctx_bull
            entry_div_ctx_bear = div_ctx_bear
            
            last_zone_bull = z_bull
            
            # Phase 3: Calculate divergence for CURRENT bar (Pine isconfirmed L.565-602)
            div_bull_now = False
            div_bear_now = False
            
            if div_type > 0 and div_ind_mask > 0:
                net_div_score = 0
                for ind in range(8):
                    if not (div_ind_mask & (1 << ind)):
                        continue
                    bits = div_bits_arr[t, ind]
                    ind_bull = False
                    ind_bear = False
                    if div_type == 1:  # REGULAR only
                        if reg_inv == 0:  # Tradicional
                            ind_bull = (bits & 1) > 0   # bit0 pos_reg → bull
                            ind_bear = (bits & 4) > 0   # bit2 neg_reg → bear
                        else:             # Invertida
                            ind_bull = (bits & 4) > 0   # bit2 neg_reg → bull
                            ind_bear = (bits & 1) > 0   # bit0 pos_reg → bear
                    elif div_type == 2:  # HIDDEN only
                        if hid_inv == 0:  # Tradicional
                            ind_bull = (bits & 8) > 0   # bit3 neg_hid → bull
                            ind_bear = (bits & 2) > 0   # bit1 pos_hid → bear
                        else:             # Invertida
                            ind_bull = (bits & 2) > 0   # bit1 pos_hid → bull
                            ind_bear = (bits & 8) > 0   # bit3 neg_hid → bear
                    elif div_type == 3:  # BOTH
                        if reg_inv == 0:
                            reg_bull = (bits & 1) > 0
                            reg_bear = (bits & 4) > 0
                        else:
                            reg_bull = (bits & 4) > 0
                            reg_bear = (bits & 1) > 0
                        if hid_inv == 0:
                            hid_bull = (bits & 8) > 0
                            hid_bear = (bits & 2) > 0
                        else:
                            hid_bull = (bits & 2) > 0
                            hid_bear = (bits & 8) > 0
                        ind_bull = reg_bull or hid_bull
                        ind_bear = reg_bear or hid_bear
                    if ind_bull:
                        net_div_score += 1
                    if ind_bear:
                        net_div_score -= 1
                # Umbral consenso = 1 (matches Pine i_div_showlimit default)
                div_bull_now = net_div_score >= 1
                div_bear_now = net_div_score <= -1
            
            # Save current bar's div_now for next bar's entry evaluation
            div_bull_now_saved = div_bull_now
            div_bear_now_saved = div_bear_now
            
            # Phase 4: Update div_ctx with CURRENT bar's div (Pine isconfirmed L.608-616)
            # This happens AFTER entry was already evaluated, so only affects future bars
            if div_bull_now:
                div_ctx_bull = True
            if div_bear_now:
                div_ctx_bear = True
            
            if position != 0 and t >= acct_start:
                exit_signal = False
                cancel_signal = False
                div_exit_signal = False
                sl_exit_signal = False
                sl_emergency_signal = False
                tf_exit_signal = False
                zone_exit_signal = False
                exit_price = close_p
                
                if use_ts == 1 and t > entry_bar:
                    prev_low = low_arr[t - 1]
                    prev_high = high_arr[t - 1]
                    if position == 1:
                        potential_stop = prev_low * (1 - ts_pct / 100)
                        if potential_stop > sl_level:
                            sl_level = potential_stop
                    elif position == -1:
                        potential_stop = prev_high * (1 + ts_pct / 100)
                        if sl_level == 0.0 or potential_stop < sl_level:
                            sl_level = potential_stop
                
                if position == 1:
                    emerg_level = entry_price * (1 - sl_emergency_pct / 100)
                    if low_p <= emerg_level:
                        exit_signal = True
                        sl_exit_signal = True
                        sl_emergency_signal = True
                        exit_price = emerg_level
                elif position == -1:
                    emerg_level = entry_price * (1 + sl_emergency_pct / 100)
                    if high_p >= emerg_level:
                        exit_signal = True
                        sl_exit_signal = True
                        sl_emergency_signal = True
                        exit_price = emerg_level
                
                if not exit_signal and sl_level > 0:
                    if position == 1 and close_p < sl_level:
                        exit_signal = True
                        sl_exit_signal = True
                    elif position == -1 and close_p > sl_level:
                        exit_signal = True
                        sl_exit_signal = True
                
                if not exit_signal and div_exit == 1 and div_type > 0:
                    if position == 1 and div_bear_now:
                        exit_signal = True
                        div_exit_signal = True
                    elif position == -1 and div_bull_now:
                        exit_signal = True
                        div_exit_signal = True
                
                if not exit_signal and exit_mask > 0:
                    exit_count_bull = 0
                    exit_count_active = 0
                    
                    for bit in range(4):
                        if (exit_mask >> bit) & 1:
                            exit_count_active += 1
                            if (f_forming >> bit) & 1:
                                exit_count_bull += 1
                    
                    if position == 1 and exit_count_active > 0 and exit_count_bull == 0:
                        exit_signal = True
                        tf_exit_signal = True
                    elif position == -1 and exit_count_active > 0 and exit_count_bull == exit_count_active:
                        exit_signal = True
                        tf_exit_signal = True

                if not exit_signal:
                    if position == 1 and z_bear:
                        exit_signal = True
                        zone_exit_signal = True
                    elif position == -1 and z_bull:
                        exit_signal = True
                        zone_exit_signal = True
                
                if not exit_signal and cancel_tf == 1:
                    cancel_signal = False
                    ts_entry_i = timestamps_i64[entry_bar]
                    ts_now_i = timestamps_i64[t]
                    entry_day = ts_entry_i // 86400000
                    current_day = ts_now_i // 86400000
                    same_daily = (entry_day == current_day)
                    
                    eff = entry_filters_forming
                    f_now = filters_forming_arr[t]
                    # Fix fidelidad: usar resolved[t] (barra HTF actual finalizada)
                    # en vez de resolved[entry_bar] (barra HTF anterior a la entrada)
                    efr = filters_resolved_arr[t]

                    if (entry_mask >> 1) & 1:
                        entry_4h = (ts_entry_i // 3600000) // 4
                        now_4h = (ts_now_i // 3600000) // 4
                        if entry_4h == now_4h:
                            if ((eff >> 1) & 1) != ((f_now >> 1) & 1):
                                cancel_signal = True
                        else:
                            if ((eff >> 1) & 1) != ((efr >> 1) & 1):
                                cancel_signal = True

                    if not cancel_signal and (entry_mask >> 2) & 1:
                        if same_daily:
                            if ((eff >> 2) & 1) != ((f_now >> 2) & 1):
                                cancel_signal = True
                        else:
                            if ((eff >> 2) & 1) != ((efr >> 2) & 1):
                                cancel_signal = True
                
                if exit_signal or cancel_signal:
                    if position == 1:
                        trade_pnl = (exit_price - entry_price) / entry_price * 100
                    else:
                        trade_pnl = (entry_price - exit_price) / entry_price * 100
                    
                    # Restar comisión round-trip
                    trade_pnl -= commission_pct
                    
                    pnl += trade_pnl
                    trades += 1
                    if trade_pnl > 0:
                        wins += 1
                        gross_profit += trade_pnl
                    else:
                        gross_loss += abs(trade_pnl)
                    if cancel_signal:
                        cancels += 1
                    
                    if pnl > peak_pnl:
                        peak_pnl = pnl
                    dd = peak_pnl - pnl
                    if dd > max_dd:
                        max_dd = dd

                    # --- Cluster accounting (additive, does not affect global results) ---
                    if has_clusters:
                        cl_idx = cluster_labels[entry_bar]
                        if 0 <= cl_idx < n_clusters:
                            cl_pnl[cl_idx] += trade_pnl
                            cl_trades[cl_idx] += 1
                            if trade_pnl > 0:
                                cl_wins[cl_idx] += 1
                                cl_gp[cl_idx] += trade_pnl
                            else:
                                cl_gl[cl_idx] += abs(trade_pnl)
                            if cl_pnl[cl_idx] > cl_peak[cl_idx]:
                                cl_peak[cl_idx] = cl_pnl[cl_idx]
                            cl_dd = cl_peak[cl_idx] - cl_pnl[cl_idx]
                            if cl_dd > cl_maxdd[cl_idx]:
                                cl_maxdd[cl_idx] = cl_dd
                    # --- fin cluster accounting ---

                    # --- G1.1 Path γ 2026-04-29 Sesión 2 Frame 2 — per-trade tracking ---
                    # Solo si return_per_trade=True. Sentinel arrays (1,1) si flag=False
                    # → bounds check pt_idx < shape[1] = 1 falla en 2do trade y se omite
                    # silenciosamente (zero memory impact production callers).
                    # Granular enum: priority sl(emergency vs hit) > div > tf_exit > zone_exit
                    # > cancel (mutuamente exclusivos por construcción kernel current logic).
                    # Path γ SUSTITUYE Path α reduced enum Sesión 1B (decisión P1 Sesión 1).
                    if return_per_trade:
                        pt_idx = pt_count[c]
                        if pt_idx < pt_entry_bar.shape[1]:
                            pt_entry_bar[c, pt_idx] = entry_bar
                            pt_exit_bar[c, pt_idx] = t
                            pt_side[c, pt_idx] = 0 if position == 1 else 1
                            pt_pnl[c, pt_idx] = trade_pnl
                            if sl_exit_signal:
                                if sl_emergency_signal:
                                    pt_reason[c, pt_idx] = 1  # REASON_TF_SL_EMERGENCY
                                else:
                                    pt_reason[c, pt_idx] = 0  # REASON_TF_SL_HIT
                            elif div_exit_signal:
                                pt_reason[c, pt_idx] = 2  # REASON_TF_DIV_EXIT
                            elif tf_exit_signal:
                                pt_reason[c, pt_idx] = 3  # REASON_TF_TF_EXIT
                            elif zone_exit_signal:
                                pt_reason[c, pt_idx] = 4  # REASON_TF_ZONE_EXIT
                            else:
                                pt_reason[c, pt_idx] = 5  # REASON_TF_CANCEL_TF
                            if has_clusters:
                                cl_idx_pt = cluster_labels[entry_bar]
                                if 0 <= cl_idx_pt < n_clusters:
                                    pt_cluster[c, pt_idx] = cl_idx_pt
                                else:
                                    pt_cluster[c, pt_idx] = 0
                            else:
                                pt_cluster[c, pt_idx] = 0
                            # Path α' supplement preserved Path γ: entry_price siempre = close_p
                            # al abrir; exit_price = emerg_level si sl_emergency intrabar O close_p
                            # otherwise. Granular enum Path γ distingue sl_emergency vs sl_hit
                            # explícitamente (REASON_TF_SL_EMERGENCY=1 vs REASON_TF_SL_HIT=0); arrays
                            # entry_price + exit_price preservan precio exacto para audit refactor R6.
                            pt_entry_price[c, pt_idx] = entry_price
                            pt_exit_price[c, pt_idx] = exit_price
                            pt_count[c] = pt_idx + 1
                    # --- fin per-trade tracking ---

                    if position == 1:
                        div_ctx_bull = False
                    else:
                        div_ctx_bear = False

                    # Cooldown uniforme (convención Pine canónica i_cooldown_bars).
                    # Las 4 ramas del switch original (emergency/sl/div/cancel) colapsaban
                    # matemáticamente a `t` con cooldown_bars=1 (valor productivo único).
                    # Unificado 2026-04-22 tras confirmación Opción A: cooldown=1 siempre operacional
                    # en Pine histórico, la diferenciación por tipo exit era código muerto.
                    # Ver §13.3 "Cooldown asimétrico" RESUELTO + §13.4 entrada 2026-04-22.
                    if sl_emergency_signal or sl_exit_signal or div_exit_signal or cancel_signal:
                        cooldown_until = t + cooldown_bars - 1

                    position = 0
                    entry_price = 0.0
                    sl_level = 0.0
                    entry_filters_forming = 0
            
            if z_bear:
                div_ctx_bull = False
            if z_bull:
                div_ctx_bear = False
            
            if position == 0 and t >= acct_start:
                if t <= cooldown_until:
                    continue
                
                long_cond = False
                short_cond = False
                
                tf_entry_ok_bull = True
                tf_entry_ok_bear = True
                
                if (entry_mask >> 0) & 1:
                    if not ((f_forming >> 0) & 1): tf_entry_ok_bull = False
                    if ((f_forming >> 0) & 1): tf_entry_ok_bear = False
                if (entry_mask >> 1) & 1:
                    if not ((f_forming >> 1) & 1): tf_entry_ok_bull = False
                    if ((f_forming >> 1) & 1): tf_entry_ok_bear = False
                if (entry_mask >> 2) & 1:
                    if not ((f_forming >> 2) & 1): tf_entry_ok_bull = False
                    if ((f_forming >> 2) & 1): tf_entry_ok_bear = False
                if (entry_mask >> 3) & 1:
                    if not ((f_forming >> 3) & 1): tf_entry_ok_bull = False
                    if not ((f_forming >> 11) & 1): tf_entry_ok_bear = False
                if (entry_mask >> 4) & 1:
                    if not ((f_forming >> 4) & 1): tf_entry_ok_bull = False
                    if not ((f_forming >> 12) & 1): tf_entry_ok_bear = False
                
                # Pine-faithful: entries use div state from PREVIOUS bar
                # entry_div_ctx was snapshotted before current bar's div updated it
                # prev_div_bull/bear_now is the div_raw from previous bar
                effective_ctx_bull = entry_div_ctx_bull or prev_div_bull_now
                effective_ctx_bear = entry_div_ctx_bear or prev_div_bear_now
                
                if div_entry_mode == 0:
                    if z_bull:
                        if entry_tf_count > 0:
                            long_cond = tf_entry_ok_bull
                        else:
                            long_cond = True
                    if z_bear:
                        if entry_tf_count > 0:
                            short_cond = tf_entry_ok_bear
                        else:
                            short_cond = True
                
                elif div_entry_mode == 1:
                    if z_bull:
                        if entry_tf_count > 0:
                            long_cond = tf_entry_ok_bull and effective_ctx_bull
                        elif exit_tf_count > 0:
                            long_cond = effective_ctx_bull
                        else:
                            long_cond = prev_div_bull_now
                    
                    if z_bear:
                        if entry_tf_count > 0:
                            short_cond = tf_entry_ok_bear and effective_ctx_bear
                        elif exit_tf_count > 0:
                            short_cond = effective_ctx_bear
                        else:
                            short_cond = prev_div_bear_now
                
                elif div_entry_mode == 2:
                    if z_bull:
                        if entry_tf_count > 0:
                            long_cond = tf_entry_ok_bull or prev_div_bull_now
                        else:
                            long_cond = True
                    
                    if z_bear:
                        if entry_tf_count > 0:
                            short_cond = tf_entry_ok_bear or prev_div_bear_now
                        else:
                            short_cond = True
                
                if long_cond and exit_mask > 0:
                    exit_count_bull = 0
                    for bit in range(4):
                        if (exit_mask >> bit) & 1 and (f_forming >> bit) & 1:
                            exit_count_bull += 1
                    if exit_count_bull == 0:
                        long_cond = False
                
                if short_cond and exit_mask > 0:
                    exit_count_bull = 0
                    exit_count_active = 0
                    for bit in range(4):
                        if (exit_mask >> bit) & 1:
                            exit_count_active += 1
                            if (f_forming >> bit) & 1:
                                exit_count_bull += 1
                    if exit_count_bull == exit_count_active:
                        short_cond = False
                
                if long_cond:
                    position = 1
                    entry_price = close_p
                    entry_bar = t
                    entry_filters_forming = f_forming
                    sl_level = low_p * (1 - sl_pct / 100)
                elif short_cond:
                    position = -1
                    entry_price = close_p
                    entry_bar = t
                    entry_filters_forming = f_forming
                    sl_level = high_p * (1 + sl_pct / 100)
        
        results[c, 0] = pnl
        results[c, 1] = trades
        results[c, 2] = wins
        results[c, 3] = cancels
        results[c, 4] = max_dd
        results[c, 5] = gross_profit
        results[c, 6] = gross_loss

        # Store per-cluster results
        if has_clusters:
            for k in range(n_clusters):
                cl_pnl_out[c, k] = cl_pnl[k]
                cl_trades_out[c, k] = cl_trades[k]
                cl_wins_out[c, k] = cl_wins[k]
                cl_maxdd_out[c, k] = cl_maxdd[k]
                cl_gp_out[c, k] = cl_gp[k]
                cl_gl_out[c, k] = cl_gl[k]

    return results, cl_pnl_out, cl_trades_out, cl_wins_out, cl_maxdd_out, cl_gp_out, cl_gl_out

# ============================================
# SCORING v6.3 - COMISIONES + PF FUERTE
# ============================================
# Cambios vs v6.2:
#   1. PnL ya viene NET (comisiones restadas en engine)
#   2. PF multiplicador directo: PF^0.5
#      PF=1.0 → 1.0 (neutro), PF=1.5 → 1.22, PF=2.0 → 1.41
#      PF=3.0 → 1.73, PF=5.0 → 2.24
#      Con comisiones, PF<1 = sistema perdedor → score negativo por PnL<0
#   3. Activity sin cambios (log scale)
#   4. Resto igual
# ============================================

# Configuración del ranking v7.1
TRAIN_RATIO = 0.50
MIN_TRADES_TRAIN = 15
MIN_TRADES_TEST = 15
# MIN_PNL_ANNUAL_TEST: dinámico, se calcula según B&H del test
MIN_PF_TEST = 1.2               # 🆕 PF mínimo neto en test
MAX_DD_TEST = 25.0
TOP_TRAIN = 20000
TOP_FINAL = 100

W_TRAIN = 0.35
W_TEST = 0.40              # ⬆️ Test tiene más peso (out-of-sample importa más)
W_FULL = 0.25

def calc_score_v63(pnl, max_dd, gross_profit, gross_loss, trades, cancels, n_bars_period):
    """
    Score v6.3 - Con comisiones en PnL + PF multiplicador fuerte.
    
    PnL ya viene NET de comisiones (restadas en el engine).
    PF se calcula sobre gross_profit/gross_loss (también net de comisiones).
    """
    n = len(pnl)
    
    # Horas en el período → años
    hours = n_bars_period
    years = hours / 8760.0
    if years < 0.1:
        years = 0.1
    
    # 1. PnL ANUALIZADO (ya net de comisiones)
    pnl_annual = pnl / years
    
    # 2. DRAWDOWN FACTOR: 1 / (1 + MaxDD/10)
    dd_factor = 1.0 / (1.0 + max_dd / 10.0)
    
    # 3. PF MULTIPLICADOR FUERTE: sqrt(min(PF, 5))
    # PF=1 → 1.0, PF=1.5 → 1.22, PF=2 → 1.41, PF=3 → 1.73, PF=5 → 2.24
    # Con comisiones, PF refleja edge REAL después de costes
    profit_factor = np.where(gross_loss > 0, gross_profit / gross_loss, 
                             np.where(gross_profit > 0, 5.0, 0.0))
    pf_capped = np.minimum(profit_factor, 5.0)
    pf_mult = np.sqrt(pf_capped)
    
    # 4. ACTIVITY FACTOR (sin cambios vs v6.2)
    trades_per_year = trades / years
    log_norm = np.log(1.0 + 100.0 / 20.0)
    activity_factor = np.log(1.0 + trades_per_year / 20.0) / log_norm
    
    # 5. CANCEL FACTOR (sin cambios)
    cancel_rate = np.where(trades > 0, cancels / trades, 0.0)
    cancel_factor = 1.0 - cancel_rate * 0.3
    
    # SCORE FINAL = pnl_annual * dd_factor * pf_mult * activity * cancel
    # pf_mult ahora es multiplicador directo (no suavizado 0.5+0.5*x)
    score = pnl_annual * dd_factor * pf_mult * activity_factor * cancel_factor
    
    # PnL negativo → score negativo
    score = np.where(pnl > 0, score, -np.abs(pnl_annual))
    
    return score, pnl_annual, dd_factor, profit_factor, activity_factor, cancel_rate

def run_on_slice(configs, data, start_bar, end_bar, sl_pct, sl_emergency_pct, ts_pct, cooldown_bars, commission_pct, warmup=100, cluster_labels=None, n_clusters=1, return_per_trade=False, max_trades_per_config=5000):
    """Run simulation on a slice of precalculated data.

    warmup: number of bars before start_bar used to build state (div_ctx,
            last_zone_bull, etc.) without opening trades or accumulating stats.
            The kernel iterates from [actual_start .. end_bar) but only accounts
            trades from start_bar onwards, yielding exactly (end_bar - start_bar)
            operable bars.
    cluster_labels: int64 array of cluster IDs per bar (or None for no clustering).
    n_clusters: number of clusters (1 if no clustering).
    return_per_trade: if True, allocate per-trade arrays (entry_bar, exit_bar, side,
        pnl, reason, cluster) and append to return tuple. Si False (default): backward
        compat — returns same 7-element tuple del kernel original. G1.1 Path α
        2026-04-28 Sesión 1B (ROADMAP_PRE_RECICLAJE.md AGGRESSIVE pura recalibrada).
    max_trades_per_config: bound conservativo per-config trades count cuando
        return_per_trade=True. Default 5000 (cubrir kernels 3y con safe margin).
        Memory: n_configs × max_trades × 6 arrays × ~6 bytes = scope acotado audit/
        analyzer single specialist (NO walk-forward production sweep masivo).
    """
    n_data = len(data['close'])
    actual_start = max(0, start_bar - warmup)
    accounting_start = start_bar - actual_start  # offset within slice
    s, e = actual_start, end_bar
    ts_raw = data['timestamps'][s:e]
    ts_i64 = ts_raw.astype('datetime64[ms]').astype(np.int64)
    cl_labels_slice = cluster_labels[s:e] if cluster_labels is not None else np.zeros(e - s, dtype=np.int64)
    n_cl = n_clusters if cluster_labels is not None else 1

    if not return_per_trade:
        # Backward compat path: kwargs per-trade reciben sentinel arrays default,
        # writes condicionales no se ejecutan (return_per_trade=False default kernel).
        return run_simulation_numba(
            configs,
            data['close'][s:e], data['high'][s:e], data['low'][s:e],
            ts_i64,
            data['zone_bull'][s:e], data['zone_bear'][s:e],
            data['filters_forming'][s:e], data['filters_resolved'][s:e],
            data['div_bits'][s:e],
            sl_pct, sl_emergency_pct, ts_pct, cooldown_bars,
            commission_pct,
            accounting_start,
            cl_labels_slice,
            n_cl
        )

    # Path α return_per_trade=True: pre-allocate per-trade arrays + dispatch
    # Sesión 1B amendment Path α' supplement 2026-04-28: agregadas entry_price +
    # exit_price arrays para audit refactor Opción A clean (Sesión 1A.2 prereq).
    n_configs = len(configs)
    pt_entry_bar = np.zeros((n_configs, max_trades_per_config), dtype=np.int32)
    pt_exit_bar = np.zeros((n_configs, max_trades_per_config), dtype=np.int32)
    pt_side = np.zeros((n_configs, max_trades_per_config), dtype=np.int8)
    pt_pnl = np.zeros((n_configs, max_trades_per_config), dtype=np.float64)
    pt_reason = np.zeros((n_configs, max_trades_per_config), dtype=np.int8)
    pt_cluster = np.zeros((n_configs, max_trades_per_config), dtype=np.int8)
    pt_count = np.zeros(n_configs, dtype=np.int32)
    pt_entry_price = np.zeros((n_configs, max_trades_per_config), dtype=np.float64)
    pt_exit_price = np.zeros((n_configs, max_trades_per_config), dtype=np.float64)

    aggregates = run_simulation_numba(
        configs,
        data['close'][s:e], data['high'][s:e], data['low'][s:e],
        ts_i64,
        data['zone_bull'][s:e], data['zone_bear'][s:e],
        data['filters_forming'][s:e], data['filters_resolved'][s:e],
        data['div_bits'][s:e],
        sl_pct, sl_emergency_pct, ts_pct, cooldown_bars,
        commission_pct,
        accounting_start,
        cl_labels_slice,
        n_cl,
        True,  # return_per_trade=True
        pt_entry_bar, pt_exit_bar, pt_side, pt_pnl, pt_reason, pt_cluster, pt_count,
        pt_entry_price, pt_exit_price
    )

    # Note: arrays modificados in-place. Caller usa pt_count[c] para slice válido:
    #   trades_cfg_c = pt_entry_bar[c, :pt_count[c]]
    return aggregates, {
        "entry_bar": pt_entry_bar,
        "exit_bar": pt_exit_bar,
        "side": pt_side,  # 0=long, 1=short
        "pnl": pt_pnl,
        "reason": pt_reason,  # enum: 0=sl_exit, 1=div_exit, 2=normal_exit, 3=cancel_tf
        "cluster": pt_cluster,
        "count": pt_count,
        "entry_price": pt_entry_price,  # Sesión 1B amendment Path α' supplement
        "exit_price": pt_exit_price,    # Sesión 1B amendment Path α' supplement
        "max_trades_per_config": max_trades_per_config,
    }

def _load_cluster_labels(symbol, df):
    """Load pre-trained regime model and compute cluster labels for this symbol's data.

    Returns: (cluster_labels array or None, n_clusters int)
    """
    try:
        from regime_features import compute_regime_features
        import joblib as _joblib
    except ImportError as e:
        print(f"   [WARN] Cannot load regime model: {e}")
        return None, 1

    sym_key = symbol.replace("/USDT", "").replace("/", "")
    model_path = os.path.join("regime_models", f"{sym_key}_regime.joblib")
    if not os.path.exists(model_path):
        print(f"   [WARN] Regime model not found: {model_path}")
        return None, 1

    try:
        model_data = _joblib.load(model_path)
        gmm = model_data['gmm']
        scaler = model_data['scaler']
        lookback = model_data.get('lookback', 100)
        n_clusters = model_data['n_clusters']

        # Build DataFrame expected by compute_regime_features
        import pandas as _pd
        feat_df = _pd.DataFrame({
            'close': df['close'].values if hasattr(df['close'], 'values') else df['close'],
            'high': df['high'].values if hasattr(df['high'], 'values') else df['high'],
            'low': df['low'].values if hasattr(df['low'], 'values') else df['low'],
            'open': df['open'].values if hasattr(df['open'], 'values') else df['open'],
            'volume': df['volume'].values if hasattr(df['volume'], 'values') else df['volume'],
        })
        features, valid_mask = compute_regime_features(feat_df, lookback=lookback)

        n_bars = len(feat_df)
        labels = np.full(n_bars, -1, dtype=np.int64)
        valid_features = features[valid_mask]
        if len(valid_features) > 0:
            X = scaler.transform(valid_features)
            pred = gmm.predict(X)
            valid_indices = np.where(valid_mask)[0]
            for i, idx in enumerate(valid_indices):
                labels[idx] = pred[i]

        n_valid = int(np.sum(labels >= 0))
        print(f"   [RESEARCH] Regime model loaded: k={n_clusters}, {n_valid}/{n_bars} bars labeled")
        return labels, n_clusters

    except Exception as e:
        print(f"   [WARN] Error loading regime model: {e}")
        import traceback
        traceback.print_exc()
        return None, 1


def process_symbol(symbol):
    print(f"\n{'='*70}")
    print(f"[STATS] Procesando {symbol}")
    print(f"{'='*70}")
    
    df = fetch_all_candles(symbol, TIMEFRAME, TOTAL_CANDLES)
    if df is None or len(df) < 1000:
        print(f"   [ERROR] Datos insuficientes para {symbol}")
        return None
    
    zone_presets = SYMBOL_ZONE_PRESETS.get(symbol, [])
    if not zone_presets:
        print(f"   [WARN] Sin presets para {symbol}, saltando")
        return None
    
    n_variants = len(zone_presets) * len(HYST_VALUES)
    cluster_ids = PRESET_CLUSTER_IDS.get(symbol, [-1] * len(zone_presets))
    has_cluster_presets = any(cid >= 0 for cid in cluster_ids)

    # Load cluster labels if any preset is a cluster specialist
    cluster_labels = None
    n_clusters_actual = 1
    if has_cluster_presets:
        cluster_labels, n_clusters_actual = _load_cluster_labels(symbol, df)

    print(f"\n   [REFRESH] {len(zone_presets)} presets × {len(HYST_VALUES)} histéresis = {n_variants} variantes")

    variant_idx = 0
    for p_idx, preset in enumerate(zone_presets):
      cluster_id = cluster_ids[p_idx] if p_idx < len(cluster_ids) else -1
      for hyst_mult in HYST_VALUES:
        variant_idx += 1
        fast_type, fast_len = preset[0], preset[1]
        slow_type, slow_len = preset[4], preset[5]
        trend_type = preset[8]
        hyst_tag = f"H{hyst_mult:.1f}".replace(".", "")
        variant_label = f"{fast_type}({fast_len})/{slow_type}({slow_len})/{trend_type}_{hyst_tag}"

        print(f"\n   {'='*60}")
        print(f"   [REFRESH] Variante {variant_idx}/{n_variants}: {variant_label} (C{cluster_id})")
        print(f"   {'='*60}")

        try:
            result = _process_one_variant(df, symbol, preset, hyst_mult, variant_idx, variant_label, hyst_tag,
                                          cluster_id=cluster_id,
                                          cluster_labels=cluster_labels if cluster_id >= 0 else None,
                                          n_clusters=n_clusters_actual)
        except Exception as e:
            print(f"   [ERROR] Error en variante {variant_label}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return True


def _process_one_variant(df, symbol, preset, hyst_mult, variant_idx, variant_label, hyst_tag, cluster_id=-1, cluster_labels=None, n_clusters=1):
    """Ejecuta el flujo completo (train/test/full/ranking) para UNA variante de zonas."""

    # Resume: si el CSV de esta variante ya existe, saltar
    symbol_clean = symbol.replace("/", "")
    _check_csv = os.path.join(OUTPUT_DIR, f"full_{symbol_clean}_v{variant_idx:02d}_{hyst_tag}_C{cluster_id}.csv")
    # Fallback: check legacy name too
    _check_csv_legacy = os.path.join(OUTPUT_DIR, f"full_{symbol_clean}_v{variant_idx:02d}_{hyst_tag}.csv")
    if os.path.exists(_check_csv) or os.path.exists(_check_csv_legacy):
        print(f"   >> Saltando variante {variant_label} — CSV existe")
        return True

    data = precalculate_all_data(df, preset=preset, hyst_mult=hyst_mult, symbol=symbol)
    n_bars = len(data['close'])
    
    # Buy & Hold para referencia
    close_start = data['close'][100]
    close_end = data['close'][-1]
    bnh_pnl = (close_end - close_start) / close_start * 100
    bnh_years = (n_bars - 100) / 8760.0
    bnh_annual = bnh_pnl / bnh_years
    print(f"\n   [CHART] Buy & Hold: {bnh_pnl:+.1f}% total ({bnh_annual:+.1f}%/año)")
    
    configs = generate_valid_configs()
    print(f"   [LIST] {len(configs):,} configuraciones a evaluar")
    
    # SPLIT TEMPORAL
    split_bar = int(n_bars * TRAIN_RATIO)
    train_bars = split_bar
    test_bars = n_bars - split_bar
    
    train_start_ts = pd.Timestamp(df['timestamp'].iloc[0])
    split_ts = pd.Timestamp(df['timestamp'].iloc[split_bar])
    end_ts = pd.Timestamp(df['timestamp'].iloc[-1])
    
    # B&H para train y test por separado
    bnh_train = (data['close'][split_bar-1] - data['close'][100]) / data['close'][100] * 100
    bnh_test = (data['close'][-1] - data['close'][split_bar]) / data['close'][split_bar] * 100
    bnh_train_annual = bnh_train / (train_bars / 8760.0)
    bnh_test_annual = bnh_test / (test_bars / 8760.0)
    
    print(f"\n   [DATE] Split temporal:")
    print(f"      TRAIN: {train_bars} velas ({train_bars//24:.0f} días) | B&H: {bnh_train:+.1f}% ({bnh_train_annual:+.1f}%/año)")
    print(f"             {train_start_ts.strftime('%Y-%m-%d')} -> {split_ts.strftime('%Y-%m-%d')}")
    print(f"      TEST:  {test_bars} velas ({test_bars//24:.0f} días) | B&H: {bnh_test:+.1f}% ({bnh_test_annual:+.1f}%/año)")
    print(f"             {split_ts.strftime('%Y-%m-%d')} -> {end_ts.strftime('%Y-%m-%d')}")
    
    # ============================================
    # FASE 1: SIMULACIÓN TRAIN
    # ============================================
    print(f"\n   [START] Fase 1: Simulación TRAIN ({train_bars} velas)...")
    t0 = time.time()
    results_train, cl_pnl_tr, cl_trades_tr, cl_wins_tr, cl_maxdd_tr, cl_gp_tr, cl_gl_tr = \
        run_on_slice(configs, data, 0, split_bar,
                     SL_PERCENT, SL_EMERGENCY_PERCENT, TS_PERCENT, COOLDOWN_BARS,
                     COMMISSION_ROUND_TRIP,
                     cluster_labels=cluster_labels, n_clusters=n_clusters)
    t_train = time.time() - t0
    print(f"   [OK] Train completado en {t_train:.1f}s")

    pnl_tr = results_train[:, 0]
    trades_tr = results_train[:, 1]
    wins_tr = results_train[:, 2]
    cancels_tr = results_train[:, 3]
    maxdd_tr = results_train[:, 4]
    gp_tr = results_train[:, 5]
    gl_tr = results_train[:, 6]
    
    score_tr, pnl_ann_tr, dd_f_tr, pf_tr, act_f_tr, cr_tr = calc_score_v63(
        pnl_tr, maxdd_tr, gp_tr, gl_tr, trades_tr, cancels_tr, train_bars)
    wr_tr = np.where(trades_tr > 0, wins_tr / trades_tr * 100, 0.0)
    
    # Filtrar: trades >= MIN, PnL > 0, PnL anualizado > B&H del train
    valid_train = (trades_tr >= MIN_TRADES_TRAIN) & (pnl_tr > 0) & (pnl_ann_tr > 0)
    n_valid_train = int(np.sum(valid_train))
    print(f"   [STATS] Configs rentables con >= {MIN_TRADES_TRAIN} trades en train: {n_valid_train:,}")
    
    if n_valid_train == 0:
        print(f"   [WARN] Ninguna config supera filtros en train")
        return None
    
    # Top N del train — SELECCIÓN MIXTA: score + cupo PF
    # Garantiza que configs con PF alto no se pierdan aunque tengan score menor
    valid_indices_train = np.where(valid_train)[0]
    
    # Rama A: top por score (selección original)
    sorted_by_score = valid_indices_train[np.argsort(-score_tr[valid_indices_train])]
    
    # Rama B: top por PF (cupo para configs con mejor ratio ganancia/pérdida)
    pf_train = np.where(gl_tr[valid_indices_train] > 0, 
                        gp_tr[valid_indices_train] / gl_tr[valid_indices_train],
                        np.where(gp_tr[valid_indices_train] > 0, 5.0, 0.0))
    sorted_by_pf = valid_indices_train[np.argsort(-pf_train)]
    
    # Combinar: TOP_TRAIN por score, luego añadir cupo PF (sin duplicados)
    PF_CUPO = min(2000, TOP_TRAIN // 3)  # ~33% del espacio reservado para PF
    top_by_score = set(sorted_by_score[:TOP_TRAIN].tolist())
    pf_extras = []
    for idx in sorted_by_pf:
        if idx not in top_by_score:
            pf_extras.append(idx)
            if len(pf_extras) >= PF_CUPO:
                break
    
    top_train_indices = np.array(
        sorted_by_score[:TOP_TRAIN].tolist() + pf_extras, dtype=np.int64)
    
    n_pf_extras = len(pf_extras)
    print(f"   [TOP] Top {TOP_TRAIN} por score + {n_pf_extras} cupo PF = {len(top_train_indices)} para validación test")
    
    # ============================================
    # FASE 2: SIMULACIÓN TEST
    # ============================================
    print(f"\n   [START] Fase 2: Simulación TEST ({test_bars} velas, {len(top_train_indices)} configs)...")
    t0 = time.time()
    
    top_configs = configs[top_train_indices]
    results_test, cl_pnl_te_all, cl_trades_te_all, cl_wins_te_all, cl_maxdd_te_all, cl_gp_te_all, cl_gl_te_all = \
        run_on_slice(top_configs, data, split_bar, n_bars,
                     SL_PERCENT, SL_EMERGENCY_PERCENT, TS_PERCENT, COOLDOWN_BARS,
                     COMMISSION_ROUND_TRIP,
                     cluster_labels=cluster_labels, n_clusters=n_clusters)
    t_test = time.time() - t0
    print(f"   [OK] Test completado en {t_test:.1f}s")

    pnl_te = results_test[:, 0]
    trades_te = results_test[:, 1]
    wins_te = results_test[:, 2]
    cancels_te = results_test[:, 3]
    maxdd_te = results_test[:, 4]
    gp_te = results_test[:, 5]
    gl_te = results_test[:, 6]
    
    score_te, pnl_ann_te, dd_f_te, pf_te, act_f_te, cr_te = calc_score_v63(
        pnl_te, maxdd_te, gp_te, gl_te, trades_te, cancels_te, test_bars)
    wr_te = np.where(trades_te > 0, wins_te / trades_te * 100, 0.0)
    
    # ============================================
    # FASE 3: SIMULACIÓN FULL
    # ============================================
    print(f"\n   [START] Fase 3: Simulación FULL ({n_bars} velas, {len(top_train_indices)} configs)...")
    t0 = time.time()
    results_full, cl_pnl_fu_all, cl_trades_fu_all, cl_wins_fu_all, cl_maxdd_fu_all, cl_gp_fu_all, cl_gl_fu_all = \
        run_on_slice(top_configs, data, 0, n_bars,
                     SL_PERCENT, SL_EMERGENCY_PERCENT, TS_PERCENT, COOLDOWN_BARS,
                     COMMISSION_ROUND_TRIP,
                     cluster_labels=cluster_labels, n_clusters=n_clusters)
    t_full = time.time() - t0
    print(f"   [OK] Full completado en {t_full:.1f}s")

    pnl_fu = results_full[:, 0]
    trades_fu = results_full[:, 1]
    wins_fu = results_full[:, 2]
    cancels_fu = results_full[:, 3]
    maxdd_fu = results_full[:, 4]
    gp_fu = results_full[:, 5]
    gl_fu = results_full[:, 6]
    
    score_fu, pnl_ann_fu, dd_f_fu, pf_fu, act_f_fu, cr_fu = calc_score_v63(
        pnl_fu, maxdd_fu, gp_fu, gl_fu, trades_fu, cancels_fu, n_bars)
    wr_fu = np.where(trades_fu > 0, wins_fu / trades_fu * 100, 0.0)

    # --- Cluster specialist: swap scoring metrics to cluster-specific ---
    # Global metrics are preserved as-is in results arrays; we create cluster
    # metric variables that will be used for scoring and written to CSV.
    has_cl_metrics = cluster_id >= 0 and cluster_labels is not None and cluster_id < n_clusters
    if has_cl_metrics:
        k = cluster_id
        print(f"   [RESEARCH] Specialist C{k}: using cluster-specific metrics for scoring")

        # Train (all configs)
        cl_pnl_tr_k = cl_pnl_tr[:, k]
        cl_trades_tr_k = cl_trades_tr[:, k].astype(np.float64)
        cl_wins_tr_k = cl_wins_tr[:, k].astype(np.float64)
        cl_maxdd_tr_k = cl_maxdd_tr[:, k]
        cl_gp_tr_k = cl_gp_tr[:, k]
        cl_gl_tr_k = cl_gl_tr[:, k]
        cl_cancels_tr_k = np.zeros_like(cl_trades_tr_k)  # cancels not tracked per cluster
        cl_score_tr, cl_pnl_ann_tr, _, cl_pf_tr, _, _ = calc_score_v63(
            cl_pnl_tr_k, cl_maxdd_tr_k, cl_gp_tr_k, cl_gl_tr_k, cl_trades_tr_k, cl_cancels_tr_k, train_bars)
        cl_wr_tr = np.where(cl_trades_tr_k > 0, cl_wins_tr_k / cl_trades_tr_k * 100, 0.0)

        # Override train scoring for specialist selection
        score_tr = cl_score_tr
        pnl_ann_tr = cl_pnl_ann_tr

        # Recompute valid_train with cluster metrics
        valid_train = (cl_trades_tr_k >= MIN_TRADES_TRAIN) & (cl_pnl_tr_k > 0) & (cl_pnl_ann_tr > 0)
        n_valid_train_cl = int(np.sum(valid_train))
        print(f"   [RESEARCH] C{k} train: {n_valid_train_cl} configs rentables (cluster metrics)")

        # Rebuild top_train_indices with cluster scoring
        valid_indices_train = np.where(valid_train)[0]
        if len(valid_indices_train) > 0:
            sorted_by_score = valid_indices_train[np.argsort(-score_tr[valid_indices_train])]
            pf_train_cl = np.where(cl_gl_tr_k[valid_indices_train] > 0,
                                   cl_gp_tr_k[valid_indices_train] / cl_gl_tr_k[valid_indices_train],
                                   np.where(cl_gp_tr_k[valid_indices_train] > 0, 5.0, 0.0))
            sorted_by_pf = valid_indices_train[np.argsort(-pf_train_cl)]
            PF_CUPO = min(2000, TOP_TRAIN // 3)
            top_by_score = set(sorted_by_score[:TOP_TRAIN].tolist())
            pf_extras = []
            for idx in sorted_by_pf:
                if idx not in top_by_score:
                    pf_extras.append(idx)
                    if len(pf_extras) >= PF_CUPO:
                        break
            top_train_indices = np.array(
                sorted_by_score[:TOP_TRAIN].tolist() + pf_extras, dtype=np.int64)
            print(f"   [RESEARCH] C{k}: {len(top_train_indices)} configs for test phase")

            # Re-run test and full with new top_train_indices
            top_configs = configs[top_train_indices]

            results_test, cl_pnl_te_all, cl_trades_te_all, cl_wins_te_all, cl_maxdd_te_all, cl_gp_te_all, cl_gl_te_all = \
                run_on_slice(top_configs, data, split_bar, n_bars,
                             SL_PERCENT, SL_EMERGENCY_PERCENT, TS_PERCENT, COOLDOWN_BARS,
                             COMMISSION_ROUND_TRIP,
                             cluster_labels=cluster_labels, n_clusters=n_clusters)
            pnl_te = results_test[:, 0]
            trades_te = results_test[:, 1]
            wins_te = results_test[:, 2]
            cancels_te = results_test[:, 3]
            maxdd_te = results_test[:, 4]
            gp_te = results_test[:, 5]
            gl_te = results_test[:, 6]

            results_full, cl_pnl_fu_all, cl_trades_fu_all, cl_wins_fu_all, cl_maxdd_fu_all, cl_gp_fu_all, cl_gl_fu_all = \
                run_on_slice(top_configs, data, 0, n_bars,
                             SL_PERCENT, SL_EMERGENCY_PERCENT, TS_PERCENT, COOLDOWN_BARS,
                             COMMISSION_ROUND_TRIP,
                             cluster_labels=cluster_labels, n_clusters=n_clusters)
            pnl_fu = results_full[:, 0]
            trades_fu = results_full[:, 1]
            wins_fu = results_full[:, 2]
            cancels_fu = results_full[:, 3]
            maxdd_fu = results_full[:, 4]
            gp_fu = results_full[:, 5]
            gl_fu = results_full[:, 6]

            # Test cluster metrics
            cl_pnl_te_k = cl_pnl_te_all[:, k]
            cl_trades_te_k = cl_trades_te_all[:, k].astype(np.float64)
            cl_wins_te_k = cl_wins_te_all[:, k].astype(np.float64)
            cl_maxdd_te_k = cl_maxdd_te_all[:, k]
            cl_gp_te_k = cl_gp_te_all[:, k]
            cl_gl_te_k = cl_gl_te_all[:, k]
            cl_cancels_te_k = np.zeros_like(cl_trades_te_k)
            cl_score_te, cl_pnl_ann_te, _, cl_pf_te, _, _ = calc_score_v63(
                cl_pnl_te_k, cl_maxdd_te_k, cl_gp_te_k, cl_gl_te_k, cl_trades_te_k, cl_cancels_te_k, test_bars)
            cl_wr_te = np.where(cl_trades_te_k > 0, cl_wins_te_k / cl_trades_te_k * 100, 0.0)

            # Full cluster metrics
            cl_pnl_fu_k = cl_pnl_fu_all[:, k]
            cl_trades_fu_k = cl_trades_fu_all[:, k].astype(np.float64)
            cl_wins_fu_k = cl_wins_fu_all[:, k].astype(np.float64)
            cl_maxdd_fu_k = cl_maxdd_fu_all[:, k]
            cl_gp_fu_k = cl_gp_fu_all[:, k]
            cl_gl_fu_k = cl_gl_fu_all[:, k]
            cl_cancels_fu_k = np.zeros_like(cl_trades_fu_k)
            cl_score_fu, cl_pnl_ann_fu, _, cl_pf_fu, _, _ = calc_score_v63(
                cl_pnl_fu_k, cl_maxdd_fu_k, cl_gp_fu_k, cl_gl_fu_k, cl_trades_fu_k, cl_cancels_fu_k, n_bars)
            cl_wr_fu = np.where(cl_trades_fu_k > 0, cl_wins_fu_k / cl_trades_fu_k * 100, 0.0)

            # Override scoring for ranking
            score_te = cl_score_te
            pnl_ann_te = cl_pnl_ann_te
            pf_te = cl_pf_te
            wr_te = cl_wr_te
            score_fu = cl_score_fu
            pnl_ann_fu = cl_pnl_ann_fu
            pf_fu = cl_pf_fu
            wr_fu = cl_wr_fu
        else:
            print(f"   [WARN] C{k}: no configs pass cluster train filter")
            return None

    # ============================================
    # FASE 4: RANKING v6.1
    # ============================================
    print(f"\n   [STATS] Generando ranking v6.1...")
    
    # Filtros duros en test
    test_years = test_bars / 8760.0
    
    # MIN_PNL dinámico según mercado: más laxo si B&H bajista
    # B&H=-77% → min 8.5%, B&H=0% → min 20%, B&H=+120% → min 38%
    min_pnl_annual_test = max(5.0, 20.0 + bnh_test_annual * 0.15)
    
    valid_test = (
        (trades_te >= MIN_TRADES_TEST) & 
        (pnl_te > 0) &
        (pnl_ann_te >= min_pnl_annual_test) &
        (pf_te >= MIN_PF_TEST) &
        (maxdd_te <= MAX_DD_TEST)
    )
    n_valid_test = int(np.sum(valid_test))
    print(f"   [STATS] Filtros test: trades>={MIN_TRADES_TEST}, PnL/año>={min_pnl_annual_test:.1f}% (dinámico), PF>={MIN_PF_TEST}, MaxDD<={MAX_DD_TEST}%")
    print(f"   [STATS] Validadas en test: {n_valid_test} / {len(top_train_indices)}")
    
    if n_valid_test == 0:
        print(f"   [WARN] Ninguna config supera filtros duros — relajando PnL a >0 y PF a >1.1")
        valid_test = (trades_te >= MIN_TRADES_TEST) & (pnl_te > 0) & (pf_te >= 1.1)
        n_valid_test = int(np.sum(valid_test))
        print(f"   [STATS] Configs con filtros relajados: {n_valid_test}")
    
    if n_valid_test == 0:
        print(f"   [WARN] Ninguna config rentable en test")
        combined_score = score_tr[top_train_indices]
        valid_combined = np.ones(len(top_train_indices), dtype=bool)
        robustness = np.zeros(len(top_train_indices))
    else:
        s_tr = score_tr[top_train_indices]
        s_te = score_te
        s_fu = score_fu
        
        # Robustness: PnL anualizado test vs train
        robustness = np.where(
            pnl_ann_tr[top_train_indices] > 0,
            np.minimum(pnl_ann_te / pnl_ann_tr[top_train_indices], 1.5),
            0.0
        )
        
        # Score combinado
        raw_combined = W_TRAIN * s_tr + W_TEST * s_te + W_FULL * s_fu
        
        # Consistency: ambos rentables → 1.2, solo train → 0.3
        both_profitable = (pnl_tr[top_train_indices] > 0) & (pnl_te > 0)
        consistency = np.where(both_profitable, 1.2, 0.3)
        
        # Bonus por superar B&H en test
        beats_bnh_test = np.where(pnl_ann_te > bnh_test_annual, 1.3, 1.0)
        beats_bnh_train = np.where(pnl_ann_tr[top_train_indices] > bnh_train_annual, 1.2, 1.0)
        
        combined_score = raw_combined * np.sqrt(np.maximum(robustness, 0.01)) * consistency * beats_bnh_test * beats_bnh_train
        
        valid_combined = valid_test
    
    # Ranking final
    valid_final_indices = np.where(valid_combined)[0]
    if len(valid_final_indices) == 0:
        valid_final_indices = np.arange(len(top_train_indices))
    
    sorted_final = valid_final_indices[np.argsort(-combined_score[valid_final_indices])]
    
    # ============================================
    # DEDUP: Eliminar configs que solo difieren en cancel_tf
    # Si cancel produce 0 cancels, cancel_tf=OFF y cancel_tf=TF dan 
    # resultados idénticos. Mantener solo la versión cancel_tf=OFF.
    # ============================================
    seen_base_configs = set()
    deduped_final = []
    for local_idx in sorted_final:
        global_idx = top_train_indices[local_idx]
        cfg = int(configs[global_idx])
        # Base config = config sin el bit 22 (cancel_tf)
        base_cfg = cfg & ~(1 << 22)
        cancel_tf_bit = (cfg >> 22) & 1
        
        if base_cfg in seen_base_configs:
            # Ya tenemos esta config (la versión sin cancel o con cancel)
            # Solo skipear si ambos tienen 0 cancels (son idénticos)
            if cancels_fu[local_idx] == 0:
                continue
        seen_base_configs.add(base_cfg)
        deduped_final.append(local_idx)
        if len(deduped_final) >= TOP_FINAL:
            break
    
    n_deduped = len(sorted_final) - len(deduped_final) 
    if n_deduped > 0:
        print(f"   [REFRESH] Dedup: {n_deduped} configs duplicadas eliminadas (cancel_tf sin efecto)")
    top_final = deduped_final
    
    # ============================================
    # CSV EXHAUSTIVO: TODAS las configs top_train sin filtros
    # ============================================
    symbol_clean = symbol.replace("/", "")
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    train_years = train_bars / 8760.0
    full_years = n_bars / 8760.0
    
    csv_file = f"{output_dir}/full_{symbol_clean}_v{variant_idx:02d}_{hyst_tag}_C{cluster_id}.csv"

    print(f"   [EDIT] Escribiendo CSV con {len(top_train_indices)} configs...")

    with open(csv_file, 'w', encoding='utf-8') as f:
        header = ("config_id,entry_tfs,n_entry_tfs,exit_tfs,cancel,sl_type,"
                  "div_entry,div_exit,div_type,variant,indicators,"
                  "pnl_tr,pnl_ann_tr,trades_tr,tpy_tr,wr_tr,maxdd_tr,pf_tr,score_tr,"
                  "pnl_te,pnl_ann_te,trades_te,tpy_te,wr_te,maxdd_te,pf_te,score_te,"
                  "pnl_fu,pnl_ann_fu,trades_fu,tpy_fu,wr_fu,maxdd_fu,pf_fu,score_fu,"
                  "robustness,combined_score,cancels_tr,cancels_te,cancels_fu,"
                  "beats_bnh_tr,beats_bnh_te,beats_bnh_fu,cluster_id")
        if has_cl_metrics:
            header += (",pnl_cl_tr,trades_cl_tr,pf_cl_tr,maxdd_cl_tr,wr_cl_tr"
                       ",pnl_cl_te,trades_cl_te,pf_cl_te,maxdd_cl_te,wr_cl_te"
                       ",pnl_cl_fu,trades_cl_fu,pf_cl_fu,maxdd_cl_fu,wr_cl_fu")
        f.write(header + "\n")
        
        for i in range(len(top_train_indices)):
            gi = top_train_indices[i]
            cfg = int(configs[gi])
            decoded = decode_config(cfg)
            
            entry_str = '+'.join(decoded['entry_tfs']) if decoded['entry_tfs'] else 'ZONA'
            exit_str = '+'.join(decoded['exit_tfs'])
            inds_str = '+'.join(decoded['div_indicators']) if decoded['div_indicators'] else '-'
            
            tpy_tr_v = trades_tr[gi] / train_years if train_years > 0 else 0
            tpy_te_v = trades_te[i] / test_years if test_years > 0 else 0
            tpy_fu_v = trades_fu[i] / full_years if full_years > 0 else 0
            
            b_tr = 1 if pnl_ann_tr[gi] > bnh_train_annual else 0
            b_te = 1 if pnl_ann_te[i] > bnh_test_annual else 0
            b_fu = 1 if pnl_ann_fu[i] > bnh_annual else 0
            
            rob_val = robustness[i] if i < len(robustness) else 0.0
            cscore_val = combined_score[i] if i < len(combined_score) else 0.0
            
            f.write(f"{cfg},{entry_str},{decoded['n_entry_tfs']},{exit_str},"
                    f"{decoded['cancel_str']},{decoded['ts_str']},"
                    f"{decoded['div_entry_mode']},{decoded['div_exit']},{decoded['div_type']},"
                    f"{decoded['variant']},{inds_str},"
                    f"{pnl_tr[gi]:.2f},{pnl_ann_tr[gi]:.2f},{int(trades_tr[gi])},{tpy_tr_v:.1f},"
                    f"{wr_tr[gi]:.1f},{maxdd_tr[gi]:.2f},{pf_tr[gi]:.3f},{score_tr[gi]:.3f},"
                    f"{pnl_te[i]:.2f},{pnl_ann_te[i]:.2f},{int(trades_te[i])},{tpy_te_v:.1f},"
                    f"{wr_te[i]:.1f},{maxdd_te[i]:.2f},{pf_te[i]:.3f},{score_te[i]:.3f},"
                    f"{pnl_fu[i]:.2f},{pnl_ann_fu[i]:.2f},{int(trades_fu[i])},{tpy_fu_v:.1f},"
                    f"{wr_fu[i]:.1f},{maxdd_fu[i]:.2f},{pf_fu[i]:.3f},{score_fu[i]:.3f},"
                    f"{rob_val:.3f},{cscore_val:.3f},"
                    f"{int(cancels_tr[gi])},{int(cancels_te[i])},{int(cancels_fu[i])},"
                    f"{b_tr},{b_te},{b_fu},{cluster_id}")
            if has_cl_metrics:
                k = cluster_id
                # Train cluster (gi indexes all configs)
                cl_pf_tr_i = cl_gp_tr[gi, k] / cl_gl_tr[gi, k] if cl_gl_tr[gi, k] > 0 else (5.0 if cl_gp_tr[gi, k] > 0 else 0.0)
                cl_wr_tr_i = cl_wins_tr[gi, k] / cl_trades_tr[gi, k] * 100 if cl_trades_tr[gi, k] > 0 else 0.0
                # Test cluster (i indexes top_train subset)
                cl_pf_te_i = cl_gp_te_all[i, k] / cl_gl_te_all[i, k] if cl_gl_te_all[i, k] > 0 else (5.0 if cl_gp_te_all[i, k] > 0 else 0.0)
                cl_wr_te_i = cl_wins_te_all[i, k] / cl_trades_te_all[i, k] * 100 if cl_trades_te_all[i, k] > 0 else 0.0
                # Full cluster
                cl_pf_fu_i = cl_gp_fu_all[i, k] / cl_gl_fu_all[i, k] if cl_gl_fu_all[i, k] > 0 else (5.0 if cl_gp_fu_all[i, k] > 0 else 0.0)
                cl_wr_fu_i = cl_wins_fu_all[i, k] / cl_trades_fu_all[i, k] * 100 if cl_trades_fu_all[i, k] > 0 else 0.0
                f.write(f",{cl_pnl_tr[gi, k]:.2f},{int(cl_trades_tr[gi, k])},{cl_pf_tr_i:.3f},{cl_maxdd_tr[gi, k]:.2f},{cl_wr_tr_i:.1f}"
                        f",{cl_pnl_te_all[i, k]:.2f},{int(cl_trades_te_all[i, k])},{cl_pf_te_i:.3f},{cl_maxdd_te_all[i, k]:.2f},{cl_wr_te_i:.1f}"
                        f",{cl_pnl_fu_all[i, k]:.2f},{int(cl_trades_fu_all[i, k])},{cl_pf_fu_i:.3f},{cl_maxdd_fu_all[i, k]:.2f},{cl_wr_fu_i:.1f}")
            f.write("\n")
    
    print(f"   [SAVE] CSV exhaustivo guardado: {csv_file}")
    
    # ============================================
    # ESCRIBIR RANKING
    # ============================================
    total_elapsed = t_train + t_test + t_full
    ranking_file = f"{output_dir}/ranking_{symbol_clean}_v{variant_idx:02d}_{hyst_tag}_C{cluster_id}.txt"
    with open(ranking_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*100}\n")
        f.write(f"RANKING {symbol} — {variant_label} - Lab v8.3 - MULTI-PRESET (10k/50-50)\n")
        f.write(f"Score = PnL_net/año * DD_factor * PF^0.5 * Activity_log * CancelFactor\n")
        f.write(f"Filtros test: PF>={MIN_PF_TEST}, PnL dinámico, MaxDD<={MAX_DD_TEST}%, trades>={MIN_TRADES_TEST}\n")
        f.write(f"Comisiones: {COMMISSION_ROUND_TRIP}% round-trip por trade\n")
        f.write(f"Activity = log(1+tpy/20)/log(6) | Dedup cancel_tf | TOP_TRAIN=20000 + cupo PF\n")
        f.write(f"{'='*100}\n\n")
        f.write(f"Total velas: {n_bars} ({n_bars//24:.0f} días, {n_bars/8760:.1f} años)\n")
        f.write(f"Buy & Hold: {bnh_pnl:+.1f}% total ({bnh_annual:+.1f}%/año)\n")
        f.write(f"Split: TRAIN {train_bars} velas ({train_bars//24:.0f}d) | TEST {test_bars} velas ({test_bars//24:.0f}d)\n")
        f.write(f"B&H Train: {bnh_train:+.1f}% ({bnh_train_annual:+.1f}%/año) | B&H Test: {bnh_test:+.1f}% ({bnh_test_annual:+.1f}%/año)\n")
        f.write(f"Período TRAIN: {train_start_ts.strftime('%Y-%m-%d')} -> {split_ts.strftime('%Y-%m-%d')}\n")
        f.write(f"Período TEST:  {split_ts.strftime('%Y-%m-%d')} -> {end_ts.strftime('%Y-%m-%d')}\n")
        f.write(f"Configs evaluadas: {len(configs):,}\n")
        f.write(f"Rentables en train (>= {MIN_TRADES_TRAIN} trades): {n_valid_train:,}\n")
        f.write(f"Validadas en test: {n_valid_test}\n")
        f.write(f"MIN_TRADES: train={MIN_TRADES_TRAIN} test={MIN_TRADES_TEST}\n")
        f.write(f"MIN_PF_TEST: {MIN_PF_TEST}\n")
        f.write(f"MIN_PNL_TEST (dinámico): {min_pnl_annual_test:.1f}%/año (base 20 + B&H*0.15)\n")
        f.write(f"Commission: {COMMISSION_ROUND_TRIP}% round-trip\n")
        f.write(f"Tiempo total: {total_elapsed:.1f}s\n\n")
        
        f.write(f"{'='*100}\n")
        f.write(f"TOP {len(top_final)} - RANKING v7.1 (por Score combinado)\n")
        f.write(f"{'='*100}\n\n")
        
        for rank, local_idx in enumerate(top_final, 1):
            global_idx = top_train_indices[local_idx]
            cfg = configs[global_idx]
            decoded = decode_config(cfg)
            
            full_years = n_bars / 8760.0
            train_years = train_bars / 8760.0
            test_years = test_bars / 8760.0
            
            rob_val = robustness[local_idx]
            
            # Trades por año
            tpy_tr = trades_tr[global_idx] / train_years
            tpy_te = trades_te[local_idx] / test_years
            tpy_fu = trades_fu[local_idx] / full_years
            
            f.write(f"#{rank:3d} | Config: {cfg} | CombScore: {combined_score[local_idx]:.1f} | Robust: {rob_val:.2f}\n")
            f.write(f"      Entry: {'+'.join(decoded['entry_tfs']) if decoded['entry_tfs'] else 'ZONA'} ({decoded['n_entry_tfs']} TFs)")
            f.write(f" | Exit: {'+'.join(decoded['exit_tfs'])}")
            f.write(f" | Cancel: {decoded['cancel_str']} | SL: {decoded['ts_str']}\n")
            f.write(f"      Div: Entry={decoded['div_entry_mode']}, Exit={decoded['div_exit']}, Type={decoded['div_type']}")
            f.write(f" | Var: {decoded['variant']} (Reg:{decoded['reg_str']}, Hid:{decoded['hid_str']})")
            if decoded['div_indicators']:
                f.write(f" | Inds: {'+'.join(decoded['div_indicators'])}")
            f.write(f"\n")
            
            f.write(f"      --- TRAIN ({train_bars//24}d) -----------------------------------------------\n")
            f.write(f"      PnL: {pnl_tr[global_idx]:+.1f}% ({pnl_ann_tr[global_idx]:+.1f}%/año)")
            f.write(f" | Trades: {int(trades_tr[global_idx])} ({tpy_tr:.0f}/año)")
            f.write(f" | WR: {wr_tr[global_idx]:.0f}%")
            f.write(f" | MaxDD: {maxdd_tr[global_idx]:.1f}%\n")
            f.write(f"      PF: {pf_tr[global_idx]:.2f}")
            f.write(f" | Activity: {act_f_tr[global_idx]:.2f}")
            f.write(f" | Cancels: {int(cancels_tr[global_idx])} ({cr_tr[global_idx]*100:.0f}%)")
            f.write(f" | Score: {score_tr[global_idx]:.1f}")
            beats_tr = "V SUPERA B&H" if pnl_ann_tr[global_idx] > bnh_train_annual else "X NO supera B&H"
            f.write(f" | {beats_tr}\n")
            
            f.write(f"      --- TEST ({test_bars//24}d) ------------------------------------------------\n")
            f.write(f"      PnL: {pnl_te[local_idx]:+.1f}% ({pnl_ann_te[local_idx]:+.1f}%/año)")
            f.write(f" | Trades: {int(trades_te[local_idx])} ({tpy_te:.0f}/año)")
            f.write(f" | WR: {wr_te[local_idx]:.0f}%")
            f.write(f" | MaxDD: {maxdd_te[local_idx]:.1f}%\n")
            f.write(f"      PF: {pf_te[local_idx]:.2f}")
            f.write(f" | Activity: {act_f_te[local_idx]:.2f}")
            f.write(f" | Cancels: {int(cancels_te[local_idx])} ({cr_te[local_idx]*100:.0f}%)")
            f.write(f" | Score: {score_te[local_idx]:.1f}")
            beats_te = "V SUPERA B&H" if pnl_ann_te[local_idx] > bnh_test_annual else "X NO supera B&H"
            f.write(f" | {beats_te}\n")
            
            f.write(f"      --- FULL ({n_bars//24}d) -------------------------------------------------\n")
            f.write(f"      PnL: {pnl_fu[local_idx]:+.1f}% ({pnl_ann_fu[local_idx]:+.1f}%/año)")
            f.write(f" | Trades: {int(trades_fu[local_idx])} ({tpy_fu:.0f}/año)")
            f.write(f" | WR: {wr_fu[local_idx]:.0f}%")
            f.write(f" | MaxDD: {maxdd_fu[local_idx]:.1f}%\n")
            f.write(f"      PF: {pf_fu[local_idx]:.2f}")
            f.write(f" | Activity: {act_f_fu[local_idx]:.2f}")
            f.write(f" | Cancels: {int(cancels_fu[local_idx])} ({cr_fu[local_idx]*100:.0f}%)")
            beats_fu = "V SUPERA B&H" if pnl_ann_fu[local_idx] > bnh_annual else "X NO supera B&H"
            f.write(f" | {beats_fu}\n")
            f.write(f"\n")
        
        # RANKING DUAL: TOP por PF BRUTO (mismos filtros)
        # ============================================
        # Reordenar las configs validadas por PF full descendente (en vez de score)
        pf_full_bruto = np.where(gl_fu > 0, gp_fu / gl_fu, 
                                  np.where(gp_fu > 0, 5.0, 0.0))
        
        # Usar mismas configs que pasaron filtros de test
        valid_for_pf = np.where(valid_combined)[0]
        if len(valid_for_pf) > 0:
            sorted_by_pf_full = valid_for_pf[np.argsort(-pf_full_bruto[valid_for_pf])]
            
            # Dedup cancel_tf también aquí
            seen_base_pf = set()
            top_pf_ranking = []
            for local_idx in sorted_by_pf_full:
                global_idx = top_train_indices[local_idx]
                cfg = int(configs[global_idx])
                base_cfg = cfg & ~(1 << 22)
                if base_cfg in seen_base_pf:
                    if cancels_fu[local_idx] == 0:
                        continue
                seen_base_pf.add(base_cfg)
                top_pf_ranking.append(local_idx)
                if len(top_pf_ranking) >= 20:
                    break
            
            f.write(f"\n{'='*100}\n")
            f.write(f"RANKING ALTERNATIVO: TOP {len(top_pf_ranking)} POR PF BRUTO (FULL)\n")
            f.write(f"Mismos filtros de test, ordenado por Profit Factor en período completo\n")
            f.write(f"{'='*100}\n\n")
            
            for pf_rank, local_idx in enumerate(top_pf_ranking, 1):
                global_idx = top_train_indices[local_idx]
                cfg = configs[global_idx]
                decoded = decode_config(cfg)
                
                pf_b = pf_full_bruto[local_idx]
                entry_str = '+'.join(decoded['entry_tfs']) if decoded['entry_tfs'] else 'ZONA'
                exit_str = '+'.join(decoded['exit_tfs'])
                div_str = f"Entry={decoded['div_entry_mode']}, Exit={decoded['div_exit']}, Type={decoded['div_type']}"
                var_str = f"Var={decoded['variant']}"
                inds_str = '+'.join(decoded['div_indicators']) if decoded['div_indicators'] else '-'
                
                full_years = n_bars / 8760.0
                tpy_fu2 = trades_fu[local_idx] / full_years
                
                # Check if this config also appears in the main ranking
                main_rank = None
                for mr, ml in enumerate(top_final, 1):
                    if ml == local_idx:
                        main_rank = mr
                        break
                rank_note = f" (= #{main_rank} en ranking principal)" if main_rank else " (NO en ranking principal)"
                
                f.write(f"PF#{pf_rank:3d} | Config: {cfg} | PF Bruto: {pf_b:.2f}{rank_note}\n")
                f.write(f"      {entry_str} | {exit_str} | {decoded['cancel_str']} | {decoded['ts_str']}\n")
                f.write(f"      {div_str} | {var_str} | Inds: {inds_str}\n")
                f.write(f"      FULL: PnL={pnl_fu[local_idx]:+.1f}% ({pnl_ann_fu[local_idx]:+.1f}%/año)")
                f.write(f" | {int(trades_fu[local_idx])}t ({tpy_fu2:.0f}/yr)")
                f.write(f" | WR={wr_fu[local_idx]:.0f}%")
                f.write(f" | MaxDD={maxdd_fu[local_idx]:.1f}%")
                f.write(f" | PF neto={pf_fu[local_idx]:.2f}")
                f.write(f" | Robust={robustness[local_idx]:.2f}\n\n")
    
    print(f"   [SAVE] Ranking guardado: {ranking_file}")
    
    # Resumen en consola
    print(f"\n   [CHART] B&H referencia: Train={bnh_train_annual:+.0f}%/año | Test={bnh_test_annual:+.0f}%/año | Full={bnh_annual:+.0f}%/año")
    print(f"\n   [TOP] TOP 5:")
    for rank, local_idx in enumerate(top_final[:5], 1):
        global_idx = top_train_indices[local_idx]
        cfg = configs[global_idx]
        decoded = decode_config(cfg)
        
        ann_tr = pnl_ann_tr[global_idx]
        ann_te = pnl_ann_te[local_idx]
        ann_fu = pnl_ann_fu[local_idx]
        tr_count = int(trades_fu[local_idx])
        tpy = tr_count / (n_bars / 8760.0)
        
        entry_str = '+'.join(decoded['entry_tfs']) if decoded['entry_tfs'] else 'ZONA'
        consistent = "V" if pnl_tr[global_idx] > 0 and pnl_te[local_idx] > 0 else "X"
        
        print(f"      #{rank} {entry_str:<15s} | Train={ann_tr:+.0f}%/yr Test={ann_te:+.0f}%/yr Full={ann_fu:+.0f}%/yr"
              f" | {tr_count}t ({tpy:.0f}/yr) | Score={combined_score[local_idx]:.0f} [{consistent}]")
    
    return True

def main():
    print("="*70)
    print("[LAB] LABORATORIO HISTÓRICO v8.3 - MULTI-PRESET + HISTÉRESIS (10k/50-50)")
    print(f"   Comisiones: {COMMISSION_ROUND_TRIP}% round-trip por trade")
    print("   Score = PnL_net/año * DD_factor * PF^0.5 * Activity(log) * CancelFactor")
    print(f"   Presets: top 5 + agresivo por Lab LITE v5c")
    print(f"   Histéresis: {HYST_VALUES}")
    print(f"   Split: 10k velas, 50/50 train/test (5k/5k)")
    print(f"   Combinatoria: ~19.6M configs × N variantes por símbolo")
    print("="*70)
    
    successful = []
    failed = []
    
    for i, symbol in enumerate(SYMBOLS, 1):
        print(f"\n[{i}/{len(SYMBOLS)}] ", end="")
        try:
            result = process_symbol(symbol)
            if result:
                successful.append((symbol, "OK"))
            else:
                failed.append((symbol, "Sin configs válidas"))
        except KeyboardInterrupt:
            print("\n\n[STOP] Detenido por usuario")
            break
        except Exception as e:
            print(f"\n[ERROR] Error procesando {symbol}: {e}")
            failed.append((symbol, str(e)))
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print("[STATS] RESUMEN FINAL")
    print(f"{'='*70}")
    print(f"   [OK] Procesados: {len(successful)}")
    print(f"   [ERROR] Fallidos: {len(failed)}")
    
    if failed:
        print(f"\n   Errores:")
        for sym, err in failed:
            print(f"      - {sym}: {err[:50]}...")
    
    print(f"\n{'='*70}")
    print("[OK] COMPLETADO")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
