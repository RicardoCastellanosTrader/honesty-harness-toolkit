#!/usr/bin/env python3
"""
Download maximum available 1h OHLCV history from Binance for all symbols.
Extends existing parquet files backwards and forwards.

[honesty-harness-toolkit — adaptación T2 Sesión 2]: se añade CLI con
`--symbols` (default: BTC/USDT, un solo símbolo) y `--all` (los 45 del
estudio). download_full() y _fetch_range() son copia byte-idéntica del
original (combolab/download_full_history.py).
"""

import os
import time
import argparse
import ccxt
import pandas as pd

CACHE_DIR = "data_cache"
TIMEFRAME = "1h"
INTERVAL_MS = 3600000  # 1 hour in ms

# Start from earliest possible date (Binance launched ~2017-07)
GLOBAL_SINCE_MS = int(pd.Timestamp("2017-01-01").timestamp() * 1000)

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


def cache_path(symbol):
    sc = symbol.replace("/", "")
    return os.path.join(CACHE_DIR, f"{sc}_{TIMEFRAME}.parquet")


def download_full(symbol, exchange):
    """Download full history for a symbol, extending existing cache."""
    path = cache_path(symbol)
    existing_df = None

    if os.path.exists(path):
        existing_df = pd.read_parquet(path)
        earliest_existing = existing_df['timestamp_ms'].min()
        latest_existing = existing_df['timestamp_ms'].max()
        print(f"  Existing: {len(existing_df)} bars, "
              f"{pd.to_datetime(earliest_existing, unit='ms')} -> "
              f"{pd.to_datetime(latest_existing, unit='ms')}")
    else:
        print(f"  No existing cache")

    # Phase 1: Download backwards from earliest existing (or from now if no cache)
    backward_candles = []
    if existing_df is not None:
        # Download from global start to earliest existing
        since = GLOBAL_SINCE_MS
        end_before = int(earliest_existing)
        if since < end_before:
            print(f"  Phase 1: Downloading backwards from {pd.to_datetime(end_before, unit='ms')}...")
            backward_candles = _fetch_range(exchange, symbol, since, end_before)
            print(f"  Phase 1: Got {len(backward_candles)} backward candles")
    else:
        # No existing data - download everything from the start
        since = GLOBAL_SINCE_MS
        print(f"  Phase 1: Downloading full history from {pd.to_datetime(since, unit='ms')}...")

    # Phase 2: Download forwards from latest existing to now
    forward_candles = []
    now_ms = int(time.time() * 1000)
    if existing_df is not None:
        since_forward = int(latest_existing) + INTERVAL_MS
        if since_forward < now_ms:
            print(f"  Phase 2: Downloading forward from {pd.to_datetime(since_forward, unit='ms')}...")
            forward_candles = _fetch_range(exchange, symbol, since_forward, now_ms)
            print(f"  Phase 2: Got {len(forward_candles)} forward candles")

    # If no existing data, download everything in one go
    if existing_df is None:
        all_candles = _fetch_range(exchange, symbol, since, now_ms)
        if not all_candles:
            print(f"  ERROR: No data available for {symbol}")
            return 0
        df = pd.DataFrame(all_candles, columns=['timestamp_ms', 'open', 'high', 'low', 'close', 'volume'])
    else:
        # Combine all data
        parts = []
        if backward_candles:
            parts.append(pd.DataFrame(backward_candles, columns=['timestamp_ms', 'open', 'high', 'low', 'close', 'volume']))
        parts.append(existing_df)
        if forward_candles:
            parts.append(pd.DataFrame(forward_candles, columns=['timestamp_ms', 'open', 'high', 'low', 'close', 'volume']))
        df = pd.concat(parts, ignore_index=True)

    df = df.drop_duplicates(subset=['timestamp_ms']).sort_values('timestamp_ms').reset_index(drop=True)
    df.to_parquet(path, index=False)

    ts_min = pd.to_datetime(df['timestamp_ms'].min(), unit='ms')
    ts_max = pd.to_datetime(df['timestamp_ms'].max(), unit='ms')
    print(f"  SAVED: {len(df)} bars, {ts_min} -> {ts_max}")
    return len(df)


def _fetch_range(exchange, symbol, since_ms, until_ms):
    """Fetch all candles in a time range."""
    candles = []
    current = since_ms
    consecutive_errors = 0

    while current < until_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=current, limit=1000)
            if not ohlcv:
                break
            # Filter to range
            ohlcv = [c for c in ohlcv if c[0] < until_ms]
            if not ohlcv:
                break
            candles.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            # Progress every 10K candles
            if len(candles) % 10000 < 1000:
                print(f"    ... {len(candles)} candles so far "
                      f"(at {pd.to_datetime(last_ts, unit='ms')})")
            if last_ts >= until_ms - INTERVAL_MS:
                break
            current = last_ts + INTERVAL_MS
            consecutive_errors = 0
            time.sleep(0.05)  # Rate limit
        except Exception as e:
            consecutive_errors += 1
            if consecutive_errors >= 5:
                print(f"    ERROR after 5 retries: {e}")
                break
            print(f"    Retry {consecutive_errors}/5: {e}")
            time.sleep(2 * consecutive_errors)

    return candles


def main(symbols=None):
    os.makedirs(CACHE_DIR, exist_ok=True)
    exchange = ccxt.binance({'enableRateLimit': True})

    if symbols is None:
        symbols = ["BTC/USDT"]

    print(f"Downloading full 1h history for {len(symbols)} symbols...")
    print(f"Target: maximum available history from Binance (since 2017)\n")

    results = {}
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] {symbol}")
        try:
            count = download_full(symbol, exchange)
            results[symbol] = count
        except Exception as e:
            print(f"  FAILED: {e}")
            results[symbol] = -1

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for symbol, count in sorted(results.items(), key=lambda x: -x[1]):
        sc = symbol.replace("/", "")
        print(f"  {sc}: {count} bars")

    total = sum(c for c in results.values() if c > 0)
    failed = sum(1 for c in results.values() if c <= 0)
    print(f"\nTotal: {total} bars across {len(symbols)-failed}/{len(symbols)} symbols")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Descarga histórico 1h de Binance (default: solo BTC/USDT)")
    parser.add_argument('--symbols', nargs='+', default=None,
                        help="Símbolos a descargar (default: BTC/USDT)")
    parser.add_argument('--all', action='store_true',
                        help="Descarga los 45 símbolos del estudio (masivo: ~45×~79k barras)")
    args = parser.parse_args()
    main(symbols=SYMBOLS if args.all else args.symbols)
