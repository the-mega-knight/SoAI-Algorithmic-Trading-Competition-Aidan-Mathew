import ccxt
import time
import os
import csv
import sys
from datetime import datetime, timezone, timedelta

# Symbols to download
SYMBOLS = [
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "XRP/USD",
    "DOGE/USD",
    "LINK/USD",
    "AVAX/USD",
    "ADA/USD",
]

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'historical')
os.makedirs(DATA_DIR, exist_ok=True)

# Settings
TIMEFRAME = '1m'
LIMIT = 1000  # max per fetch_ohlcv call (exchange dependent)
# Target ~365 days
DAYS = 365
MILLISECONDS_IN_MINUTE = 60 * 1000

# Rate limit / retry settings
REQUEST_DELAY = 0.34  # seconds between requests (be conservative)
MAX_RETRIES = 5


def ms_now():
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def timestamp_to_iso(ms):
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


def download_symbol(exchange, symbol):
    out_name = f"{symbol.replace('/', '_')}_1m_spot.csv"
    out_path = os.path.join(DATA_DIR, out_name)

    # don't overwrite existing file; write incrementally
    temp_path = out_path + '.part'

    # compute since timestamp (12 months ago)
    now_ms = ms_now()
    since_ms = now_ms - DAYS * 24 * 60 * 60 * 1000

    # align since to minute
    since_ms = since_ms - (since_ms % MILLISECONDS_IN_MINUTE)

    print(f"Downloading {symbol} -> {out_path}")

    # if temp exists, resume from last timestamp
    if os.path.exists(temp_path):
        try:
            with open(temp_path, 'r', newline='') as f:
                last = None
                for row in csv.reader(f):
                    last = row
                if last:
                    last_ts = int(last[0])
                    since_ms = last_ts + MILLISECONDS_IN_MINUTE
                    print(f"Resuming {symbol} from {timestamp_to_iso(since_ms)}")
        except Exception as e:
            print(f"Warning: could not resume {symbol}: {e}")

    # Open temp file for append
    with open(temp_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # write header if file was just created
        if csvfile.tell() == 0:
            writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        finished = False
        consecutive_empty = 0
        while not finished:
            # Stop when since >= last complete minute
            if since_ms >= now_ms - MILLISECONDS_IN_MINUTE:
                print(f"Reached most recent complete candle for {symbol} at {timestamp_to_iso(since_ms)}")
                break

            retries = 0
            while True:
                try:
                    ohlcvs = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=since_ms, limit=LIMIT)
                    break
                except ccxt.NetworkError as e:
                    retries += 1
                    if retries > MAX_RETRIES:
                        print(f"Network error downloading {symbol}: {e}")
                        return False
                    wait = 2 ** retries
                    print(f"Network error, retrying {symbol} in {wait}s... ({e})")
                    time.sleep(wait)
                except ccxt.ExchangeError as e:
                    print(f"Exchange error for {symbol}: {e}")
                    return False
                except Exception as e:
                    print(f"Unexpected error for {symbol}: {e}")
                    return False

            if not ohlcvs:
                # no data returned; avoid tight loop
                consecutive_empty += 1
                if consecutive_empty > 5:
                    print(f"No more data available for {symbol}; stopping at {timestamp_to_iso(since_ms)}")
                    break
                time.sleep(1)
                continue

            consecutive_empty = 0

            # Write rows, ensure chronological order
            for bar in ohlcvs:
                ts = int(bar[0])
                # skip any bars earlier than since_ms (shouldn't happen)
                if ts < since_ms:
                    continue
                open_, high, low, close, volume = bar[1], bar[2], bar[3], bar[4], bar[5]
                writer.writerow([ts, open_, high, low, close, volume])

            last_ts = int(ohlcvs[-1][0])
            since_ms = last_ts + MILLISECONDS_IN_MINUTE

            # be polite
            time.sleep(REQUEST_DELAY)

    # Move .part to final file (atomic-ish)
    try:
        os.replace(temp_path, out_path)
    except Exception as e:
        print(f"Failed to finalize {out_path}: {e}")
        return False

    return True


def validate_file(path):
    import csv
    from collections import Counter
    counts = {'rows': 0, 'dup_ts': 0, 'nan_values': 0, 'non_numeric': 0}
    first_ts = None
    last_ts = None
    ts_prev = None
    monotonic = True
    high_low_violations = 0
    volume_negative = 0

    ts_list = []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts['rows'] += 1
            try:
                ts = int(row['timestamp'])
                ts_list.append(ts)
            except Exception:
                counts['non_numeric'] += 1
                continue
            if first_ts is None:
                first_ts = ts
            last_ts = ts
            if ts_prev is not None and ts <= ts_prev:
                monotonic = False
            ts_prev = ts
            # check numeric OHLCV
            try:
                o = float(row['open']); h = float(row['high']); l = float(row['low']); c = float(row['close']); v = float(row['volume'])
            except Exception:
                counts['nan_values'] += 1
                continue
            if h < max(o, c):
                high_low_violations += 1
            if l > min(o, c):
                high_low_violations += 1
            if v < 0:
                volume_negative += 1

    dup_count = len(ts_list) - len(set(ts_list))
    counts['dup_ts'] = dup_count

    return {
        'rows': counts['rows'],
        'first_ts': first_ts,
        'last_ts': last_ts,
        'dup_ts': counts['dup_ts'],
        'nan_values': counts['nan_values'],
        'non_numeric': counts['non_numeric'],
        'monotonic': monotonic,
        'high_low_violations': high_low_violations,
        'volume_negative': volume_negative,
    }


def main():
    exchange = ccxt.coinbase({
        'enableRateLimit': True,
    })

    # Some installs may expose .rateLimit; ensure a conservative delay
    print('Exchange loaded:', exchange.id)

    results = {}
    for symbol in SYMBOLS:
        success = download_symbol(exchange, symbol)
        if not success:
            print(f"Failed to download {symbol}")
            continue
        out_name = f"{symbol.replace('/', '_')}_1m_spot.csv"
        out_path = os.path.join(DATA_DIR, out_name)
        v = validate_file(out_path)
        results[symbol] = (out_path, v)

    # Print concise report
    for sym, (path, v) in results.items():
        first_iso = datetime.fromtimestamp(v['first_ts']/1000.0, tz=timezone.utc).isoformat() if v['first_ts'] else 'N/A'
        last_iso = datetime.fromtimestamp(v['last_ts']/1000.0, tz=timezone.utc).isoformat() if v['last_ts'] else 'N/A'
        print('\nSymbol:', sym)
        print('  file:', path)
        print('  rows:', v['rows'])
        print('  first_ts:', first_iso)
        print('  last_ts:', last_iso)
        print('  duplicate timestamps:', v['dup_ts'])
        print('  nan/non-numeric OHLCV count:', v['nan_values'] + v['non_numeric'])
        print('  monotonic timestamps:', v['monotonic'])
        print('  high/low violations:', v['high_low_violations'])
        print('  negative volume count:', v['volume_negative'])


if __name__ == '__main__':
    main()
