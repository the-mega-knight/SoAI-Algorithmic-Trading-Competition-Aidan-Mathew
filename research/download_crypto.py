import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
import pandas as pd

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

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

TIMEFRAME = "1m"
DAYS = 30

OUTPUT_DIR = Path("data")

# ---------------------------------------------------------------------
# Exchange
# ---------------------------------------------------------------------

exchange = ccxt.coinbase(
    {
        "enableRateLimit": True,
    }
)

# ---------------------------------------------------------------------
# Download function
# ---------------------------------------------------------------------

def download_symbol(symbol: str, days: int) -> None:
    print(f"\nDownloading {symbol}...")

    exchange.load_markets()

    if symbol not in exchange.markets:
        print(f"WARNING: {symbol} is not available on Coinbase")
        return

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    since = int(start.timestamp() * 1000)
    end_timestamp = int(end.timestamp() * 1000)

    all_candles = []

    while since < end_timestamp:
        print(
            f"  Fetching from "
            f"{datetime.fromtimestamp(since / 1000, timezone.utc)}"
        )

        candles = exchange.fetch_ohlcv(
            symbol,
            timeframe=TIMEFRAME,
            since=since,
            limit=300,
        )

        if not candles:
            print("  No more data returned.")
            break

        all_candles.extend(candles)

        last_timestamp = candles[-1][0]

        # Move forward one minute past the last candle.
        next_since = last_timestamp + 60_000

        if next_since <= since:
            print("  WARNING: timestamp did not advance.")
            break

        since = next_since

        # Be polite to the exchange API.
        time.sleep(exchange.rateLimit / 1000)

    if not all_candles:
        print(f"WARNING: No data downloaded for {symbol}")
        return

    # -----------------------------------------------------------------
    # Convert to DataFrame
    # -----------------------------------------------------------------

    df = pd.DataFrame(
        all_candles,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ],
    )

    # Convert timestamps from milliseconds to UTC datetime.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    # Remove duplicate candles.
    df = df.drop_duplicates(subset="timestamp")

    # Sort chronologically.
    df = df.sort_values("timestamp")

    # Keep only the requested date range.
    df = df[
        (df["timestamp"] >= pd.Timestamp(start))
        & (df["timestamp"] <= pd.Timestamp(end))
    ]

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    filename = symbol.replace("/", "_") + "_1m_spot.csv"
    path = OUTPUT_DIR / filename

    df.to_csv(path, index=False)

    print(
        f"  Saved {len(df):,} candles to {path}"
    )

    print(
        f"  Range: "
        f"{df['timestamp'].iloc[0]} → "
        f"{df['timestamp'].iloc[-1]}"
    )

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("Starting crypto data download")
    print(f"Exchange: {exchange.name}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"History: {DAYS} days")
    print(f"Symbols: {len(SYMBOLS)}")

    for symbol in SYMBOLS:
        try:
            download_symbol(symbol, DAYS)
        except Exception as exc:
            print(
                f"ERROR downloading {symbol}: "
                f"{type(exc).__name__}: {exc}"
            )

    print("\nDownload complete.")

if __name__ == "__main__":
    main()
