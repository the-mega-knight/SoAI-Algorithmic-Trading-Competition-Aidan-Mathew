"""
Builds a synthetic 5-stock equal-weight benchmark series for the local
backtest harness.

Why this exists
----------------
Lumibot's Strategy.run_backtest(benchmark_asset=...) only accepts a single
symbol for the tearsheet comparison line - it can't natively average
multiple tickers. Comparing our 5-stock momentum portfolio against AAPL
alone isn't a fair benchmark (AAPL is just one of the five names we hold,
and its solo performance doesn't represent "what if I'd just bought and
held the universe").

This script fixes that by synthesizing a single composite price series -
an equal-weighted, daily-rebalanced buy-and-hold of the same 5 stocks the
strategy trades (STOCK_SLEEVE_SYMBOLS in strategies/params.py) - and
writing it out as data/BENCH5_daily.csv in the same schema Lumibot expects
(open, high, low, close, volume, timestamp). backtest.py then loads it
like any other symbol once strategies/params.py points STOCK_BENCH at it.

Usage
-----
    python3 scripts/fetch_stock_data.py   # if you haven't already
    python3 scripts/build_benchmark.py
    python3 backtest.py
"""
from pathlib import Path

import pandas as pd

from strategies.params import STOCK_SLEEVE_SYMBOLS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_SYMBOL = "BENCH5"


def _load_close(symbol: str) -> pd.Series:
    path = DATA_DIR / f"{symbol}_daily.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/fetch_stock_data.py first."
        )
    df = pd.read_csv(path, usecols=["close", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    return df["close"]


def build_benchmark() -> pd.DataFrame:
    closes = {sym: _load_close(sym) for sym in STOCK_SLEEVE_SYMBOLS}
    prices = pd.DataFrame(closes).dropna(how="any")
    if prices.empty:
        raise RuntimeError("No overlapping dates across the 5 sleeve symbols.")

    # Equal-weight, daily-rebalanced composite: average of each stock's
    # daily return, compounded into a single synthetic price series
    # starting at 100. This mirrors "split $1 into 5 equal chunks, one per
    # stock, rebalanced back to equal weight every day."
    daily_returns = prices.pct_change().dropna(how="all").mean(axis=1)
    composite = (1 + daily_returns).cumprod() * 100.0
    composite = pd.concat([pd.Series([100.0], index=[prices.index[0]]), composite])
    composite = composite[~composite.index.duplicated(keep="first")].sort_index()

    out = pd.DataFrame(
        {
            "open": composite,
            "high": composite,
            "low": composite,
            "close": composite,
            "volume": 0,
        }
    )
    out.index.name = "timestamp"
    return out.reset_index()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = build_benchmark()
    out_path = DATA_DIR / f"{OUTPUT_SYMBOL}_daily.csv"
    df.to_csv(out_path, index=False)
    print(f"[INFO] Wrote {len(df)} rows to {out_path}")
    print(f"[INFO] Now set STOCK_BENCH = \"{OUTPUT_SYMBOL}\" in strategies/params.py (already done if you pulled latest).")


if __name__ == "__main__":
    main()
