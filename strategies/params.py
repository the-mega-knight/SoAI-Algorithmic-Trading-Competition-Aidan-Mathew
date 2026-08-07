"""
Parameters shared by the local backtest harness.

The official competition execution environment is provided by the
IntelligenceX technical team and may expose its own universe. This file
controls the *local* backtest harness only (``backtest.py``), so you can
iterate quickly during development.

How to customise
----------------
1. List the equity / spot tickers you want to trade in
   ``STOCK_SLEEVE_SYMBOLS``.
2. List the crypto tickers (quoted in USD) in ``CRYPTO_SLEEVE_SYMBOLS``.
3. Pick benchmarks for the Lumibot tearsheet comparison line via
   ``STOCK_BENCH`` / ``CRYPTO_BENCH``.
4. Make sure each symbol has a matching ``data/{SYMBOL}_1m_spot.csv``
   file before running ``python backtest.py``.

NOTE: the template's placeholder ``EXAMPLE`` symbol has been removed below.
Its CSV only spans 2026-08-01 -> 2026-08-29 with a flat $400 price, and
because backtest.py clamps the run window to the INTERSECTION of every
loaded symbol's date range, leaving it in (including as STOCK_BENCH /
CRYPTO_BENCH, which also get loaded as data) silently truncated the real
crypto backtest from the full ~1 month available (2026-07-06 -> 2026-08-05)
down to just 5 days (2026-08-01 -> 08-05).
"""

# Equity / spot tickers traded by the local backtest. Add your own symbols
# here and drop the matching CSV files into ``data/``.
STOCK_SLEEVE_SYMBOLS: list[str] = []

# Crypto tickers (quoted in USD). Leave empty if your strategy is stocks-only.
CRYPTO_SLEEVE_SYMBOLS: list[str] = [
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "XRP/USD",
    "DOGE/USD",
    "LINK/USD",
    "AVAX/USD",
    "ADA/USD",
]

# Benchmark symbols. Left blank - backtest.py now passes benchmark_asset=None
# directly to avoid both the EXAMPLE-placeholder date clamp above and the
# broken Lumibot Yahoo-Finance benchmark fetch (same bug already fixed on
# the Aidan branch for BENCH5).
STOCK_BENCH: str = ""
CRYPTO_BENCH: str = ""

# Derived set used by ``backtest.py`` to decide whether a loaded symbol
# should be modelled as ``Asset.AssetType.CRYPTO`` vs ``STOCK``. Do not
# edit directly; change ``CRYPTO_SLEEVE_SYMBOLS`` instead.
CRYPTO_SYMBOLS: set[str] = set(CRYPTO_SLEEVE_SYMBOLS)
