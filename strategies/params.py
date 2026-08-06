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

The template ships with a single placeholder symbol (``EXAMPLE``) so the
harness runs end-to-end out of the box against the sample CSV in ``data/``.
"""

# Equity / spot tickers traded by the local backtest. Add your own symbols
# here and drop the matching CSV files into ``data/``.
STOCK_SLEEVE_SYMBOLS: list[str] = [
    "EXAMPLE",
]

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

# Benchmark symbols. Used by Lumibot to render the comparison line on the
# generated tearsheet HTML.
STOCK_BENCH: str = "EXAMPLE"
CRYPTO_BENCH: str = "EXAMPLE"

# Derived set used by ``backtest.py`` to decide whether a loaded symbol
# should be modelled as ``Asset.AssetType.CRYPTO`` vs ``STOCK``. Do not
# edit directly; change ``CRYPTO_SLEEVE_SYMBOLS`` instead.
CRYPTO_SYMBOLS: set[str] = set(CRYPTO_SLEEVE_SYMBOLS)
