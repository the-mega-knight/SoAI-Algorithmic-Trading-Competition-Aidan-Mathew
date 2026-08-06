"""
Parameters shared by the local backtest harness.

The official competition execution environment is provided by the
IntelligenceX technical team and may expose its own universe. This file
controls the *local* backtest harness only (``backtest.py``), so you can
iterate quickly during development.

Our strategy trades a large-cap US equity universe - see
``strategies/strategy.py`` for the full rule set.
"""

# Equity tickers traded by the strategy and the local backtest. Each needs
# a matching data/{SYMBOL}_daily.csv (see scripts/fetch_stock_data.py).
STOCK_SLEEVE_SYMBOLS: list[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

# No crypto sleeve in this version of the strategy.
CRYPTO_SLEEVE_SYMBOLS: list[str] = []

# Benchmark symbol used by Lumibot to render the comparison line on the
# generated tearsheet HTML. BENCH5 is a synthetic equal-weight, daily
# rebalanced composite of the 5 sleeve symbols above (see
# scripts/build_benchmark.py) - a single-AAPL benchmark isn't a fair
# comparison since AAPL is just one of the five names we actually hold.
# Run `python3 scripts/build_benchmark.py` (after fetch_stock_data.py) to
# generate data/BENCH5_daily.csv before running backtest.py.
STOCK_BENCH: str = "BENCH5"
CRYPTO_BENCH: str = "AAPL"

# Derived set used by ``backtest.py``. Empty here since there's no crypto
# sleeve; kept so backtest.py's classification logic still works untouched.
CRYPTO_SYMBOLS: set[str] = set(CRYPTO_SLEEVE_SYMBOLS)
