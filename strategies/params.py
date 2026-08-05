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
# generated tearsheet HTML.
STOCK_BENCH: str = "AAPL"
CRYPTO_BENCH: str = "AAPL"

# Derived set used by ``backtest.py``. Empty here since there's no crypto
# sleeve; kept so backtest.py's classification logic still works untouched.
CRYPTO_SYMBOLS: set[str] = set(CRYPTO_SLEEVE_SYMBOLS)
