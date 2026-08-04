"""
Parameters shared by the local backtest harness.

The official competition execution environment is provided by the
IntelligenceX technical team and may expose its own universe. This file
controls the *local* backtest harness only (``backtest.py``), so you can
iterate quickly during development.

Our strategy trades a crypto-only universe (BTC, ETH, SOL) - see
``strategies/strategy.py`` for the full rule set. STOCK_SLEEVE_SYMBOLS is
intentionally empty; STOCK_BENCH is pointed at BTC purely so the
Lumibot tearsheet has a benchmark line to compare against (the "STOCK_"
prefix is just the template's naming, not a claim that BTC is an equity).
"""

# Equity / spot tickers traded by the local backtest. Empty: this strategy
# is crypto-only.
STOCK_SLEEVE_SYMBOLS: list[str] = []

# Crypto tickers (quoted in USD), matching Strategy.CRYPTO_UNIVERSE in
# strategies/strategy.py. Each needs a matching data/{SYMBOL}_1m_spot.csv.
CRYPTO_SLEEVE_SYMBOLS: list[str] = ["BTC", "ETH", "SOL"]

# Benchmark symbols used by Lumibot to render the comparison line on the
# generated tearsheet HTML. backtest.py only wires up STOCK_BENCH, so we
# point it at BTC (loaded via the crypto CSV) to get a meaningful benchmark.
STOCK_BENCH: str = "BTC"
CRYPTO_BENCH: str = "BTC"

# Derived set used by ``backtest.py`` to decide whether a loaded symbol
# should be modelled as ``Asset.AssetType.CRYPTO`` vs ``STOCK``. Do not
# edit directly; change ``CRYPTO_SLEEVE_SYMBOLS`` instead.
CRYPTO_SYMBOLS: set[str] = set(CRYPTO_SLEEVE_SYMBOLS)
