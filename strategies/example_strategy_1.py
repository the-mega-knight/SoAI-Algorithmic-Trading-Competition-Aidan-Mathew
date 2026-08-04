"""
Example Strategy 1 - Daily DCA into SPY.

Reference implementation showing the absolute minimum a Lumibot strategy
needs: a class that inherits from ``lumibot.strategies.Strategy`` and
implements :meth:`initialize` plus :meth:`on_trading_iteration`.

Behaviour
---------
Once per trading day, buy one share of SPY at market.

How to run this via ``backtest.py``
-----------------------------------
1. Make sure you have minute-bar data for SPY at ``data/SPY_1m_spot.csv``
   (or change the ticker below to a symbol you do have data for, e.g.
   ``"EXAMPLE"`` to use the placeholder CSV that ships with the template).
2. Add the ticker to ``STOCK_SLEEVE_SYMBOLS`` in ``strategies/params.py``.
3. In ``backtest.py`` swap the strategy import line::

       from strategies.example_strategy_1 import example_strategy_1 as Strategy
"""

from lumibot.strategies import Strategy


class example_strategy_1(Strategy):
    """Buy one share of SPY at market every trading day."""

    def initialize(self):
        # Wake up once per trading day.
        self.sleeptime = "1D"

    def on_trading_iteration(self):
        # Submit a market buy for one share of SPY every iteration.
        order = self.create_order("SPY", 1, "buy")
        self.submit_order(order)
