"""
Example Strategy 2 - One-shot buy-and-hold of SPY.

Reference implementation showing how to use ``first_iteration`` style state
to do a single trade at the start of a backtest, then hold passively.

Behaviour
---------
On the very first ``on_trading_iteration`` call, invest (almost) all
available cash into SPY using fractional shares, then do nothing for the
rest of the run - a classic buy-and-hold baseline.

How to run this via ``backtest.py``
-----------------------------------
1. Make sure you have minute-bar data for SPY at ``data/SPY_1m_spot.csv``
   (or change the ticker below to a symbol you do have data for).
2. Add the ticker to ``STOCK_SLEEVE_SYMBOLS`` in ``strategies/params.py``.
3. In ``backtest.py`` swap the strategy import line::

       from strategies.example_strategy_2 import example_strategy_2 as Strategy
"""

from lumibot.strategies import Strategy


class example_strategy_2(Strategy):
    """Invest all cash in SPY on the first iteration, then hold."""

    def initialize(self):
        # Wake up once per trading day - we only trade on the first call.
        self.sleeptime = "1D"

        # Track whether the initial buy has already gone through. Using a
        # plain attribute (instead of ``self.first_iteration``) means the
        # one-shot trade still works even if the first call no-ops because
        # data is unavailable.
        self.has_bought = False

    def on_trading_iteration(self):
        if self.has_bought:
            return

        cash = self.get_cash()
        if cash <= 0:
            return

        price = self.get_last_price("SPY")
        if price is None or price <= 0:
            # No price yet (e.g. market not open); try again next iteration.
            return

        # Fractional shares let us deploy (almost) every available dollar.
        quantity = cash / price
        if quantity <= 0:
            return

        order = self.create_order("SPY", quantity, "buy")
        self.submit_order(order)
        self.has_bought = True

        self.log_message(
            f"[example_strategy_2] initial buy-and-hold in SPY. "
            f"cash={self.get_cash():,.2f}, positions={self.get_positions()}"
        )
