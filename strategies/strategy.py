from lumibot.strategies import Strategy as _LumibotStrategy


class Strategy(_LumibotStrategy):
    """
    3-day cross-sectional momentum strategy (research implementation).

    - Wake once per day (self.sleeptime = "1D").
    - Universe: BTC/USD, ETH/USD, SOL/USD, XRP/USD, DOGE/USD, LINK/USD, AVAX/USD, ADA/USD
    - At each iteration, compute 3-day momentum for each symbol using only
      completed historical daily closes:

        momentum = close[t-1] / close[t-4] - 1

      where iloc[-1] is the most recently *completed* daily close returned by
      the data API. If fewer than 4 completed daily closes are available for a
      symbol it is excluded for that iteration.

    - Rank by momentum, pick top 2, allocate 50% / 50%.
    - Rebalance to target weights; liquidate assets that drop out of the top 2.

    Notes on timing and safety:
    - The implementation intentionally reads only completed daily bars from
      self.get_historical_prices(..., timestep="day") and references iloc[-1]
      and iloc[-4] to avoid using the current day's intraday data.
    - If fewer than 2 valid symbols are available the strategy no-ops.
    """

    def initialize(self):
        # Wake up once per trading day.
        self.sleeptime = "1D"

        # Explicit universe and limits
        self.symbols = [
            "BTC/USD",
            "ETH/USD",
            "SOL/USD",
            "XRP/USD",
            "DOGE/USD",
            "LINK/USD",
            "AVAX/USD",
            "ADA/USD",
        ]
        self.max_holdings = 2

        # Minimal internal state
        self._last_targets = {}

        self.log_message("[strategy] 3-day momentum initialized")

    def on_trading_iteration(self):
        # Gather momentum signals
        signals = []  # list of (symbol, momentum)

        for symbol in self.symbols:
            try:
                bars = self.get_historical_prices(symbol, length=5, timestep="day")
            except Exception:
                self.log_message(f"[warn] historical data call failed for {symbol}")
                continue

            if bars is None or getattr(bars, "df", None) is None or bars.df.empty:
                # No data available for this symbol in this iteration.
                # Keep logging minimal.
                continue

            close = bars.df.get("close")
            if close is None or len(close.dropna()) < 4:
                # Need at least 4 completed daily closes to compute close[t-1]/close[t-4]
                continue

            # Use iloc[-1] and iloc[-4] which refer to the most recently completed
            # daily close and the close 3 trading days earlier respectively.
            try:
                recent = float(close.iloc[-1])
                prior = float(close.iloc[-4])
            except Exception:
                continue

            if prior == 0:
                continue

            momentum = recent / prior - 1.0
            signals.append((symbol, momentum))

        # If not enough valid symbols, do nothing this iteration.
        if len(signals) < self.max_holdings:
            return

        # Rank by momentum descending and take top N
        signals.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [s for s, _ in signals[: self.max_holdings]]

        # Build target weights (equal-weighted among top picks)
        target_weights = {s: 0.0 for s in self.symbols}
        for s in top_symbols:
            target_weights[s] = 1.0 / self.max_holdings

        # Convert weights to dollar targets and submit minimal orders to move
        # current holdings toward targets. Avoid creating zero-quantity orders.
        portfolio_value = float(self.get_portfolio_value())

        # Build current position lookup
        qty_lookup = {}
        try:
            positions = self.get_positions() or []
            for p in positions:
                # Position.symbol exists on Position objects
                qty_lookup[getattr(p, "symbol", None)] = float(getattr(p, "quantity", 0.0))
        except Exception:
            positions = []

        # For every symbol in universe, compute desired quantity and submit diff order
        for symbol in self.symbols:
            target_w = float(target_weights.get(symbol, 0.0))
            # If portfolio_value is zero or invalid, skip trading to avoid division errors
            if portfolio_value is None or portfolio_value <= 0:
                return

            price = self.get_last_price(symbol)
            # If price unavailable, skip trading this symbol
            if price is None or price <= 0:
                continue

            desired_value = portfolio_value * target_w
            desired_qty = desired_value / price

            current_qty = float(qty_lookup.get(symbol, 0.0))
            diff = desired_qty - current_qty

            # Avoid tiny micro trades; require a minimal quantity magnitude
            if abs(diff) < 1e-8:
                continue

            if diff > 0:
                # Buy the difference
                order = self.create_order(symbol, diff, "buy")
                self.submit_order(order)
            else:
                # Sell the excess
                order = self.create_order(symbol, abs(diff), "sell")
                self.submit_order(order)

        # Done for this iteration
        return
