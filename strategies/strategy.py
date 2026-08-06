from lumibot.strategies import Strategy as _LumibotStrategy
import numpy as np


class Strategy(_LumibotStrategy):
    """
RSI Filter Mean Reversion Strategy (research implementation).

- Wake once per day (self.sleeptime = "1D").
- Universe: BTC/USD, ETH/USD, SOL/USD, XRP/USD, DOGE/USD,
  LINK/USD, AVAX/USD, ADA/USD.
- At each iteration, retrieve completed historical daily closes for
  each asset and compute:
    - 14-day RSI using a simple moving average (SMA) of gains and losses.
    - 3-day momentum:

        momentum = close[t-1] / close[t-4] - 1

      where iloc[-1] is the most recently completed daily close returned
      by the data API.

- Filter the universe to assets with RSI < 35.
- Among the qualifying assets, select the two with the most negative
  3-day momentum (largest recent declines).
- Portfolio allocation:
    - Two qualifying assets: 50% / 50%.
    - One qualifying asset: 50% invested, 50% held in cash.
    - No qualifying assets: remain 100% in cash.
- Rebalance once per day and exit any position that no longer satisfies
  the selection criteria.

Notes on timing and safety:
- The implementation intentionally uses only completed daily bars from
  self.get_historical_prices(..., timestep="day") and references
  iloc[-1] and iloc[-4] to avoid look-ahead bias.
- Assets with insufficient historical data are skipped until enough
  completed daily closes are available to compute RSI and momentum.
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

    def _rsi(self, series, window: int = 14):
        """Simple RSI implementation matching research (SMA of gains/losses)."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window, min_periods=window).mean()
        avg_loss = loss.rolling(window, min_periods=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def on_trading_iteration(self):
        # RSI Filter Mean Reversion
        # For each symbol compute:
        #  - RSI(14) using completed daily closes (SMA of gains/losses)
        #  - 3-day completed momentum: close[t-1] / close[t-4] - 1
        # Select symbols with RSI < 35. From those, pick the two assets with the
        # lowest 3-day momentum (most negative). Equal-weight 50/50. If fewer than
        # two qualified assets, allocate as available. If none qualify, stay in cash.

        signals = []  # list of (symbol, momentum, rsi)

        for symbol in self.symbols:
            try:
                # Request enough history for RSI(14) + momentum (needs 4 closes)
                bars = self.get_historical_prices(symbol, length=20, timestep="day")
            except Exception:
                self.log_message(f"[warn] historical data call failed for {symbol}")
                continue

            if bars is None or getattr(bars, "df", None) is None or bars.df.empty:
                continue

            close = bars.df.get("close")
            if close is None:
                continue

            # Need sufficient history for RSI(14) and 3-day momentum (4 closes)
            if len(close.dropna()) < 18:
                continue

            try:
                recent = float(close.iloc[-1])
                prior = float(close.iloc[-4])
            except Exception:
                continue

            if prior == 0:
                continue

            momentum = recent / prior - 1.0
            rsi_series = self._rsi(close, 14)
            rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.empty else None

            if rsi_val is None or np.isnan(rsi_val):
                continue

            signals.append((symbol, momentum, rsi_val))

        # Filter by RSI < 35
        qualified = [(s, m, r) for (s, m, r) in signals if r < 35]

        if len(qualified) == 0:
            # No trades; remain in cash
            return

        # Rank by momentum ascending (most negative first) and pick up to max_holdings
        qualified.sort(key=lambda x: x[1])
        picks = qualified[: self.max_holdings]
        picked_symbols = [s for s, _, _ in picks]

        # Build target weights (equal-weighted among picks)
        target_weights = {s: 0.0 for s in self.symbols}
        if len(picked_symbols) == 1:
            # Match research: allocate only 50% to a single qualifying asset and keep 50% cash
            target_weights[picked_symbols[0]] = 0.5
        else:
            for s in picked_symbols:
                target_weights[s] = 1.0 / len(picked_symbols)

        # Convert weights to dollar targets and submit orders
        portfolio_value = float(self.get_portfolio_value())

        qty_lookup = {}
        try:
            positions = self.get_positions() or []
            for p in positions:
                qty_lookup[getattr(p, "symbol", None)] = float(getattr(p, "quantity", 0.0))
        except Exception:
            positions = []

        for symbol in self.symbols:
            target_w = float(target_weights.get(symbol, 0.0))
            if portfolio_value is None or portfolio_value <= 0:
                return

            price = self.get_last_price(symbol)
            if price is None or price <= 0:
                continue

            desired_value = portfolio_value * target_w
            desired_qty = desired_value / price

            current_qty = float(qty_lookup.get(symbol, 0.0))
            diff = desired_qty - current_qty

            if abs(diff) < 1e-8:
                continue

            if diff > 0:
                order = self.create_order(symbol, diff, "buy")
                self.submit_order(order)
            else:
                order = self.create_order(symbol, abs(diff), "sell")
                self.submit_order(order)

        return
