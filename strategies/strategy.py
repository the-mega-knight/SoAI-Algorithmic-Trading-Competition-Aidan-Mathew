"""
SoAI 2026 AI Algorithmic Trading Competition - participant entrypoint.

Strategy: Volatility-Scaled Crypto Momentum
--------------------------------------------
One-sentence hypothesis: crypto assets in a confirmed short-term uptrend
(10-day SMA above 20-day SMA, with positive 10-day momentum) tend to keep
trending for long enough to be worth holding, so we hold every asset that
qualifies at once, weighted inversely to its own recent volatility (so no
single volatile name dominates portfolio risk), and drop it the instant
the trend breaks.

Every decision below is mechanical - no discretionary judgment calls, so
the exact same inputs always produce the exact same output (a requirement
for both honest backtesting and passing the official verification run).

Rules (zero discretion)
------------------------
Universe:        BTC, ETH, SOL (spot, USD-quoted; see CRYPTO_UNIVERSE)
Cadence:         once per day (self.sleeptime = "1D")
Entry signal:    SMA(10) > SMA(20)  AND  10-day ROC > 0
Exit signal:     SMA(10) <= SMA(20)  (trend break -> flatten immediately)
Position sizing: inverse-volatility weighting across assets with an active
                 entry signal, capped at MAX_ASSET_WEIGHT per asset and
                 MAX_TOTAL_EXPOSURE of the portfolio in total
Risk control:    portfolio-level drawdown circuit breaker - if portfolio
                 value drawns down more than MAX_DRAWDOWN_PCT from its
                 running peak, flatten everything and pause new entries
                 for DRAWDOWN_COOLDOWN_DAYS trading days
Rebalance rule:  only trade an asset when the target position differs from
                 the current one by more than MIN_REBALANCE_PCT of
                 portfolio value, to avoid churning on noise

See the project README for the full writeup, and
`trading-backtest-methodology` / `trading-position-sizer` /
`trading-strategy-signal-toolkit` for the methodology this was built with.

The official execution environment imports the class defined here, so:

* Keep the class name ``Strategy``.
* Keep this file at ``strategies/strategy.py``.
* Keep the import path ``from strategies.strategy import Strategy``.
"""

from lumibot.strategies import Strategy as _LumibotStrategy


class Strategy(_LumibotStrategy):
    # ------------------------------------------------------------------
    # Tunable parameters (all part of the codified rule set above -
    # changing these is "refine", not "add discretion", per the
    # trading-backtest-methodology stress-testing workflow).
    # ------------------------------------------------------------------
    CRYPTO_UNIVERSE = ["BTC", "ETH", "SOL"]

    FAST_SMA_DAYS = 10
    SLOW_SMA_DAYS = 20
    MOMENTUM_LOOKBACK_DAYS = 10
    VOL_LOOKBACK_DAYS = 20

    MAX_ASSET_WEIGHT = 0.50       # no single asset above 50% of portfolio
    MAX_TOTAL_EXPOSURE = 0.90     # always keep >=10% cash buffer
    MIN_REBALANCE_PCT = 0.02      # ignore rebalances smaller than 2% of NAV

    MAX_DRAWDOWN_PCT = 0.25       # 25% drawdown from peak trips the breaker
    DRAWDOWN_COOLDOWN_DAYS = 5    # trading days to stay flat after a trip

    # How much minute history to pull per iteration to build daily bars.
    # SLOW_SMA_DAYS + a safety margin, converted to minutes.
    _HISTORY_DAYS = max(SLOW_SMA_DAYS, MOMENTUM_LOOKBACK_DAYS, VOL_LOOKBACK_DAYS) + 10

    # ------------------------------------------------------------------
    # Lifecycle: setup
    # ------------------------------------------------------------------
    def initialize(self):
        self.sleeptime = "1D"

        # Running peak portfolio value, for the drawdown circuit breaker.
        self.peak_portfolio_value = None
        # Trading days remaining in the post-breaker cooldown (0 = not in cooldown).
        self.cooldown_days_left = 0

        self.log_message(
            f"Strategy initialized. Universe={self.CRYPTO_UNIVERSE}, "
            f"sleeptime={self.sleeptime}"
        )

    # ------------------------------------------------------------------
    # Lifecycle: per-step decision making
    # ------------------------------------------------------------------
    def on_trading_iteration(self):
        portfolio_value = self.get_portfolio_value()

        # -- Step 1: update the running peak and check the circuit breaker.
        if self.peak_portfolio_value is None or portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value

        drawdown = (portfolio_value / self.peak_portfolio_value) - 1.0 if self.peak_portfolio_value else 0.0

        if self.cooldown_days_left > 0:
            self.cooldown_days_left -= 1
            self._flatten_all()
            self.log_message(
                f"[cooldown] {self.cooldown_days_left} day(s) left. "
                f"portfolio=${portfolio_value:,.2f} drawdown={drawdown:.1%}"
            )
            return

        if drawdown <= -self.MAX_DRAWDOWN_PCT:
            self._flatten_all()
            self.cooldown_days_left = self.DRAWDOWN_COOLDOWN_DAYS
            self.log_message(
                f"[circuit-breaker] drawdown {drawdown:.1%} breached "
                f"-{self.MAX_DRAWDOWN_PCT:.0%} limit. Flattened and entering "
                f"{self.DRAWDOWN_COOLDOWN_DAYS}-day cooldown."
            )
            return

        # -- Step 2: build daily signals for every asset in the universe.
        signals = {}
        for symbol in self.CRYPTO_UNIVERSE:
            daily = self._get_daily_bars(symbol)
            if daily is None or len(daily) < self.SLOW_SMA_DAYS + 1:
                continue  # not enough history yet - sit this one out
            signals[symbol] = self._compute_signal(daily)

        # -- Step 3: qualify assets with an active entry signal.
        qualifying = {s: v for s, v in signals.items() if v["entry"]}

        # -- Step 4: inverse-volatility weights among qualifying assets,
        #    capped per-asset and in total.
        target_weights = self._inverse_vol_weights(qualifying)

        # -- Step 5: translate target weights into target dollar values,
        #    then into orders, for every symbol in the universe (assets
        #    that dropped out of `qualifying` get a target weight of 0,
        #    which flattens them).
        for symbol in self.CRYPTO_UNIVERSE:
            target_weight = target_weights.get(symbol, 0.0)
            self._rebalance_to_weight(symbol, target_weight, portfolio_value)

        self.log_message(
            f"[Strategy] portfolio=${portfolio_value:,.2f} "
            f"cash=${self.get_cash():,.2f} drawdown={drawdown:.1%} "
            f"qualifying={list(qualifying.keys())} weights={target_weights}"
        )

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------
    def _get_daily_bars(self, symbol):
        """
        Pull minute bars and resample to daily OHLC ourselves, rather than
        relying on the backtesting engine's own timestep resampling - this
        keeps behavior identical across the local Pandas/CSV backtest and
        the official minute-bar-only live feed.
        """
        length_minutes = self._HISTORY_DAYS * 24 * 60
        bars = self.get_historical_prices(symbol, length_minutes, "minute")
        if bars is None or bars.df is None or bars.df.empty:
            return None

        df = bars.df
        daily = df.resample("1D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        return daily

    def _compute_signal(self, daily):
        close = daily["close"]

        sma_fast = close.rolling(self.FAST_SMA_DAYS).mean().iloc[-1]
        sma_slow = close.rolling(self.SLOW_SMA_DAYS).mean().iloc[-1]

        roc = close.pct_change(periods=self.MOMENTUM_LOOKBACK_DAYS).iloc[-1]

        daily_returns = close.pct_change().dropna()
        vol = daily_returns.tail(self.VOL_LOOKBACK_DAYS).std()
        if vol is None or vol != vol or vol <= 0:  # NaN or non-positive guard
            vol = float("inf")  # effectively zero-weights an unusable series

        entry = bool(sma_fast > sma_slow) and bool(roc > 0)

        return {
            "entry": entry,
            "sma_fast": sma_fast,
            "sma_slow": sma_slow,
            "roc": roc,
            "vol": vol,
        }

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    def _inverse_vol_weights(self, qualifying):
        """
        weight_i = (1/vol_i) / sum(1/vol_j), then capped at
        MAX_ASSET_WEIGHT per asset and rescaled so the total never
        exceeds MAX_TOTAL_EXPOSURE.
        """
        if not qualifying:
            return {}

        inv_vol = {s: (1.0 / v["vol"]) for s, v in qualifying.items() if v["vol"] > 0}
        total_inv_vol = sum(inv_vol.values())
        if total_inv_vol <= 0:
            return {}

        raw_weights = {s: w / total_inv_vol for s, w in inv_vol.items()}

        # Cap per-asset weight, then rescale the remainder proportionally
        # among uncapped assets (simple iterative water-filling; the
        # universe is tiny (<=3 assets) so a fixed-point loop is plenty).
        weights = dict(raw_weights)
        for _ in range(len(weights) + 1):
            over = {s: w for s, w in weights.items() if w > self.MAX_ASSET_WEIGHT}
            if not over:
                break
            capped_total = sum(self.MAX_ASSET_WEIGHT for _ in over)
            remaining = 1.0 - capped_total
            under = {s: w for s, w in weights.items() if s not in over}
            under_total = sum(under.values())
            for s in over:
                weights[s] = self.MAX_ASSET_WEIGHT
            if under_total > 0:
                for s in under:
                    weights[s] = (under[s] / under_total) * remaining

        # Scale the whole book down to MAX_TOTAL_EXPOSURE.
        scale = self.MAX_TOTAL_EXPOSURE
        return {s: w * scale for s, w in weights.items()}

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------
    def _current_position_value(self, symbol):
        position = self.get_position(symbol)
        if position is None or position.quantity in (None, 0):
            return 0.0, 0.0
        price = self.get_last_price(symbol)
        if price is None:
            return float(position.quantity), 0.0
        return float(position.quantity), float(position.quantity) * float(price)

    def _rebalance_to_weight(self, symbol, target_weight, portfolio_value):
        price = self.get_last_price(symbol)
        if price is None or price <= 0:
            return  # no tradable price this iteration - skip safely

        current_qty, current_value = self._current_position_value(symbol)
        target_value = target_weight * portfolio_value
        delta_value = target_value - current_value

        if abs(delta_value) < self.MIN_REBALANCE_PCT * max(portfolio_value, 1.0):
            return  # inside the no-trade band - avoid churn/fees on noise

        delta_qty = delta_value / price
        if delta_qty > 0:
            # Buying: never spend more than available cash.
            affordable_qty = self.get_cash() / price
            delta_qty = min(delta_qty, affordable_qty)
            if delta_qty <= 0:
                return
            order = self.create_order(symbol, delta_qty, "buy")
            self.submit_order(order)
        else:
            sell_qty = min(abs(delta_qty), current_qty) if current_qty else 0
            if sell_qty <= 0:
                return
            order = self.create_order(symbol, sell_qty, "sell")
            self.submit_order(order)

    def _flatten_all(self):
        for symbol in self.CRYPTO_UNIVERSE:
            qty, _ = self._current_position_value(symbol)
            if qty and qty > 0:
                order = self.create_order(symbol, qty, "sell")
                self.submit_order(order)
