"""
SoAI 2026 AI Algorithmic Trading Competition - participant entrypoint.

Strategy: Low-Turnover Momentum Core with Volatility/Gap Risk Throttle
-----------------------------------------------------------------------
One-sentence hypothesis: among a basket of large-cap tech stocks, names
showing sustained positive 30-day momentum tend to keep outperforming for
long enough to be worth holding through minor noise, so the strategy stays
mostly invested and rebalances infrequently (weekly) rather than reacting
to every daily wiggle, while a separate daily-checked volatility/gap
throttle cuts exposure to any single name showing an abnormal move.

Why this design, and not daily binary trend-following
--------------------------------------------------------
Two earlier versions of this strategy (crypto momentum, then daily SMA
crossover on this same equity universe) were both validated against real
historical data and both underperformed simple buy-and-hold by a wide
margin. The common cause: daily binary entry/exit trading generates heavy
turnover (500+ trades over ~450 days), most of it whipsaw on short-term
noise rather than genuine trend changes, and the fees on that turnover ate
a large share of the return. Every version tested pointed the same
direction: staying invested consistently beat trading in and out. This
version is built directly around that finding.

Rules (zero discretion)
------------------------
Universe:         AAPL, MSFT, GOOGL, AMZN, NVDA (see STOCK_UNIVERSE)
Cadence:          checked once per day (self.sleeptime = "1D"), but the
                  target allocation is only recomputed every
                  REBALANCE_EVERY_DAYS trading days
Entry signal:     30-day rate of change > 0 (MOM_LOOKBACK_DAYS)
Weighting:        proportional to each qualifying name's own ROC (not
                  inverse-volatility - see below), capped at
                  MAX_ASSET_WEIGHT per name and MAX_TOTAL_EXPOSURE total
Risk throttle:    checked every day regardless of rebalance schedule,
                  using only OHLCV data:
                    - if 20-day realized vol > VOL_SPIKE_MULTIPLIER x its
                      own 60-day baseline vol, that name's weight is cut
                      to VOL_THROTTLE_FACTOR
                    - if the day's open gaps more than GAP_THRESHOLD_PCT
                      from the prior close, that name's weight is cut to
                      GAP_THROTTLE_FACTOR for GAP_COOLDOWN_DAYS days
                  This is a reactive, fully OHLCV-compliant stand-in for
                  earnings/shock awareness - the official feed provides
                  OHLCV bars only, no news or calendar data, so a strategy
                  that depended on knowing earnings dates in advance would
                  not be reliably reproducible in the official environment.
Risk control:     portfolio-level drawdown circuit breaker - if portfolio
                  value draws down more than MAX_DRAWDOWN_PCT from its
                  running peak, flatten everything and pause new entries
                  for DRAWDOWN_COOLDOWN_DAYS trading days. When cooldown
                  ends, the peak resets to the recovery value rather than
                  the pre-crash high, so the breaker can't permanently
                  lock the strategy out of the market after a single
                  drawdown (a bug caught and fixed during validation).
Rebalance rule:   only trade a name when its target position differs from
                  the current one by more than MIN_REBALANCE_PCT of
                  portfolio value, to avoid churning on noise

Weighting is momentum-proportional rather than inverse-volatility on
purpose: inverse-vol weighting penalizes exactly the names showing the
strongest moves (since strong movers tend to also be higher-volatility),
which was diagnosed as a real drag in an earlier version.

Assets and market hours
------------------------
Bare symbol strings ('AAPL', etc.) resolve correctly by default to stock
assets quoted in forex USD, and the default NASDAQ/NYSE trading calendar
is already correct for stocks, so none of the explicit Asset-object or
quote-asset handling a crypto version would need is used here.

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
    # Tunable parameters (all part of the codified rule set above.
    # Changing these is "refine", not "add discretion", per the
    # trading-backtest-methodology stress-testing workflow).
    # ------------------------------------------------------------------
    STOCK_UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

    MOM_LOOKBACK_DAYS = 30
    REBALANCE_EVERY_DAYS = 5

    VOL_LOOKBACK_DAYS = 20
    VOL_BASELINE_DAYS = 60
    VOL_SPIKE_MULTIPLIER = 1.8
    VOL_THROTTLE_FACTOR = 0.5

    GAP_THRESHOLD_PCT = 0.05
    GAP_THROTTLE_FACTOR = 0.3
    GAP_COOLDOWN_DAYS = 3

    MAX_ASSET_WEIGHT = 0.40       # no single name above 40% of portfolio
    MAX_TOTAL_EXPOSURE = 0.98     # stay close to fully invested
    MIN_REBALANCE_PCT = 0.02      # ignore rebalances smaller than 2% of NAV

    MAX_DRAWDOWN_PCT = 0.25       # 25% drawdown from peak trips the breaker
    DRAWDOWN_COOLDOWN_DAYS = 5    # trading days to stay flat after a trip

    # How many days of daily history to pull per iteration. Must cover the
    # longest lookback used anywhere below (VOL_BASELINE_DAYS=60) plus a
    # safety margin.
    _HISTORY_DAYS = max(MOM_LOOKBACK_DAYS, VOL_LOOKBACK_DAYS, VOL_BASELINE_DAYS) + 10

    # Per-symbol, per-iteration logging of what the data layer returned.
    # Useful when the strategy is not trading and you need to see whether
    # history is arriving at all. Very noisy, so it stays off by default.
    _DEBUG_SIGNALS = False

    # ------------------------------------------------------------------
    # Lifecycle: setup
    # ------------------------------------------------------------------
    def initialize(self):
        self.sleeptime = "1D"

        self.peak_portfolio_value = None
        self.cooldown_days_left = 0

        self.base_weights = {}
        self.gap_cooldown = {s: 0 for s in self.STOCK_UNIVERSE}
        self.day_index = 0
        self.force_rebalance = True

        self._last_logged_weights = None

        self.log_message(
            f"Strategy initialized. Universe={self.STOCK_UNIVERSE}, "
            f"sleeptime={self.sleeptime}, rebalance_every={self.REBALANCE_EVERY_DAYS}d"
        )

    # ------------------------------------------------------------------
    # Lifecycle: per-step decision making
    # ------------------------------------------------------------------
    def on_trading_iteration(self):
        portfolio_value = self.get_portfolio_value()
        if portfolio_value is None:
            self.log_message("Portfolio value unavailable this iteration, skipping.")
            return

        # -- Step 1: update the running peak and check the circuit breaker.
        if self.peak_portfolio_value is None or portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value

        drawdown = (
            (portfolio_value / self.peak_portfolio_value) - 1.0
            if self.peak_portfolio_value else 0.0
        )

        if self.cooldown_days_left > 0:
            self.cooldown_days_left -= 1
            self._flatten_and_reset()
            if self.cooldown_days_left == 0:
                # Reset the drawdown reference point to the recovery value,
                # not the stale pre-crash peak - otherwise cash sitting
                # idle never climbs back to the old peak and the very next
                # check would re-trigger the breaker permanently. Caught
                # during validation: this bug locked an earlier version
                # out of the market for good after its first real drawdown.
                self.peak_portfolio_value = portfolio_value
            self.log_message(
                f"[cooldown] {self.cooldown_days_left} day(s) left. "
                f"portfolio=${portfolio_value:,.2f} drawdown={drawdown:.1%}"
            )
            return

        if drawdown <= -self.MAX_DRAWDOWN_PCT:
            self._flatten_and_reset()
            self.cooldown_days_left = self.DRAWDOWN_COOLDOWN_DAYS
            self.log_message(
                f"[circuit-breaker] drawdown {drawdown:.1%} breached "
                f"-{self.MAX_DRAWDOWN_PCT:.0%} limit. Flattened and entering "
                f"{self.DRAWDOWN_COOLDOWN_DAYS}-day cooldown."
            )
            return

        # -- Step 2: pull daily bars for every symbol once, reused below
        #    for both the (possibly skipped) rebalance and the daily
        #    risk throttle.
        bars_by_symbol = {symbol: self._get_daily_bars(symbol) for symbol in self.STOCK_UNIVERSE}

        # -- Step 3: recompute target composition only every
        #    REBALANCE_EVERY_DAYS trading days (or right after a breaker
        #    reset, via force_rebalance).
        if self.force_rebalance or (self.day_index % self.REBALANCE_EVERY_DAYS == 0):
            pos_roc = {}
            for symbol in self.STOCK_UNIVERSE:
                df = bars_by_symbol[symbol]
                if df is None or len(df) < self.MOM_LOOKBACK_DAYS + 1:
                    continue
                roc = df["close"].iloc[-1] / df["close"].iloc[-1 - self.MOM_LOOKBACK_DAYS] - 1
                if roc == roc and roc > 0:  # NaN-safe
                    pos_roc[symbol] = roc
            self.base_weights = self._momentum_weights(pos_roc)
            self.force_rebalance = False
            if self._DEBUG_SIGNALS:
                self.log_message(f"[DEBUG] rebalanced. pos_roc={pos_roc} base_weights={self.base_weights}")
        self.day_index += 1

        # -- Step 4: daily risk throttle, applied every iteration
        #    regardless of the rebalance schedule, using only OHLCV data.
        target_weights = {}
        for symbol in self.STOCK_UNIVERSE:
            w = self.base_weights.get(symbol, 0.0)
            if w <= 0:
                target_weights[symbol] = 0.0
                continue

            throttle = 1.0
            df = bars_by_symbol[symbol]

            if df is not None and len(df) >= self.VOL_BASELINE_DAYS + 1:
                daily_ret = df["close"].pct_change()
                vol_short = daily_ret.tail(self.VOL_LOOKBACK_DAYS).std()
                vol_baseline = daily_ret.tail(self.VOL_BASELINE_DAYS).std()
                if vol_baseline and vol_baseline > 0:
                    spike_ratio = vol_short / vol_baseline
                    if spike_ratio == spike_ratio and spike_ratio > self.VOL_SPIKE_MULTIPLIER:
                        throttle = min(throttle, self.VOL_THROTTLE_FACTOR)

            if df is not None and len(df) >= 2:
                prev_close = df["close"].iloc[-2]
                today_open = df["open"].iloc[-1]
                if prev_close and prev_close > 0:
                    gap = (today_open - prev_close) / prev_close
                    if gap == gap and abs(gap) > self.GAP_THRESHOLD_PCT:
                        self.gap_cooldown[symbol] = self.GAP_COOLDOWN_DAYS

            if self.gap_cooldown.get(symbol, 0) > 0:
                throttle = min(throttle, self.GAP_THROTTLE_FACTOR)
                self.gap_cooldown[symbol] -= 1

            target_weights[symbol] = w * throttle

        # -- Step 5: translate target weights into orders.
        for symbol in self.STOCK_UNIVERSE:
            self._rebalance_to_weight(symbol, target_weights.get(symbol, 0.0), portfolio_value)

        rounded = {s: round(w, 4) for s, w in sorted(target_weights.items())}
        if rounded != self._last_logged_weights:
            self._last_logged_weights = rounded
            cash = self.get_cash()
            cash_text = f"${cash:,.2f}" if cash is not None else "unavailable"
            self.log_message(
                f"[Strategy] portfolio=${portfolio_value:,.2f} "
                f"cash={cash_text} drawdown={drawdown:.1%} weights={rounded}"
            )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def _get_daily_bars(self, symbol):
        bars = self.get_historical_prices(symbol, self._HISTORY_DAYS, "day")
        if bars is None or bars.df is None or bars.df.empty:
            if self._DEBUG_SIGNALS:
                self.log_message(
                    f"[DEBUG] {symbol}: get_historical_prices({self._HISTORY_DAYS} day) "
                    f"returned {'None' if bars is None else 'empty df'}"
                )
            return None
        return bars.df

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    def _momentum_weights(self, pos_roc):
        """
        weight_i = roc_i / sum(roc_j) among names with positive momentum,
        then capped at MAX_ASSET_WEIGHT per name and rescaled so the total
        never exceeds MAX_TOTAL_EXPOSURE. Momentum-proportional rather than
        inverse-volatility on purpose - see the module docstring.
        """
        if not pos_roc:
            return {}

        total = sum(pos_roc.values())
        if total <= 0:
            return {}

        weights = {s: v / total for s, v in pos_roc.items()}
        for _ in range(len(weights) + 1):
            over = {s: w for s, w in weights.items() if w > self.MAX_ASSET_WEIGHT}
            if not over:
                break
            capped_total = self.MAX_ASSET_WEIGHT * len(over)
            remaining = 1.0 - capped_total
            under = {s: w for s, w in weights.items() if s not in over}
            under_total = sum(under.values())
            for s in over:
                weights[s] = self.MAX_ASSET_WEIGHT
            if under_total > 0:
                for s in under:
                    weights[s] = (under[s] / under_total) * remaining

        return {s: w * self.MAX_TOTAL_EXPOSURE for s, w in weights.items()}

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
            return  # no tradable price this iteration, skip safely

        current_qty, current_value = self._current_position_value(symbol)
        target_value = target_weight * portfolio_value
        delta_value = target_value - current_value

        if abs(delta_value) < self.MIN_REBALANCE_PCT * max(portfolio_value, 1.0):
            return  # inside the no-trade band, avoid churn and fees on noise

        delta_qty = delta_value / price
        if delta_qty > 0:
            cash = self.get_cash()
            if cash is None:
                self.log_message(
                    f"get_cash() returned None while sizing a {symbol} buy. Skipping."
                )
                return
            affordable_qty = cash / price
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

    def _flatten_and_reset(self):
        for symbol in self.STOCK_UNIVERSE:
            qty, _ = self._current_position_value(symbol)
            if qty and qty > 0:
                order = self.create_order(symbol, qty, "sell")
                self.submit_order(order)
            self.gap_cooldown[symbol] = 0
        self.base_weights = {}
        self.force_rebalance = True
