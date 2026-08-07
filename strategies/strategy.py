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

Why momentum, and not mean-reversion or daily binary trend-following
------------------------------------------------------------------------
Two earlier versions of this strategy (crypto momentum, then daily SMA
crossover on this same equity universe) were both validated against real
historical data and both underperformed simple buy-and-hold by a wide
margin. The common cause: daily binary entry/exit trading generates heavy
turnover (500+ trades over ~450 days), most of it whipsaw on short-term
noise rather than genuine trend changes, and the fees on that turnover ate
a large share of the return. Every version tested pointed the same
direction: staying invested consistently beat trading in and out.

A short-horizon RSI mean-reversion variant was also built and tested
side by side with this one (see mean_reversion.py / regime_test_mr.py).
It won convincingly in the 2022 bear-market regime and roughly matched
buy-and-hold in the most recent real month, but lost badly to both
buy-and-hold and this momentum strategy in every trending regime
(2018-19, 2020, 2021, 2023-24), since a dip-buying strategy mostly sits
out of a market that never dips. Two pieces of outside evidence point the
same direction as that result: academic work on cryptocurrency return
reversal (Zaremba et al. 2021) finds that reversal is a small/illiquid-
asset phenomenon and the most liquid ~2% of assets show momentum instead;
and work on trading-strategy capacity (Bonelli, Landier, Simon, Thesmar
2019) shows that faster-turning signals give back more of their gross
edge to trading costs than slow-moving ones, independent of scale. Both
point toward momentum being the structurally sounder fit for five
already-hyper-liquid, large-cap names, so momentum was kept as the
strategy actually submitted.

Rules (zero discretion)
------------------------
Universe:         AAPL, MSFT, GOOGL, AMZN, NVDA (see STOCK_UNIVERSE)
Cadence:          checked once per day (self.sleeptime = "1D"), but the
                  target allocation is only recomputed every
                  REBALANCE_EVERY_DAYS trading days
Entry signal:     30-day rate of change > 0 (MOM_LOOKBACK_DAYS) opens/holds
                  a long; ROC < 0 opens/holds a short (see "Long/short
                  variant" below). A flat ROC (exactly 0 or NaN) holds no
                  position either direction.
Weighting:        proportional to each qualifying name's own |ROC| (not
                  inverse-volatility - see below), capped at
                  MAX_ASSET_WEIGHT per name and MAX_TOTAL_EXPOSURE total,
                  applied independently to the long book and the short
                  book (see below).

Long/short variant (v4): the original strategy only ever went long or
                  flat. This version extends _momentum_weights to also
                  size short positions for negative-ROC names,
                  symmetrically to how positive-ROC names are sized long
                  - same per-name cap (MAX_ASSET_WEIGHT), same total-book
                  cap (MAX_TOTAL_EXPOSURE), just applied to the short book
                  independently. That means gross exposure (long + short)
                  can now reach up to ~2x MAX_TOTAL_EXPOSURE if the
                  universe splits evenly between positive- and
                  negative-momentum names, versus the original's max of
                  1x. This is a materially larger risk profile than the
                  long-only version and has not been validated against
                  real market data as of this commit (see
                  README/handover notes) - treat it as a research variant
                  until backtested.
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
                  Loosened (v3.2) relative to the original calibration
                  after regime testing showed the tighter throttle was
                  giving back meaningful bull-market upside for very
                  little extra 2022-style crash protection - the
                  portfolio-level circuit breaker below, not this daily
                  throttle, was doing the real work in that test.
Known-event derisk: a small, symmetric, non-predictive exception to the
                  "no calendar data" rule above: NVDA's own publicly
                  scheduled earnings date is hardcoded (EARNINGS_RISK_DATES)
                  and that name's weight is proactively capped on that one
                  date. This does not bet on the report's direction (beat
                  or miss) - it only shrinks the position size ahead of a
                  known binary, single-stock event, the same way a human
                  trader would size down before a coin flip with unusually
                  large stakes. If the date list is ever empty or stale,
                  the strategy trades exactly as if this section did not
                  exist.
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

import datetime as _dt

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
    VOL_SPIKE_MULTIPLIER = 2.3    # v3.2: loosened from 1.8, see module docstring
    VOL_THROTTLE_FACTOR = 0.7     # v3.2: loosened from 0.5

    GAP_THRESHOLD_PCT = 0.05
    GAP_THROTTLE_FACTOR = 0.5     # v3.2: loosened from 0.3
    GAP_COOLDOWN_DAYS = 3

    MAX_ASSET_WEIGHT = 0.50       # v3.2: raised from 0.40
    MAX_TOTAL_EXPOSURE = 1.00     # v3.2: raised from 0.98
    MIN_REBALANCE_PCT = 0.02      # ignore rebalances smaller than 2% of NAV

    MAX_DRAWDOWN_PCT = 0.25       # 25% drawdown from peak trips the breaker
    DRAWDOWN_COOLDOWN_DAYS = 5    # trading days to stay flat after a trip

    # Publicly scheduled, single-stock binary events known well in advance
    # (e.g. earnings calls). Symmetric, non-predictive position-size cap on
    # those specific dates only - see "Known-event derisk" in the module
    # docstring. Dates are the trading day of the announcement itself
    # (NVDA reports after the close on this date, so the cap applies to
    # the day the position is held into the report, not the day after).
    EARNINGS_RISK_DATES = {
        "NVDA": ["2026-08-26"],
    }
    EARNINGS_DERISK_FACTOR = 0.5  # halve the name's weight on that date

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
            neg_roc = {}
            for symbol in self.STOCK_UNIVERSE:
                df = bars_by_symbol[symbol]
                if df is None or len(df) < self.MOM_LOOKBACK_DAYS + 1:
                    continue
                roc = df["close"].iloc[-1] / df["close"].iloc[-1 - self.MOM_LOOKBACK_DAYS] - 1
                if roc != roc:  # NaN-safe
                    continue
                if roc > 0:
                    pos_roc[symbol] = roc
                elif roc < 0:
                    neg_roc[symbol] = -roc  # magnitude, sign re-applied in _momentum_weights
            self.base_weights = self._momentum_weights(pos_roc, neg_roc)
            self.force_rebalance = False
            if self._DEBUG_SIGNALS:
                self.log_message(
                    f"[DEBUG] rebalanced. pos_roc={pos_roc} neg_roc={neg_roc} "
                    f"base_weights={self.base_weights}"
                )
        self.day_index += 1

        # -- Step 4: daily risk throttle, applied every iteration
        #    regardless of the rebalance schedule, using only OHLCV data.
        today_str = self._today_str()
        target_weights = {}
        for symbol in self.STOCK_UNIVERSE:
            w = self.base_weights.get(symbol, 0.0)
            if w == 0:
                target_weights[symbol] = 0.0
                continue

            # throttle is a magnitude multiplier in [0, 1] applied to w
            # regardless of sign, so it shrinks both long and short
            # positions toward flat under the same vol/gap risk signals -
            # it never flips a position's direction.
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

            # Known-event derisk: symmetric size cap on a publicly
            # scheduled binary event date (see module docstring). Not a
            # directional bet - just less size going into a known
            # single-stock risk day.
            if today_str in self.EARNINGS_RISK_DATES.get(symbol, []):
                throttle = min(throttle, self.EARNINGS_DERISK_FACTOR)

            target_weights[symbol] = w * throttle

        # -- Step 5: translate target weights into orders.
        for symbol in self.STOCK_UNIVERSE:
            self._rebalance_to_weight(symbol, target_weights.get(symbol, 0.0), portfolio_value)

        # -- Step 6: indicator logging for the tearsheet's "indicators"
        #    chart. Lumibot leaves that chart empty unless add_line() is
        #    called during the run - this is purely observational and has
        #    zero effect on trading decisions or order sizing above.
        #    Portfolio Value is dollars in the millions; drawdown/weights
        #    are 0-100. Those were originally all on one implicit
        #    "default_plot" and shared a y-axis, which visually flattened
        #    the weight lines to zero next to the much larger portfolio
        #    value line. Splitting them into two named plots via
        #    plot_name= fixes that - each gets its own y-axis/subplot.
        self.add_line("Portfolio Value", portfolio_value, plot_name="Portfolio Value ($)")
        self.add_line("Drawdown %", drawdown * 100, plot_name="Drawdown & Weights (%)")
        for symbol in self.STOCK_UNIVERSE:
            self.add_line(
                f"{symbol} Weight %",
                target_weights.get(symbol, 0.0) * 100,
                plot_name="Drawdown & Weights (%)",
            )

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

    def _today_str(self):
        """Current simulated/live trading date as 'YYYY-MM-DD', used only
        for the known-event derisk check above."""
        dt = self.get_datetime()
        if dt is None:
            return ""
        if isinstance(dt, _dt.datetime):
            return dt.date().isoformat()
        if isinstance(dt, _dt.date):
            return dt.isoformat()
        return str(dt)[:10]

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    def _momentum_weights(self, pos_roc, neg_roc=None):
        """
        Long book: weight_i = roc_i / sum(roc_j) among positive-momentum
        names, capped at MAX_ASSET_WEIGHT per name and rescaled so the
        book total never exceeds MAX_TOTAL_EXPOSURE.

        Short book (v4, long/short variant): the same algorithm, applied
        independently to negative-momentum names using |roc|, then negated
        - same per-name and per-book caps as the long side, by design (see
        "Long/short variant" in the module docstring). Momentum-
        proportional rather than inverse-volatility on purpose - see the
        module docstring.

        Returns a single dict keyed by symbol: positive weight = long,
        negative weight = short. A symbol can never appear in both
        pos_roc and neg_roc (a name's ROC has one sign), so there's no
        collision to resolve.
        """
        long_weights = self._capped_weights(pos_roc)
        short_weights = self._capped_weights(neg_roc or {})
        combined = dict(long_weights)
        for s, w in short_weights.items():
            combined[s] = -w
        return combined

    def _capped_weights(self, roc_magnitudes):
        """Shared sizing algorithm for one book (long or short): weight_i
        = roc_i / sum(roc_j), capped at MAX_ASSET_WEIGHT per name and
        rescaled so the book total never exceeds MAX_TOTAL_EXPOSURE.
        roc_magnitudes must be non-negative (sign is applied by the
        caller)."""
        if not roc_magnitudes:
            return {}

        total = sum(roc_magnitudes.values())
        if total <= 0:
            return {}

        weights = {s: v / total for s, v in roc_magnitudes.items()}
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
        """Move symbol's position toward target_weight * portfolio_value.
        target_weight (and therefore current/target position value) can
        be negative in the long/short variant, meaning a short position -
        see _execute_qty_delta for how a change that crosses through zero
        (e.g. covering a short and going long in the same rebalance) is
        split into the correct pair of orders."""
        price = self.get_last_price(symbol)
        if price is None or price <= 0:
            return  # no tradable price this iteration, skip safely

        current_qty, current_value = self._current_position_value(symbol)
        target_value = target_weight * portfolio_value
        delta_value = target_value - current_value

        if abs(delta_value) < self.MIN_REBALANCE_PCT * max(portfolio_value, 1.0):
            return  # inside the no-trade band, avoid churn and fees on noise

        delta_qty = delta_value / price
        self._execute_qty_delta(symbol, current_qty, delta_qty, price)

    def _execute_qty_delta(self, symbol, current_qty, delta_qty, price):
        """Apply a signed change in share quantity to a position that may
        currently be long, short, or flat, splitting the change into up
        to two orders when it crosses through zero (e.g. current_qty=-100
        short, delta_qty=+150 -> buy_to_cover 100, then buy 50)."""
        current_qty = current_qty or 0.0
        remaining = delta_qty

        if remaining > 0:
            # Increasing net quantity: cover any existing short first,
            # then use whatever's left to open/add to a long.
            if current_qty < 0:
                cover_qty = min(remaining, -current_qty)
                if cover_qty > 0:
                    order = self.create_order(symbol, cover_qty, "buy_to_cover")
                    self.submit_order(order)
                    remaining -= cover_qty
            if remaining > 0:
                cash = self.get_cash()
                if cash is None:
                    self.log_message(
                        f"get_cash() returned None while sizing a {symbol} buy. Skipping."
                    )
                    return
                affordable_qty = cash / price
                buy_qty = min(remaining, affordable_qty)
                if buy_qty > 0:
                    order = self.create_order(symbol, buy_qty, "buy")
                    self.submit_order(order)

        elif remaining < 0:
            # Decreasing net quantity: sell off any existing long first,
            # then use whatever's left to open/add to a short.
            need = -remaining
            if current_qty > 0:
                sell_qty = min(need, current_qty)
                if sell_qty > 0:
                    order = self.create_order(symbol, sell_qty, "sell")
                    self.submit_order(order)
                    need -= sell_qty
            if need > 0:
                order = self.create_order(symbol, need, "sell_short")
                self.submit_order(order)

    def _flatten_and_reset(self):
        for symbol in self.STOCK_UNIVERSE:
            qty, _ = self._current_position_value(symbol)
            if qty and qty > 0:
                order = self.create_order(symbol, qty, "sell")
                self.submit_order(order)
            elif qty and qty < 0:
                # v4 (long/short variant): the circuit breaker must also
                # cover shorts, not just sell longs, or it would leave a
                # short position running unmanaged through the exact
                # drawdown event it exists to protect against.
                order = self.create_order(symbol, abs(qty), "buy_to_cover")
                self.submit_order(order)
            self.gap_cooldown[symbol] = 0
        self.base_weights = {}
        self.force_rebalance = True
