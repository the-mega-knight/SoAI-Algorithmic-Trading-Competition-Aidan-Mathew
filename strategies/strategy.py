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

Every decision below is mechanical, with no discretionary judgment calls,
so the same inputs always produce the same output. That is a requirement
for both honest backtesting and for passing the official verification run.

Rules (zero discretion)
------------------------
Universe:        BTC, ETH, SOL (spot, USD-quoted; see CRYPTO_UNIVERSE)
Cadence:         once per day (self.sleeptime = "1D")
Entry signal:    SMA(10) > SMA(20)  AND  10-day ROC > 0
Exit signal:     SMA(10) <= SMA(20)  (trend break, flatten immediately)
Position sizing: inverse-volatility weighting across assets with an active
                 entry signal, capped at MAX_ASSET_WEIGHT per asset and
                 MAX_TOTAL_EXPOSURE of the portfolio in total
Risk control:    portfolio-level drawdown circuit breaker. If portfolio
                 value draws down more than MAX_DRAWDOWN_PCT from its
                 running peak, flatten everything and pause new entries
                 for DRAWDOWN_COOLDOWN_DAYS trading days
Rebalance rule:  only trade an asset when the target position differs from
                 the current one by more than MIN_REBALANCE_PCT of
                 portfolio value, to avoid churning on noise

Assets, quotes and market hours
-------------------------------
Lumibot resolves a bare symbol string such as "BTC" to a stock asset
quoted in forex USD. This strategy trades crypto, so initialize() builds
Asset objects with asset_type CRYPTO and passes them, along with the
account quote asset, to every data and order call. The quote asset is
whatever the runner configured (backtest.py hands one to run_backtest);
the fallback is USD as a crypto asset. The asset and quote used here have
to match the ones the price data was registered under, otherwise the
history lookups return nothing and the strategy sits flat forever.

Crypto trades continuously, so initialize() sets the market to 24/7.
Left on the default NASDAQ calendar, the broker reports the market closed
overnight and at weekends and the trading iteration is skipped.

See the project README for the full writeup, and
`trading-backtest-methodology` / `trading-position-sizer` /
`trading-strategy-signal-toolkit` for the methodology this was built with.

The official execution environment imports the class defined here, so:

* Keep the class name ``Strategy``.
* Keep this file at ``strategies/strategy.py``.
* Keep the import path ``from strategies.strategy import Strategy``.
"""

from lumibot.strategies import Strategy as _LumibotStrategy
from lumibot.entities import Asset


class Strategy(_LumibotStrategy):
    # ------------------------------------------------------------------
    # Tunable parameters (all part of the codified rule set above.
    # Changing these is "refine", not "add discretion", per the
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
    # SLOW_SMA_DAYS plus a safety margin, converted to minutes.
    _HISTORY_DAYS = max(SLOW_SMA_DAYS, MOMENTUM_LOOKBACK_DAYS, VOL_LOOKBACK_DAYS) + 10

    # Per-symbol, per-iteration logging of what the data layer returned.
    # Useful when the strategy is not trading and you need to see whether
    # history is arriving at all. Very noisy, so it stays off by default.
    _DEBUG_SIGNALS = False

    # ------------------------------------------------------------------
    # Lifecycle: setup
    # ------------------------------------------------------------------
    def initialize(self):
        self.sleeptime = "1D"

        # Crypto has no session boundaries. On the default calendar the
        # broker reports the market closed and the iteration is skipped.
        try:
            self.set_market("24/7")
        except Exception as exc:  # pragma: no cover - depends on the runner
            self.log_message(
                f"Could not set the market to 24/7 ({exc}). Continuing on the "
                f"runner default, which may skip iterations outside session hours."
            )

        # Read the account quote asset rather than assigning one. Lumibot
        # sets its own _quote_asset during construction and registers the
        # cash position against it, so replacing it here would strand that
        # cash position and make get_cash() return None.
        self._usd_quote = getattr(self, "_quote_asset", None) or Asset(
            symbol="USD", asset_type=Asset.AssetType.CRYPTO
        )

        # Explicit crypto assets. A bare "BTC" string resolves to a stock.
        self._trade_assets = {
            symbol: Asset(symbol=symbol, asset_type=Asset.AssetType.CRYPTO)
            for symbol in self.CRYPTO_UNIVERSE
        }

        # Running peak portfolio value, for the drawdown circuit breaker.
        self.peak_portfolio_value = None
        # Trading days remaining in the post-breaker cooldown (0 = not in cooldown).
        self.cooldown_days_left = 0
        # Last set of target weights written to the log, so the summary
        # line only prints when the target allocation actually moves.
        self._last_logged_weights = None

        self.log_message(
            f"Strategy initialized. Universe={self.CRYPTO_UNIVERSE}, "
            f"sleeptime={self.sleeptime}, quote={self._usd_quote}"
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
                if self._DEBUG_SIGNALS:
                    got = 0 if daily is None else len(daily)
                    self.log_message(
                        f"[DEBUG] {symbol}: skipping, only {got} daily bars "
                        f"after resample (need >= {self.SLOW_SMA_DAYS + 1}). "
                        f"daily_is_none={daily is None}"
                    )
                continue  # not enough history yet, sit this one out
            signals[symbol] = self._compute_signal(daily)

        # -- Step 3: qualify assets with an active entry signal.
        qualifying = {s: v for s, v in signals.items() if v["entry"]}

        if self._DEBUG_SIGNALS and signals and not qualifying:
            self.log_message(
                "[DEBUG] signals computed but none qualified: "
                + ", ".join(
                    f"{s}(sma_fast={v['sma_fast']:.2f}, sma_slow={v['sma_slow']:.2f}, "
                    f"roc={v['roc']:.4f})"
                    for s, v in signals.items()
                )
            )

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

        # The iteration summary is only worth a log line when the target
        # allocation moved. Printing it every iteration buries the
        # interesting entries and exits in thousands of identical rows.
        rounded = {s: round(w, 4) for s, w in sorted(target_weights.items())}
        if rounded != self._last_logged_weights:
            self._last_logged_weights = rounded
            cash = self.get_cash()
            cash_text = f"${cash:,.2f}" if cash is not None else "unavailable"
            self.log_message(
                f"[Strategy] portfolio=${portfolio_value:,.2f} "
                f"cash={cash_text} drawdown={drawdown:.1%} "
                f"qualifying={list(qualifying.keys())} weights={rounded}"
            )

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------
    def _get_daily_bars(self, symbol):
        """
        Pull minute bars and resample to daily OHLC here, rather than
        relying on the backtesting engine's own timestep resampling. This
        keeps behavior identical across the local Pandas/CSV backtest and
        the official minute-bar-only live feed.
        """
        asset = self._trade_assets[symbol]
        length_minutes = self._HISTORY_DAYS * 24 * 60
        bars = self.get_historical_prices(
            asset, length_minutes, "minute", quote=self._usd_quote
        )
        if bars is None or bars.df is None or bars.df.empty:
            if self._DEBUG_SIGNALS:
                self.log_message(
                    f"[DEBUG] {symbol}: get_historical_prices({length_minutes} min) "
                    f"returned {'None' if bars is None else 'empty df'}"
                )
            return None

        df = bars.df
        daily = df.resample("1D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        if self._DEBUG_SIGNALS:
            self.log_message(
                f"[DEBUG] {symbol}: pulled {len(df)} minute bars "
                f"({df.index.min()} to {df.index.max()}), resampled to "
                f"{len(daily)} daily bars"
            )
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
        asset = self._trade_assets[symbol]
        position = self.get_position(asset)
        if position is None or position.quantity in (None, 0):
            return 0.0, 0.0
        price = self.get_last_price(asset, quote=self._usd_quote)
        if price is None:
            return float(position.quantity), 0.0
        return float(position.quantity), float(position.quantity) * float(price)

    def _rebalance_to_weight(self, symbol, target_weight, portfolio_value):
        asset = self._trade_assets[symbol]
        price = self.get_last_price(asset, quote=self._usd_quote)
        if price is None or price <= 0:
            return  # no tradable price this iteration, skip safely

        current_qty, current_value = self._current_position_value(symbol)
        target_value = target_weight * portfolio_value
        delta_value = target_value - current_value

        if abs(delta_value) < self.MIN_REBALANCE_PCT * max(portfolio_value, 1.0):
            return  # inside the no-trade band, avoid churn and fees on noise

        delta_qty = delta_value / price
        if delta_qty > 0:
            # Buying: never spend more than available cash. A None here
            # means the broker could not report a balance, which is worth
            # surfacing rather than silently sizing the order at zero.
            cash = self.get_cash()
            if cash is None:
                self.log_message(
                    f"get_cash() returned None while sizing a {symbol} buy. "
                    f"Skipping this order. Check that the account quote asset "
                    f"matches the quote the price data was registered under."
                )
                return
            affordable_qty = cash / price
            delta_qty = min(delta_qty, affordable_qty)
            if delta_qty <= 0:
                return
            order = self.create_order(asset, delta_qty, "buy", quote=self._usd_quote)
            self.submit_order(order)
        else:
            sell_qty = min(abs(delta_qty), current_qty) if current_qty else 0
            if sell_qty <= 0:
                return
            order = self.create_order(asset, sell_qty, "sell", quote=self._usd_quote)
            self.submit_order(order)

    def _flatten_all(self):
        for symbol in self.CRYPTO_UNIVERSE:
            asset = self._trade_assets[symbol]
            qty, _ = self._current_position_value(symbol)
            if qty and qty > 0:
                order = self.create_order(asset, qty, "sell", quote=self._usd_quote)
                self.submit_order(order)
