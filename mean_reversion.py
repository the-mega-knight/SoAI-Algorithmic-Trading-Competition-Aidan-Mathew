"""
Short-horizon RSI mean-reversion strategy: buy the dip, sell the bounce.

Independent implementation from vector_validate.py's momentum-core
strategy, so the two can be tested side by side on the same data/regime
windows and compared honestly rather than one replacing the other.

Signal: 7-day RSI per stock.
  - Entry: RSI < entry_rsi (default 30, classic oversold threshold)
  - Exit:  RSI > exit_rsi (default 55 - "back to neutral", not the
           classic 70 overbought level, since the goal is to capture
           the bounce and get out, not ride a full oscillation cycle)

Safety exits (mean-reversion's failure mode is buying a stock that
keeps falling instead of bouncing, so these bound that risk):
  - stop_loss_pct: hard exit if position is down this much from entry
  - max_hold_days: forced exit if RSI hasn't recovered by then

Position sizing: equal weight across all currently-triggered names,
capped by max_asset_weight / max_total_exposure (can hold several
positions at once if multiple stocks dip together).

Risk management: keeps the same 25%-drawdown circuit breaker as the
momentum strategy (cheap tail insurance, rarely fires). Deliberately
omits the volatility/gap throttle used in vector_validate.py - that
throttle suppresses exposure exactly when volatility spikes, which is
exactly when this strategy wants to buy, so it would work against its
own entry logic.

Checked daily (not every N days) since RSI signals decay faster than
the momentum strategy's 30-day lookback.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json

DATA_DIR = Path(__file__).resolve().parent / "data"
UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

# Must exceed the largest lookback used below (rsi_period) with margin,
# so every parameter combination gets a fully-formed RSI before trading
# starts, and different parameter values still evaluate the exact same
# slice of days (fair sensitivity comparisons).
WARMUP_DAYS = 30

DEFAULT_PARAMS = dict(
    rsi_period=7,
    entry_rsi=30.0,
    exit_rsi=55.0,
    stop_loss_pct=0.08,
    max_hold_days=10,
    max_asset_weight=0.30,
    max_total_exposure=0.90,
    max_drawdown_pct=0.25,
    cooldown_days=5,
    fee_bps=2.0,
)


def load_daily(symbol):
    df = pd.read_csv(DATA_DIR / f"{symbol}_daily.csv", parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def compute_rsi(close, period):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)  # no losses in window -> RSI 100
    return rsi


def run_backtest(params, start_cash=1_000_000, verbose=False, start_date=None, end_date=None):
    daily = {s: load_daily(s) for s in UNIVERSE}
    common_index = None
    for s in UNIVERSE:
        idx = daily[s].index
        common_index = idx if common_index is None else common_index.intersection(idx)
    common_index = common_index.sort_values()

    sig = {}
    for s in UNIVERSE:
        close = daily[s]["close"].reindex(common_index)
        rsi = compute_rsi(close, params["rsi_period"])
        sig[s] = pd.DataFrame({"close": close, "rsi": rsi})

    cash = start_cash
    qty = {s: 0.0 for s in UNIVERSE}
    entry_price = {s: None for s in UNIVERSE}
    hold_days = {s: 0 for s in UNIVERSE}
    equity_curve = []
    peak = start_cash
    cooldown = 0
    trade_count = 0
    fee_rate = params["fee_bps"] / 10_000.0

    dates = common_index[WARMUP_DAYS:]
    if start_date is not None:
        dates = dates[dates >= pd.Timestamp(start_date)]
    if end_date is not None:
        dates = dates[dates <= pd.Timestamp(end_date)]

    def _sell(s, price):
        nonlocal cash, trade_count
        if qty[s] > 0:
            proceeds = qty[s] * price
            fee = proceeds * fee_rate
            cash += (proceeds - fee)
            trade_count += 1
        qty[s] = 0.0
        entry_price[s] = None
        hold_days[s] = 0

    def _flatten_and_reset():
        for s in UNIVERSE:
            _sell(s, prices[s])

    for dt in dates:
        prices = {s: sig[s].loc[dt, "close"] for s in UNIVERSE}
        port_value = cash + sum(qty[s] * prices[s] for s in UNIVERSE)
        if port_value > peak:
            peak = port_value
        drawdown = port_value / peak - 1.0 if peak else 0.0

        if cooldown > 0:
            cooldown -= 1
            _flatten_and_reset()
            if cooldown == 0:
                peak = port_value
            equity_curve.append((dt, cash))
            continue

        if drawdown <= -params["max_drawdown_pct"]:
            _flatten_and_reset()
            cooldown = params["cooldown_days"]
            equity_curve.append((dt, cash))
            continue

        # Manage existing positions first: stop-loss, RSI recovery exit,
        # max holding period.
        for s in UNIVERSE:
            if qty[s] <= 0:
                continue
            price = prices[s]
            hold_days[s] += 1
            rsi = sig[s].loc[dt, "rsi"]
            loss_pct = (price / entry_price[s]) - 1.0 if entry_price[s] else 0.0
            should_exit = (
                loss_pct <= -params["stop_loss_pct"]
                or (rsi == rsi and rsi > params["exit_rsi"])
                or hold_days[s] >= params["max_hold_days"]
            )
            if should_exit:
                _sell(s, price)

        # New entries: any name currently oversold and not already held.
        candidates = [
            s for s in UNIVERSE
            if qty[s] == 0
            and sig[s].loc[dt, "rsi"] == sig[s].loc[dt, "rsi"]
            and sig[s].loc[dt, "rsi"] < params["entry_rsi"]
        ]

        if candidates:
            port_value = cash + sum(qty[s] * prices[s] for s in UNIVERSE)
            already_held = [s for s in UNIVERSE if qty[s] > 0]
            n_total = len(already_held) + len(candidates)
            target_weight = min(params["max_asset_weight"], params["max_total_exposure"] / max(n_total, 1))
            for s in candidates:
                price = prices[s]
                target_value = target_weight * port_value
                affordable = cash / price
                buy_qty = min(target_value / price, affordable)
                if buy_qty <= 0:
                    continue
                cost = buy_qty * price
                fee = cost * fee_rate
                if cost + fee > cash:
                    buy_qty = cash / (price * (1 + fee_rate))
                    cost = buy_qty * price
                    fee = cost * fee_rate
                if buy_qty <= 0:
                    continue
                cash -= (cost + fee)
                qty[s] += buy_qty
                entry_price[s] = price
                hold_days[s] = 0
                trade_count += 1

        port_value = cash + sum(qty[s] * prices[s] for s in UNIVERSE)
        equity_curve.append((dt, port_value))

    ec = pd.Series({d: v for d, v in equity_curve}).sort_index()
    returns = ec.pct_change().dropna()

    ann_factor = 252
    sharpe = np.sqrt(ann_factor) * returns.mean() / returns.std() if returns.std() > 0 else np.nan
    downside = returns[returns < 0]
    sortino = np.sqrt(ann_factor) * returns.mean() / downside.std() if len(downside) and downside.std() > 0 else np.nan
    running_max = ec.cummax()
    dd_series = ec / running_max - 1
    max_dd = dd_series.min()
    terminal_return = ec.iloc[-1] / ec.iloc[0] - 1 if len(ec) else np.nan

    result = {
        "terminal_return": float(terminal_return),
        "sharpe": float(sharpe) if sharpe == sharpe else None,
        "sortino": float(sortino) if sortino == sortino else None,
        "max_drawdown": float(max_dd),
        "trade_count": trade_count,
        "num_days": len(ec),
        "final_value": float(ec.iloc[-1]) if len(ec) else None,
    }
    if verbose:
        print(json.dumps(result, indent=2))
    return result, ec


if __name__ == "__main__":
    print("=== Mean-reversion baseline (full history) ===")
    result, ec = run_backtest(DEFAULT_PARAMS, verbose=True)

    print("\n=== Buy & hold benchmark (equal weight, no trading) ===")
    for s in UNIVERSE:
        d = load_daily(s)
        close = d["close"].reindex(ec.index).dropna()
        bh_return = close.iloc[-1] / close.iloc[0] - 1
        print(f"{s}: {bh_return:.2%} buy & hold over {len(close)} days")
