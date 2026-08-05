"""
Independent vectorized re-implementation of the strategy's decision rules,
used to validate + stress-test the logic (per trading-backtest-methodology)
without depending on a full Lumibot install in this sandbox.

v2 changes from the original momentum strategy, per the underperformance-vs-
buy-and-hold diagnosis:
  - Added a 100-day regime filter: only enter when price is above its own
    100-day SMA, to avoid trading noise in non-trending stretches.
  - Added a 2-day exit confirmation: a single-day trend break no longer
    flattens the position immediately, cutting whipsaw round-trips.
  - Switched sizing from inverse-volatility to momentum-weighted (weight
    proportional to recent ROC), so strong trends get more capital instead
    of being penalized for their own volatility.
  - Raised MAX_ASSET_WEIGHT and MAX_TOTAL_EXPOSURE since the two filters
    above should reduce the need for such conservative caps.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import itertools
import json

DATA_DIR = Path(__file__).resolve().parent / "data"
UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

DEFAULT_PARAMS = dict(
    fast=15, slow=25, mom=10, vol_lookback=20, regime_days=100,
    exit_confirm_days=2,
    max_asset_weight=0.65, max_total_exposure=0.98,
    min_rebalance_pct=0.02, max_drawdown_pct=0.25, cooldown_days=5,
    fee_bps=2.0,  # matches the official 2bps fee stated on the competition site
)


def load_daily(symbol):
    df = pd.read_csv(DATA_DIR / f"{symbol}_daily.csv", parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def momentum_weights(qualifying_roc, max_asset_weight, max_total_exposure):
    if not qualifying_roc:
        return {}
    scores = {s: max(v, 1e-6) for s, v in qualifying_roc.items()}
    total = sum(scores.values())
    if total <= 0:
        return {}
    weights = {s: sc / total for s, sc in scores.items()}
    for _ in range(len(weights) + 1):
        over = {s: w for s, w in weights.items() if w > max_asset_weight}
        if not over:
            break
        capped_total = max_asset_weight * len(over)
        remaining = 1.0 - capped_total
        under = {s: w for s, w in weights.items() if s not in over}
        under_total = sum(under.values())
        for s in over:
            weights[s] = max_asset_weight
        if under_total > 0:
            for s in under:
                weights[s] = (under[s] / under_total) * remaining
    return {s: w * max_total_exposure for s, w in weights.items()}


def run_backtest(params, start_cash=1_000_000, verbose=False):
    daily = {s: load_daily(s) for s in UNIVERSE}
    common_index = None
    for s in UNIVERSE:
        idx = daily[s].index
        common_index = idx if common_index is None else common_index.intersection(idx)
    common_index = common_index.sort_values()

    fast, slow, mom = params["fast"], params["slow"], params["mom"]
    vol_lb, regime_days = params["vol_lookback"], params["regime_days"]

    sig = {}
    for s in UNIVERSE:
        close = daily[s]["close"].reindex(common_index)
        sma_fast = close.rolling(fast).mean()
        sma_slow = close.rolling(slow).mean()
        sma_regime = close.rolling(regime_days).mean()
        roc = close.pct_change(periods=mom)
        daily_ret = close.pct_change()
        vol = daily_ret.rolling(vol_lb).std()

        entry_raw = (sma_fast > sma_slow) & (roc > 0) & (close > sma_regime)
        exit_raw = (sma_fast <= sma_slow)

        sig[s] = pd.DataFrame({
            "close": close, "entry_raw": entry_raw, "exit_raw": exit_raw,
            "roc": roc, "vol": vol,
        })

    cash = start_cash
    qty = {s: 0.0 for s in UNIVERSE}
    in_position = {s: False for s in UNIVERSE}
    exit_streak = {s: 0 for s in UNIVERSE}
    equity_curve = []
    peak = start_cash
    cooldown = 0
    trade_count = 0
    fee_rate = params["fee_bps"] / 10_000.0

    warmup = max(fast, slow, mom, vol_lb, regime_days) + 1
    dates = common_index[warmup:]

    def _flatten_and_reset():
        nonlocal cash, trade_count
        for s in UNIVERSE:
            if qty[s] > 0:
                cash += qty[s] * prices[s] * (1 - fee_rate)
                trade_count += 1
                qty[s] = 0.0
            in_position[s] = False
            exit_streak[s] = 0

    for dt in dates:
        prices = {s: sig[s].loc[dt, "close"] for s in UNIVERSE}
        port_value = cash + sum(qty[s] * prices[s] for s in UNIVERSE)
        if port_value > peak:
            peak = port_value
        drawdown = port_value / peak - 1.0 if peak else 0.0

        if cooldown > 0:
            cooldown -= 1
            _flatten_and_reset()
            equity_curve.append((dt, cash))
            continue

        if drawdown <= -params["max_drawdown_pct"]:
            _flatten_and_reset()
            cooldown = params["cooldown_days"]
            equity_curve.append((dt, cash))
            continue

        # -- hysteresis: update in_position / exit_streak per asset
        for s in UNIVERSE:
            row = sig[s].loc[dt]
            if pd.isna(row["entry_raw"]) or pd.isna(row["exit_raw"]):
                continue  # not enough history yet for this asset
            if not in_position[s]:
                if bool(row["entry_raw"]):
                    in_position[s] = True
                    exit_streak[s] = 0
            else:
                if bool(row["exit_raw"]):
                    exit_streak[s] += 1
                else:
                    exit_streak[s] = 0
                if exit_streak[s] >= params["exit_confirm_days"]:
                    in_position[s] = False
                    exit_streak[s] = 0

        qualifying_roc = {
            s: sig[s].loc[dt, "roc"] for s in UNIVERSE
            if in_position[s] and sig[s].loc[dt, "roc"] == sig[s].loc[dt, "roc"]
        }
        weights = momentum_weights(qualifying_roc, params["max_asset_weight"], params["max_total_exposure"])

        for s in UNIVERSE:
            target_w = weights.get(s, 0.0)
            price = prices[s]
            current_value = qty[s] * price
            target_value = target_w * port_value
            delta_value = target_value - current_value
            if abs(delta_value) < params["min_rebalance_pct"] * max(port_value, 1.0):
                continue
            delta_qty = delta_value / price
            if delta_qty > 0:
                affordable = cash / price
                delta_qty = min(delta_qty, affordable)
                if delta_qty <= 0:
                    continue
                cost = delta_qty * price
                fee = cost * fee_rate
                if cost + fee > cash:
                    delta_qty = cash / (price * (1 + fee_rate))
                    cost = delta_qty * price
                    fee = cost * fee_rate
                cash -= (cost + fee)
                qty[s] += delta_qty
                trade_count += 1
            else:
                sell_qty = min(abs(delta_qty), qty[s])
                if sell_qty <= 0:
                    continue
                proceeds = sell_qty * price
                fee = proceeds * fee_rate
                cash += (proceeds - fee)
                qty[s] -= sell_qty
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
    print("=== Baseline run (v2: regime filter + exit confirm + momentum weighting) ===")
    result, ec = run_backtest(DEFAULT_PARAMS, verbose=True)

    print("\n=== Parameter sensitivity grid (fast/slow SMA, tighter band around 15/25) ===")
    grid_results = []
    for fast, slow in itertools.product([10, 12, 13, 14, 15, 16, 17, 18, 20], [20, 22, 24, 25, 26, 28, 30, 32, 35]):
        if fast >= slow:
            continue
        p = dict(DEFAULT_PARAMS)
        p["fast"], p["slow"] = fast, slow
        r, _ = run_backtest(p)
        grid_results.append({"fast": fast, "slow": slow, **r})

    grid_df = pd.DataFrame(grid_results).sort_values("sharpe", ascending=False)
    pd.set_option("display.width", 120)
    print(grid_df.to_string(index=False))

    print("\n=== Sensitivity: exit_confirm_days (1, 2, 3) ===")
    for d in [1, 2, 3]:
        p = dict(DEFAULT_PARAMS)
        p["exit_confirm_days"] = d
        r, _ = run_backtest(p)
        print(f"exit_confirm_days={d}: {json.dumps(r)}")

    print("\n=== Sensitivity: regime_days (50, 100, 150) ===")
    for d in [50, 100, 150]:
        p = dict(DEFAULT_PARAMS)
        p["regime_days"] = d
        r, _ = run_backtest(p)
        print(f"regime_days={d}: {json.dumps(r)}")

    print("\n=== Stress test: 2x slippage/fee ===")
    p_stress = dict(DEFAULT_PARAMS)
    p_stress["fee_bps"] = DEFAULT_PARAMS["fee_bps"] * 4
    result_stress, _ = run_backtest(p_stress, verbose=True)

    print("\n=== Stress test: no drawdown circuit breaker ===")
    p_nobreaker = dict(DEFAULT_PARAMS)
    p_nobreaker["max_drawdown_pct"] = 1.0
    result_nobreaker, _ = run_backtest(p_nobreaker, verbose=True)

    print("\n=== Sub-period robustness (split data into thirds) ===")
    n = len(ec)
    third = n // 3
    for i, label in enumerate(["first third", "middle third", "last third"]):
        seg = ec.iloc[i * third: (i + 1) * third] if i < 2 else ec.iloc[i * third:]
        if len(seg) < 2:
            continue
        seg_return = seg.iloc[-1] / seg.iloc[0] - 1
        print(f"{label}: {seg_return:.2%} over {len(seg)} days")

    print("\n=== Buy & hold benchmark (equal weight, no trading) ===")
    for s in UNIVERSE:
        d = load_daily(s)
        close = d["close"].reindex(ec.index).dropna()
        bh_return = close.iloc[-1] / close.iloc[0] - 1
        print(f"{s}: {bh_return:.2%} buy & hold over {len(close)} days")
