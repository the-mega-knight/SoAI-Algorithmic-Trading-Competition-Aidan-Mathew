"""
Independent vectorized re-implementation of the strategy's decision rules,
used to validate + stress-test the logic (per trading-backtest-methodology)
without depending on a full Lumibot install in this sandbox.

This is NOT the official backtest (that's backtest.py, driven by the real
Lumibot engine) -- it's a from-scratch pandas re-derivation of the exact
same rules in strategies/strategy.py, run against the same CSVs, so we can
sanity-check behavior and stress-test parameters quickly. Both
implementations should be checked for agreement once `python backtest.py`
can actually be run (e.g. on a machine where the full lumibot install
completes).
"""
import numpy as np
import pandas as pd
from pathlib import Path
import itertools
import json

DATA_DIR = Path(__file__).resolve().parent / "data"
UNIVERSE = ["BTC", "ETH", "SOL"]

DEFAULT_PARAMS = dict(
    fast=10, slow=20, mom=10, vol_lookback=20,
    max_asset_weight=0.50, max_total_exposure=0.90,
    min_rebalance_pct=0.02, max_drawdown_pct=0.25, cooldown_days=5,
    fee_bps=2.0,  # matches the official 2bps fee stated on the competition site
)


def load_daily(symbol):
    df = pd.read_csv(DATA_DIR / f"{symbol}_1m_spot.csv", parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    daily = df.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    return daily


def inverse_vol_weights(qualifying_vol, max_asset_weight, max_total_exposure):
    if not qualifying_vol:
        return {}
    inv_vol = {s: 1.0 / v for s, v in qualifying_vol.items() if v > 0}
    total = sum(inv_vol.values())
    if total <= 0:
        return {}
    weights = {s: w / total for s, w in inv_vol.items()}
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

    fast, slow, mom, vol_lb = params["fast"], params["slow"], params["mom"], params["vol_lookback"]
    sig = {}
    for s in UNIVERSE:
        close = daily[s]["close"].reindex(common_index)
        sma_fast = close.rolling(fast).mean()
        sma_slow = close.rolling(slow).mean()
        roc = close.pct_change(periods=mom)
        daily_ret = close.pct_change()
        vol = daily_ret.rolling(vol_lb).std()
        entry = (sma_fast > sma_slow) & (roc > 0)
        sig[s] = pd.DataFrame({"close": close, "entry": entry, "vol": vol})

    cash = start_cash
    qty = {s: 0.0 for s in UNIVERSE}
    equity_curve = []
    peak = start_cash
    cooldown = 0
    trade_count = 0
    fee_rate = params["fee_bps"] / 10_000.0

    warmup = max(fast, slow, mom, vol_lb) + 1
    dates = common_index[warmup:]

    for dt in dates:
        prices = {s: sig[s].loc[dt, "close"] for s in UNIVERSE}
        port_value = cash + sum(qty[s] * prices[s] for s in UNIVERSE)
        if port_value > peak:
            peak = port_value
        drawdown = port_value / peak - 1.0 if peak else 0.0

        if cooldown > 0:
            cooldown -= 1
            for s in UNIVERSE:
                if qty[s] > 0:
                    cash += qty[s] * prices[s] * (1 - fee_rate)
                    trade_count += 1
                    qty[s] = 0.0
            equity_curve.append((dt, cash))
            continue

        if drawdown <= -params["max_drawdown_pct"]:
            for s in UNIVERSE:
                if qty[s] > 0:
                    cash += qty[s] * prices[s] * (1 - fee_rate)
                    trade_count += 1
                    qty[s] = 0.0
            cooldown = params["cooldown_days"]
            equity_curve.append((dt, cash))
            continue

        qualifying_vol = {
            s: sig[s].loc[dt, "vol"] for s in UNIVERSE
            if bool(sig[s].loc[dt, "entry"]) and sig[s].loc[dt, "vol"] > 0
        }
        weights = inverse_vol_weights(qualifying_vol, params["max_asset_weight"], params["max_total_exposure"])

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

    ann_factor = 365  # crypto trades every day
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
    print("=== Baseline run (default params) ===")
    result, ec = run_backtest(DEFAULT_PARAMS, verbose=True)

    print("\n=== Parameter sensitivity grid (fast/slow SMA) ===")
    grid_results = []
    for fast, slow in itertools.product([5, 8, 10, 12, 15], [15, 20, 25, 30]):
        if fast >= slow:
            continue
        p = dict(DEFAULT_PARAMS)
        p["fast"], p["slow"] = fast, slow
        r, _ = run_backtest(p)
        grid_results.append({"fast": fast, "slow": slow, **r})

    grid_df = pd.DataFrame(grid_results).sort_values("sharpe", ascending=False)
    pd.set_option("display.width", 120)
    print(grid_df.to_string(index=False))

    print("\n=== Stress test: 2x slippage/fee ===")
    p_stress = dict(DEFAULT_PARAMS)
    p_stress["fee_bps"] = DEFAULT_PARAMS["fee_bps"] * 4  # crude stand-in for slippage+fee friction
    result_stress, _ = run_backtest(p_stress, verbose=True)

    print("\n=== Stress test: no drawdown circuit breaker ===")
    p_nobreaker = dict(DEFAULT_PARAMS)
    p_nobreaker["max_drawdown_pct"] = 1.0  # effectively disables it
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
