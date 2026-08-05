"""
Independent vectorized re-implementation of the strategy's decision rules,
used to validate + stress-test the logic (per trading-backtest-methodology)
without depending on a full Lumibot install in this sandbox.

v3: low-turnover momentum core + OHLCV-only volatility/gap risk throttle.
See the commit message / README for the full reasoning. In short: v1
(daily binary trend-following on crypto) and v2 (daily binary trend-
following on stocks, then a first attempt at reducing turnover) both lost
to simple buy-and-hold on real data. This version is built directly around
that finding - stay close to fully invested, rebalance infrequently, and
use volatility/gap spikes (not a forbidden external calendar) as the sole
risk-reduction trigger between rebalances.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import itertools
import json

DATA_DIR = Path(__file__).resolve().parent / "data"
UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

# Fixed regardless of parameters, so every sensitivity run evaluates the
# exact same slice of days. Must exceed the largest lookback used anywhere
# in DEFAULT_PARAMS or the grid/sensitivity sweeps below (vol_baseline_days
# = 60 is the largest).
WARMUP_DAYS = 65

DEFAULT_PARAMS = dict(
    mom_lookback=20,
    rebalance_every=5,
    vol_lookback=20,
    vol_baseline_days=60,
    vol_spike_multiplier=1.8,
    vol_throttle_factor=0.5,
    gap_threshold_pct=0.05,
    gap_throttle_factor=0.3,
    gap_cooldown_days=3,
    max_asset_weight=0.40,
    max_total_exposure=0.98,
    min_rebalance_pct=0.02,
    max_drawdown_pct=0.25,
    cooldown_days=5,
    fee_bps=2.0,  # matches the official 2bps fee stated on the competition site
)


def load_daily(symbol):
    df = pd.read_csv(DATA_DIR / f"{symbol}_daily.csv", parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def momentum_weights(pos_roc, max_asset_weight, max_total_exposure):
    if not pos_roc:
        return {}
    total = sum(pos_roc.values())
    if total <= 0:
        return {}
    weights = {s: v / total for s, v in pos_roc.items()}
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

    mom_lb = params["mom_lookback"]
    vol_lb, vol_base_days = params["vol_lookback"], params["vol_baseline_days"]

    sig = {}
    for s in UNIVERSE:
        open_ = daily[s]["open"].reindex(common_index)
        close = daily[s]["close"].reindex(common_index)
        roc = close.pct_change(periods=mom_lb)
        daily_ret = close.pct_change()
        vol_short = daily_ret.rolling(vol_lb).std()
        vol_baseline = daily_ret.rolling(vol_base_days).std()
        vol_spike_ratio = vol_short / vol_baseline.replace(0, np.nan)
        gap_pct = (open_ - close.shift(1)) / close.shift(1)

        sig[s] = pd.DataFrame({
            "close": close, "roc": roc,
            "vol_spike_ratio": vol_spike_ratio, "gap_pct": gap_pct,
        })

    cash = start_cash
    qty = {s: 0.0 for s in UNIVERSE}
    equity_curve = []
    peak = start_cash
    cooldown = 0
    trade_count = 0
    fee_rate = params["fee_bps"] / 10_000.0

    base_weights = {}
    gap_cooldown = {s: 0 for s in UNIVERSE}
    day_index = 0
    force_rebalance = True

    dates = common_index[WARMUP_DAYS:]

    def _flatten_and_reset():
        nonlocal cash, trade_count, base_weights, force_rebalance
        for s in UNIVERSE:
            if qty[s] > 0:
                cash += qty[s] * prices[s] * (1 - fee_rate)
                trade_count += 1
                qty[s] = 0.0
            gap_cooldown[s] = 0
        base_weights = {}
        force_rebalance = True

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

        # -- rebalance schedule: recompute target composition only every
        #    N trading days (or immediately after a breaker reset)
        if force_rebalance or (day_index % params["rebalance_every"] == 0):
            pos_roc = {
                s: sig[s].loc[dt, "roc"] for s in UNIVERSE
                if sig[s].loc[dt, "roc"] == sig[s].loc[dt, "roc"] and sig[s].loc[dt, "roc"] > 0
            }
            base_weights = momentum_weights(pos_roc, params["max_asset_weight"], params["max_total_exposure"])
            force_rebalance = False
        day_index += 1

        # -- daily risk throttle: applied every day regardless of rebalance
        #    schedule, using only OHLCV-derived signals (vol spike, gap)
        target_weights = {}
        for s in UNIVERSE:
            w = base_weights.get(s, 0.0)
            if w <= 0:
                target_weights[s] = 0.0
                continue
            throttle = 1.0
            spike = sig[s].loc[dt, "vol_spike_ratio"]
            if spike == spike and spike > params["vol_spike_multiplier"]:
                throttle = min(throttle, params["vol_throttle_factor"])
            gap = sig[s].loc[dt, "gap_pct"]
            if gap == gap and abs(gap) > params["gap_threshold_pct"]:
                gap_cooldown[s] = params["gap_cooldown_days"]
            if gap_cooldown[s] > 0:
                throttle = min(throttle, params["gap_throttle_factor"])
                gap_cooldown[s] -= 1
            target_weights[s] = w * throttle

        for s in UNIVERSE:
            target_w = target_weights.get(s, 0.0)
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
    print("=== Baseline run (v3: weekly momentum core + vol/gap throttle) ===")
    result, ec = run_backtest(DEFAULT_PARAMS, verbose=True)

    print("\n=== Grid: mom_lookback x rebalance_every ===")
    grid_results = []
    for mom_lb, rebal in itertools.product([10, 15, 20, 25, 30], [3, 5, 10, 15]):
        p = dict(DEFAULT_PARAMS)
        p["mom_lookback"], p["rebalance_every"] = mom_lb, rebal
        r, _ = run_backtest(p)
        grid_results.append({"mom_lookback": mom_lb, "rebalance_every": rebal, **r})

    grid_df = pd.DataFrame(grid_results).sort_values("sharpe", ascending=False)
    pd.set_option("display.width", 120)
    print(grid_df.to_string(index=False))

    print("\n=== Sensitivity: vol_spike_multiplier (1.5, 1.8, 2.2) ===")
    for m in [1.5, 1.8, 2.2]:
        p = dict(DEFAULT_PARAMS)
        p["vol_spike_multiplier"] = m
        r, _ = run_backtest(p)
        print(f"vol_spike_multiplier={m}: {json.dumps(r)}")

    print("\n=== Sensitivity: gap_threshold_pct (0.03, 0.05, 0.08) ===")
    for g in [0.03, 0.05, 0.08]:
        p = dict(DEFAULT_PARAMS)
        p["gap_threshold_pct"] = g
        r, _ = run_backtest(p)
        print(f"gap_threshold_pct={g}: {json.dumps(r)}")

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
