"""
UNCONFIRMED / RESEARCH ONLY - not part of the submitted strategy.

Two changes from hedge_validate.py / directional_hedge_validate.py, aimed
directly at what killed those versions:

1. COLLAR structure instead of naked puts: on every name currently held,
   sell an OTM call (collect premium) and use that premium to fund an OTM
   put (pay premium), sized 1:1 against the actual shares held (a real
   collar, not a budget-based bet). Selling the call funds most/all of the
   put's cost, instead of the put being pure out-of-pocket insurance - the
   tradeoff is giving up upside above the call strike, not giving up cash.

2. Rolling ~1-month window test, not just full-year regimes. The
   competition's actual scored window is 16 Aug-15 Sep 2026 (~1 month).
   The earlier full 10-year tests punished repeated hedge cost via ~500
   rebalance cycles of compounding - that's the wrong shape for "should
   we do this for one ~21-trading-day window." This walks a 21-trading-
   day window forward across the whole 10y history and reports the
   distribution of outcomes, which is a much closer proxy for the actual
   decision.

Same pricing caveat as before: premiums are Black-Scholes with realized
vol x a vol-risk-premium fudge factor as the IV proxy, since no real
option quotes are available locally.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from vector_validate import DEFAULT_PARAMS, UNIVERSE, load_daily, momentum_weights, WARMUP_DAYS

RISK_FREE_RATE = 0.04


def bs_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return max(float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)), 0.0)


def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return max(float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)), 0.0)


COLLAR_DEFAULTS = dict(
    put_otm_pct=0.08,
    call_otm_pct=0.08,
    dte_trading_days=15,
    vol_premium_mult=1.3,
    enabled=True,
)


def _load_signals(params):
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
        realized_vol_ann = vol_short * np.sqrt(252)
        sig[s] = pd.DataFrame({
            "close": close, "roc": roc, "vol_spike_ratio": vol_spike_ratio,
            "gap_pct": gap_pct, "realized_vol_ann": realized_vol_ann,
        })
    return sig, common_index


def run_backtest_collar(params, ccfg, sig, common_index, start_cash=1_000_000, start_date=None, end_date=None):
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
    if start_date is not None:
        dates = dates[dates >= pd.Timestamp(start_date)]
    if end_date is not None:
        dates = dates[dates <= pd.Timestamp(end_date)]
    dates = list(dates)
    if len(dates) < 2:
        return None, None

    pending = []  # {symbol, leg('put'/'call'), side('long'/'short'), strike, expiry_date, qty}
    premium_net = 0.0
    payoff_net = 0.0

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

    for i, dt in enumerate(dates):
        prices = {s: sig[s].loc[dt, "close"] for s in UNIVERSE}

        if ccfg["enabled"] and pending:
            still = []
            for h in pending:
                if h["expiry_date"] == dt:
                    S_T = prices[h["symbol"]]
                    intrinsic = max(h["strike"] - S_T, 0.0) if h["leg"] == "put" else max(S_T - h["strike"], 0.0)
                    flow = intrinsic * h["qty"] * (1 if h["side"] == "long" else -1)
                    cash += flow
                    payoff_net += flow
                else:
                    still.append(h)
            pending = still

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

        is_rebalance_day = force_rebalance or (day_index % params["rebalance_every"] == 0)
        if is_rebalance_day:
            pos_roc = {
                s: sig[s].loc[dt, "roc"] for s in UNIVERSE
                if sig[s].loc[dt, "roc"] == sig[s].loc[dt, "roc"] and sig[s].loc[dt, "roc"] > 0
            }
            base_weights = momentum_weights(pos_roc, params["max_asset_weight"], params["max_total_exposure"])
            force_rebalance = False
        day_index += 1

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

        # -- Collar every currently-held name, once per rebalance.
        if ccfg["enabled"] and is_rebalance_day:
            expiry_idx = min(i + ccfg["dte_trading_days"], len(dates) - 1)
            expiry_date = dates[expiry_idx]
            T_years = max(expiry_idx - i, 1) / 252.0
            for s in UNIVERSE:
                shares = qty[s]
                if shares <= 0:
                    continue
                spot = prices[s]
                iv = sig[s].loc[dt, "realized_vol_ann"]
                if not (iv == iv) or iv <= 0:
                    continue
                iv = iv * ccfg["vol_premium_mult"]

                put_strike = spot * (1 - ccfg["put_otm_pct"])
                call_strike = spot * (1 + ccfg["call_otm_pct"])
                put_prem = bs_put(spot, put_strike, T_years, RISK_FREE_RATE, iv)
                call_prem = bs_call(spot, call_strike, T_years, RISK_FREE_RATE, iv)

                # Buy the put (cost), sell the call (credit), 1:1 vs shares held.
                cash -= put_prem * shares
                cash += call_prem * shares
                premium_net += (put_prem - call_prem) * shares
                pending.append({"symbol": s, "leg": "put", "side": "long",
                                 "strike": put_strike, "expiry_date": expiry_date, "qty": shares})
                pending.append({"symbol": s, "leg": "call", "side": "short",
                                 "strike": call_strike, "expiry_date": expiry_date, "qty": shares})

        port_value = cash + sum(qty[s] * prices[s] for s in UNIVERSE)
        equity_curve.append((dt, port_value))

    ec = pd.Series({d: v for d, v in equity_curve}).sort_index()
    if len(ec) < 2:
        return None, ec
    returns = ec.pct_change().dropna()
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else np.nan
    running_max = ec.cummax()
    max_dd = (ec / running_max - 1).min()
    terminal_return = ec.iloc[-1] / ec.iloc[0] - 1

    result = {
        "terminal_return": float(terminal_return),
        "sharpe": float(sharpe) if sharpe == sharpe else None,
        "max_drawdown": float(max_dd),
        "trade_count": trade_count,
        "premium_net_cost": float(premium_net),
        "option_pnl": float(payoff_net - premium_net),
        "num_days": len(ec),
        "final_value": float(ec.iloc[-1]),
    }
    return result, ec


if __name__ == "__main__":
    sig, common_index = _load_signals(DEFAULT_PARAMS)

    print("=== Full 10y: baseline vs collar ===")
    off = dict(COLLAR_DEFAULTS, enabled=False)
    r_base, _ = run_backtest_collar(DEFAULT_PARAMS, off, sig, common_index)
    r_collar, _ = run_backtest_collar(DEFAULT_PARAMS, COLLAR_DEFAULTS, sig, common_index)
    print("baseline:", json.dumps(r_base, indent=2))
    print("collar:  ", json.dumps(r_collar, indent=2))

    REGIMES = [
        ("Pre-COVID bull (2018-2019)", "2018-01-01", "2019-12-31"),
        ("COVID crash & recovery (2020)", "2020-01-01", "2020-12-31"),
        ("2021 melt-up", "2021-01-01", "2021-12-31"),
        ("2022 rate-hike bear market", "2022-01-01", "2022-12-31"),
        ("2023-2024 AI bull run", "2023-01-01", "2024-12-31"),
        ("Most recent 12 months", "2025-08-01", "2026-08-04"),
    ]
    print(f"\n{'Regime':<32} {'Base':>9} {'Collar':>9} {'Delta':>9} {'BaseDD':>8} {'ColDD':>8} {'NetCost':>10}")
    print("-" * 90)
    for label, start, end in REGIMES:
        rb, _ = run_backtest_collar(DEFAULT_PARAMS, off, sig, common_index, start_date=start, end_date=end)
        rc, _ = run_backtest_collar(DEFAULT_PARAMS, COLLAR_DEFAULTS, sig, common_index, start_date=start, end_date=end)
        delta = rc["terminal_return"] - rb["terminal_return"]
        print(
            f"{label:<32} {rb['terminal_return']:>8.1%} {rc['terminal_return']:>8.1%} "
            f"{delta:>+8.1%} {rb['max_drawdown']:>7.1%} {rc['max_drawdown']:>7.1%} "
            f"${rc['premium_net_cost']:>9,.0f}"
        )

    print("\n=== Rolling ~1-month (21 trading day) windows across 10y ===")
    WINDOW = 21
    all_dates = list(common_index[common_index >= common_index[WARMUP_DAYS]])
    base_rets, collar_rets = [], []
    base_dds, collar_dds = [], []
    n_windows = 0
    for start_i in range(0, len(all_dates) - WINDOW, WINDOW):  # non-overlapping months
        w_start = all_dates[start_i]
        w_end = all_dates[min(start_i + WINDOW, len(all_dates) - 1)]
        rb, _ = run_backtest_collar(DEFAULT_PARAMS, off, sig, common_index, start_date=w_start, end_date=w_end)
        rc, _ = run_backtest_collar(DEFAULT_PARAMS, COLLAR_DEFAULTS, sig, common_index, start_date=w_start, end_date=w_end)
        if rb is None or rc is None:
            continue
        n_windows += 1
        base_rets.append(rb["terminal_return"])
        collar_rets.append(rc["terminal_return"])
        base_dds.append(rb["max_drawdown"])
        collar_dds.append(rc["max_drawdown"])

    base_rets, collar_rets = np.array(base_rets), np.array(collar_rets)
    base_dds, collar_dds = np.array(base_dds), np.array(collar_dds)
    win_rate = (collar_rets > base_rets).mean()
    print(f"n_windows={n_windows}")
    print(f"Mean 1mo return:   base={base_rets.mean():.2%}  collar={collar_rets.mean():.2%}")
    print(f"Median 1mo return: base={np.median(base_rets):.2%}  collar={np.median(collar_rets):.2%}")
    print(f"Worst 1mo return:  base={base_rets.min():.2%}  collar={collar_rets.min():.2%}")
    print(f"10th pct return:   base={np.percentile(base_rets,10):.2%}  collar={np.percentile(collar_rets,10):.2%}")
    print(f"Mean max_dd:       base={base_dds.mean():.2%}  collar={collar_dds.mean():.2%}")
    print(f"Worst max_dd:      base={base_dds.min():.2%}  collar={collar_dds.min():.2%}")
    print(f"Collar beats base in {win_rate:.0%} of {n_windows} 1-month windows")
