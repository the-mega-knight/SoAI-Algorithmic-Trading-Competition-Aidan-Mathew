"""
Runs the strategy (via vector_validate.run_backtest) across several
distinct historical market regimes, instead of just the single ~2-year
window used for the main validation. Per trading-backtest-methodology:
a strategy should hold up across trending, choppy, high-vol, and low-vol
periods, not just look good in whichever window it happened to be tuned
against.

Requires 10 years of daily data - run scripts/fetch_stock_data.py first
(it now pulls period="10y" instead of "2y").
"""
import json

from vector_validate import DEFAULT_PARAMS, UNIVERSE, load_daily, run_backtest

REGIMES = [
    ("Pre-COVID bull (2018-2019)", "2018-01-01", "2019-12-31"),
    ("COVID crash & recovery (2020)", "2020-01-01", "2020-12-31"),
    ("2021 melt-up", "2021-01-01", "2021-12-31"),
    ("2022 rate-hike bear market", "2022-01-01", "2022-12-31"),
    ("2023-2024 AI bull run", "2023-01-01", "2024-12-31"),
    ("Most recent 12 months", "2025-08-01", "2026-08-04"),
]

if __name__ == "__main__":
    print(f"{'Regime':<32} {'Return':>9} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7} {'Days':>6}")
    print("-" * 75)

    all_results = []
    for label, start, end in REGIMES:
        result, ec = run_backtest(DEFAULT_PARAMS, start_date=start, end_date=end)
        all_results.append((label, result))
        sharpe_str = f"{result['sharpe']:.2f}" if result["sharpe"] is not None else "n/a"
        print(
            f"{label:<32} {result['terminal_return']:>8.1%} {sharpe_str:>8} "
            f"{result['max_drawdown']:>8.1%} {result['trade_count']:>7} {result['num_days']:>6}"
        )

    print("\n=== Buy & hold comparison per regime (equal-weight average of the 5 stocks) ===")
    for label, start, end in REGIMES:
        bh_returns = []
        for s in UNIVERSE:
            d = load_daily(s)
            window = d["close"][(d.index >= start) & (d.index <= end)]
            if len(window) < 2:
                continue
            bh_returns.append(window.iloc[-1] / window.iloc[0] - 1)
        if bh_returns:
            avg_bh = sum(bh_returns) / len(bh_returns)
            print(f"{label:<32} avg buy & hold: {avg_bh:.1%}")
