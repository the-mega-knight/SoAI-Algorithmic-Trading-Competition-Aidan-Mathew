"""
Same six-regime robustness test as regime_test.py, but running the
mean-reversion strategy (mean_reversion.py) instead of the momentum
strategy (vector_validate.py). Same regime windows, same buy & hold
comparison, so results are directly comparable between the two.
"""
import pandas as pd

from mean_reversion import DEFAULT_PARAMS, UNIVERSE, load_daily, run_backtest

REGIMES = [
    ("Pre-COVID bull (2018-2019)", "2018-01-01", "2019-12-31"),
    ("COVID crash & recovery (2020)", "2020-01-01", "2020-12-31"),
    ("2021 melt-up", "2021-01-01", "2021-12-31"),
    ("2022 rate-hike bear market", "2022-01-01", "2022-12-31"),
    ("2023-2024 AI bull run", "2023-01-01", "2024-12-31"),
    ("Most recent 12 months", "2025-08-01", "2026-08-04"),
]

print(f"{'Regime':<32}{'Return':>10}{'Sharpe':>10}{'MaxDD':>10}{'Trades':>10}{'Days':>8}")
print("-" * 80)

regime_curves = {}
for label, start, end in REGIMES:
    result, ec = run_backtest(DEFAULT_PARAMS, start_date=start, end_date=end)
    regime_curves[label] = ec
    sharpe_str = f"{result['sharpe']:.2f}" if result["sharpe"] is not None else "n/a"
    print(f"{label:<32}{result['terminal_return']:>9.1%}{sharpe_str:>10}"
          f"{result['max_drawdown']:>9.1%}{result['trade_count']:>10}{result['num_days']:>8}")

print("\n=== Buy & hold comparison per regime (equal-weight average of the 5 stocks) ===")
for label, start, end in REGIMES:
    ec = regime_curves[label]
    if len(ec) < 2:
        print(f"{label:<32}not enough data")
        continue
    bh_returns = []
    for s in UNIVERSE:
        close = load_daily(s)["close"].reindex(ec.index).dropna()
        if len(close) >= 2:
            bh_returns.append(close.iloc[-1] / close.iloc[0] - 1)
    avg_bh = sum(bh_returns) / len(bh_returns) if bh_returns else float("nan")
    print(f"{label:<32}avg buy & hold: {avg_bh:>8.1%}")
