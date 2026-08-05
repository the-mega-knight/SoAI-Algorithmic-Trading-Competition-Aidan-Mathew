"""
Reruns the strategy over the trailing 1-month window (today minus ~30
calendar days through today) using whatever DEFAULT_PARAMS currently
live in vector_validate.py.

Caveat: ~21 trading days is a tiny sample. This is a quick sanity
check of very recent behavior, not a substitute for regime_test.py's
multi-year, multi-regime robustness check. Don't read too much into a
single short window - it's exactly the kind of small sample that can
make a strategy look better or worse than it really is by chance.
"""
import datetime as dt

from vector_validate import DEFAULT_PARAMS, UNIVERSE, load_daily, run_backtest

end_date = dt.date.today()
start_date = end_date - dt.timedelta(days=30)

print(f"Trailing 1-month window: {start_date} to {end_date}\n")

result, ec = run_backtest(DEFAULT_PARAMS, start_date=str(start_date), end_date=str(end_date), verbose=False)

print(f"{'Metric':<20}{'Value':>15}")
print("-" * 35)
print(f"{'Terminal return':<20}{result['terminal_return']:>14.1%}")
sharpe_str = f"{result['sharpe']:.2f}" if result["sharpe"] is not None else "n/a"
print(f"{'Sharpe':<20}{sharpe_str:>15}")
print(f"{'Max drawdown':<20}{result['max_drawdown']:>14.1%}")
print(f"{'Trades':<20}{result['trade_count']:>15}")
print(f"{'Days':<20}{result['num_days']:>15}")
print(f"{'Final value':<20}{result['final_value']:>15,.0f}")

print("\n=== Buy & hold comparison (equal weight, no trading) ===")
for s in UNIVERSE:
    d = load_daily(s)
    close = d["close"].reindex(ec.index).dropna()
    if len(close) < 2:
        print(f"{s}: not enough data in this window")
        continue
    bh_return = close.iloc[-1] / close.iloc[0] - 1
    print(f"{s:<8}avg buy & hold: {bh_return:>8.1%}")
