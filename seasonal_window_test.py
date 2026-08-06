"""
Runs the live momentum strategy (via vector_validate.run_backtest) over
the exact calendar window that matches the competition's scored period -
16 Aug to 16 Sep - in each of the past 8 years. This is purely
descriptive: it shows how this specific slice of the calendar has
historically behaved for this strategy and for buy-and-hold, using real
data. It is NOT a predictor of what 16 Aug-16 Sep 2026 will do - per the
"no market timing" decision, none of this feeds back into the strategy's
logic. It's context for the risk-appetite conversation (aggressive vs
defensive), not a signal.
"""
import datetime as _dt

from vector_validate import DEFAULT_PARAMS, UNIVERSE, load_daily, run_backtest

if __name__ == "__main__":
    current_year = _dt.date.today().year
    years = range(current_year - 8, current_year)  # past 8 completed years

    print(f"{'Window':<24} {'Strategy':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7}  {'5-stock B&H':>12}")
    print("-" * 76)
    for year in years:
        start = f"{year}-08-16"
        end = f"{year}-09-16"
        result, ec = run_backtest(DEFAULT_PARAMS, start_date=start, end_date=end)
        if result["num_days"] < 5:
            continue
        bh_returns = []
        for s in UNIVERSE:
            d = load_daily(s)
            window = d["close"][(d.index >= start) & (d.index <= end)]
            if len(window) < 2:
                continue
            bh_returns.append(window.iloc[-1] / window.iloc[0] - 1)
        avg_bh = sum(bh_returns) / len(bh_returns) if bh_returns else float("nan")
        sharpe_str = f"{result['sharpe']:.2f}" if result["sharpe"] is not None else "n/a"
        print(
            f"{year} (16 Aug-16 Sep)     {result['terminal_return']:>9.1%} {sharpe_str:>8} "
            f"{result['max_drawdown']:>7.1%} {result['trade_count']:>7}  {avg_bh:>11.1%}"
        )
