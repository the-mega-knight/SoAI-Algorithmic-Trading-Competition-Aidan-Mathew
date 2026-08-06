"""
Runs the live momentum strategy (via vector_validate.run_backtest) over
the exact calendar window that matches the competition's scored period -
16 Aug to 15 Sep - in each of the past several years. This is purely
descriptive: it shows how this specific slice of the calendar has
historically behaved for this strategy and for buy-and-hold, using real
data. It is NOT a predictor of what 16 Aug-15 Sep 2026 will do - per the
"no market timing" decision, none of this feeds back into the strategy's
logic. It's context for the risk-appetite conversation (aggressive vs
defensive), not a signal.
"""
import json

from vector_validate import DEFAULT_PARAMS, UNIVERSE, load_daily, run_backtest

if __name__ == "__main__":
    print(f"{'Window':<24} {'Strategy':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7}  {'5-stock B&H':>12}")
    print("-" * 76)
    for year in range(2016, 2026):
        start = f"{year}-08-16"
        end = f"{year}-09-15"
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
            f"{year} (16 Aug-15 Sep)     {result['terminal_return']:>9.1%} {sharpe_str:>8} "
            f"{result['max_drawdown']:>7.1%} {result['trade_count']:>7}  {avg_bh:>11.1%}"
        )
