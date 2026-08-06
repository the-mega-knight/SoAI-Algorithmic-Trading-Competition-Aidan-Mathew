"""
Simple strategy comparison: One-day Reversal vs One-day Momentum (daily resampled)

This script uses the existing research backtesting framework (DataLoader, Resampler,
DataAligner, Backtester, PerformanceMetrics) and does NOT modify any framework code.

How it works:
- Loads *_1m_spot.csv files from data/
- Resamples each symbol to 1D using Resampler.resample_ohlcv
- Aligns symbols and extracts daily close prices
- Implements two strategies that use only prior completed day's return
- Runs both strategies through Backtester (which enforces the no-lookahead rule)
- Prints concise metrics and saves a CSV with daily equity for both strategies

Run from project root:
    python research/strategy_tests.py

"""

from pathlib import Path
import importlib.util, sys
import pandas as pd
import numpy as np

# Dynamically load the research_backtest module by path so the script can be run directly
_rb_path = Path(__file__).resolve().parents[0] / "research_backtest.py"
_spec = importlib.util.spec_from_file_location("research_backtest", str(_rb_path))
_rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rb)

# Expose the framework classes/functions used below
DataLoader = _rb.DataLoader
Resampler = _rb.Resampler
DataAligner = _rb.DataAligner
Backtester = _rb.Backtester
PerformanceMetrics = _rb.PerformanceMetrics


def one_day_reversal_strategy(close_prices_df: pd.DataFrame, current_row_idx: int) -> pd.Series:
    """Select the 2 worst-performing assets based on previous completed day's return.

    close_prices_df: DataFrame containing timestamp + symbol close columns for rows up to t-1
    current_row_idx: integer index (t-1) provided by the backtester

    Returns a pd.Series indexed by symbol names with weights summing to 1.0 (if at least 2 assets available).
    """
    symbols = [c for c in close_prices_df.columns if c != "timestamp"]

    # If insufficient history, return zero weights
    if len(close_prices_df) < 2:
        return pd.Series(0.0, index=symbols)

    # Use only the two most recent completed closes: t-1 and t-2
    prev = close_prices_df[symbols].iloc[-1]
    prev2 = close_prices_df[symbols].iloc[-2]

    # Compute returns for the previous completed day: (t-1)/(t-2) - 1
    returns = prev / prev2 - 1

    # Drop NaNs (if any symbol lacks t-2) before ranking
    valid_returns = returns.dropna()

    # If fewer than 2 valid assets, return zeros
    if len(valid_returns) < 1:
        return pd.Series(0.0, index=symbols)

    # Select 2 worst-performing assets
    n_select = min(2, len(valid_returns))
    worst = valid_returns.nsmallest(n_select)

    weights = pd.Series(0.0, index=symbols)
    if len(worst) == 1:
        weights[worst.index[0]] = 1.0
    else:
        weights[worst.index] = 1.0 / n_select

    return weights


def one_day_momentum_strategy(close_prices_df: pd.DataFrame, current_row_idx: int) -> pd.Series:
    """Select the 2 best-performing assets based on previous completed day's return.

    Same interface/behavior as reversal but picks best performers.
    """
    symbols = [c for c in close_prices_df.columns if c != "timestamp"]

    if len(close_prices_df) < 2:
        return pd.Series(0.0, index=symbols)

    prev = close_prices_df[symbols].iloc[-1]
    prev2 = close_prices_df[symbols].iloc[-2]

    returns = prev / prev2 - 1
    valid_returns = returns.dropna()

    if len(valid_returns) < 1:
        return pd.Series(0.0, index=symbols)

    n_select = min(2, len(valid_returns))
    best = valid_returns.nlargest(n_select)

    weights = pd.Series(0.0, index=symbols)
    if len(best) == 1:
        weights[best.index[0]] = 1.0
    else:
        weights[best.index] = 1.0 / n_select

    return weights


def run_and_report(close_prices_daily: pd.DataFrame, strategy_fn, strategy_name: str):
    bt = Backtester(close_prices_daily)
    result = bt.run(strategy_fn)

    metrics = result["metrics"]

    print(f"\n=== {strategy_name} RESULTS ===")
    print(f"Total Return: {metrics['total_return']:.6f}")
    print(f"CAGR: {metrics['cagr']:.6f}")
    print(f"Annualized Volatility: {metrics['annualized_volatility']:.6f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.6f}")
    print(f"Average Turnover: {metrics['average_turnover']:.6f}")
    print(f"Average Transaction Cost (per period): {metrics['average_transaction_cost']:.6f}")
    print(f"Final Portfolio Value: {metrics['final_value']:.2f}")

    return result


def diagnostics_from_csv(out_path: Path):
    """Run diagnostic analysis from existing CSV file."""
    df = pd.read_csv(out_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Set timestamp index for resampling
    df_idx = df.set_index("timestamp")

    # Momentum series
    mom_eq = df_idx["momentum_equity"].astype(float)
    rev_eq = df_idx["reversal_equity"].astype(float)

    mom_ret = mom_eq.pct_change().dropna()
    rev_ret = rev_eq.pct_change().dropna()

    n_days = len(mom_ret)
    mom_pos = (mom_ret > 0).sum()
    mom_neg = (mom_ret < 0).sum()
    mom_pos_pct = mom_pos / n_days * 100 if n_days>0 else 0.0
    mom_neg_pct = mom_neg / n_days * 100 if n_days>0 else 0.0

    mom_mean = mom_ret.mean()
    mom_median = mom_ret.median()
    mom_std = mom_ret.std()

    best5 = mom_ret.sort_values(ascending=False).head(5)
    worst5 = mom_ret.sort_values().head(5)

    # Max drawdown and range for momentum
    running_max = mom_eq.cummax()
    drawdown = (mom_eq - running_max) / running_max
    max_dd = drawdown.min()
    if pd.isna(max_dd):
        dd_start = None
        dd_end = None
    else:
        dd_end = drawdown.idxmin()
        # start is the last time running_max was equal to running_max at or before dd_end
        running_max_at_end = running_max.loc[:dd_end].max()
        # find first index where running_max equals running_max_at_end
        candidates = running_max[running_max == running_max_at_end]
        if not candidates.empty:
            dd_start = candidates.index[0]
        else:
            dd_start = None

    # Weekly cumulative returns (end of week equity relative to start)
    weekly = mom_eq.resample('W').last().dropna()
    if not weekly.empty:
        weekly_cumret = (weekly / weekly.iloc[0] - 1).rename('weekly_cumret')
    else:
        weekly_cumret = pd.Series([], dtype=float)

    # Reversal positive/negative-day stats
    rev_n = len(rev_ret)
    rev_pos = (rev_ret > 0).sum()
    rev_neg = (rev_ret < 0).sum()
    rev_pos_pct = rev_pos / rev_n * 100 if rev_n>0 else 0.0
    rev_neg_pct = rev_neg / rev_n * 100 if rev_n>0 else 0.0

    # Print concise diagnostic report
    print('\n=== DIAGNOSTIC REPORT (Momentum) ===')
    print(f'1. Number of trading days: {n_days}')
    print(f'2. Positive-return days: {mom_pos} ({mom_pos_pct:.1f}%)')
    print(f'3. Negative-return days: {mom_neg} ({mom_neg_pct:.1f}%)')
    print(f'4. Mean daily return: {mom_mean:.6f}')
    print(f'5. Median daily return: {mom_median:.6f}')
    print(f'6. Std dev daily returns: {mom_std:.6f}')
    print('\n7. Best 5 momentum days (date -> return):')
    for dt, r in best5.items():
        print(f'   {pd.to_datetime(dt).date()} -> {r:.6f}')
    print('\n8. Worst 5 momentum days (date -> return):')
    for dt, r in worst5.items():
        print(f'   {pd.to_datetime(dt).date()} -> {r:.6f}')

    print('\n9. Maximum drawdown: {0:.6f}'.format(max_dd))
    if dd_start is not None and dd_end is not None:
        print(f'   Drawdown period: {dd_start.date()} -> {dd_end.date()}')
    else:
        print('   Drawdown period: N/A')

    print('\n10. Weekly cumulative returns (end-of-week):')
    for dt, val in weekly_cumret.items():
        print(f'   {dt.date()} -> {val:.6f}')

    print('\n=== DIAGNOSTIC SUMMARY (Reversal) ===')
    print(f'Positive days: {rev_pos} / {rev_n} ({rev_pos_pct:.1f}%)')
    print(f'Negative days: {rev_neg} / {rev_n} ({rev_neg_pct:.1f}%)')

    # Final factual conclusion
    # Determine concentration: check how much top 5 days contribute to total cumulative return
    total_cum = mom_eq.iloc[-1] / mom_eq.iloc[0] - 1
    top5_contrib = ( (1 + best5).prod() - 1 ) if not best5.empty else 0.0
    # Rough heuristic: if top5_contrib is a large fraction of total_cum, say concentrated
    fraction = (top5_contrib / total_cum) if total_cum != 0 else None

    if fraction is None:
        conclusion = 'Inconclusive (zero total cumulative return)'
    elif fraction >= 0.5:
        conclusion = 'Concentrated in a small number of days (top gains explain >=50% of total)'
    else:
        conclusion = 'Broadly distributed across the sample (top gains explain <50% of total)'

    print('\nConclusion:')
    print(conclusion)


def half_sample_robustness_test():
    """Run chronological half-sample robustness test for the existing one-day momentum strategy.

    Uses the same strategy implementation and framework; does not modify any framework files.
    """
    # Load & resample to daily
    loader = DataLoader(Path("data"))
    symbol_data = loader.load_all_symbols()

    resampled = {}
    for symbol, df in symbol_data.items():
        df_daily = Resampler.resample_ohlcv(df, symbol, "1D")
        resampled[symbol] = df_daily

    aligner = DataAligner(resampled)
    close_prices_daily = aligner.get_close_prices()

    n = len(close_prices_daily)
    # Prefer exactly first 15 and last 15 if dataset is >=30 days
    if n >= 30:
        first_n = 15
        second_n = 15
    else:
        first_n = n // 2
        second_n = n - first_n

    first_df = close_prices_daily.iloc[:first_n].reset_index(drop=True)
    second_df = close_prices_daily.iloc[-second_n:].reset_index(drop=True)

    # Date ranges
    first_range = (first_df["timestamp"].iloc[0].date(), first_df["timestamp"].iloc[-1].date())
    second_range = (second_df["timestamp"].iloc[0].date(), second_df["timestamp"].iloc[-1].date())

    print("\nRunning half-sample robustness test for One-Day Momentum strategy")
    print(f"First half: {first_range[0]} -> {first_range[1]} ({len(first_df)} days)")
    print(f"Second half: {second_range[0]} -> {second_range[1]} ({len(second_df)} days)")

    bt1 = Backtester(first_df)
    res1 = bt1.run(one_day_momentum_strategy)
    m1 = res1["metrics"]
    returns1 = res1["results_df"]["returns"].dropna()
    pos1 = (returns1 > 0).sum()
    neg1 = (returns1 < 0).sum()

    bt2 = Backtester(second_df)
    res2 = bt2.run(one_day_momentum_strategy)
    m2 = res2["metrics"]
    returns2 = res2["results_df"]["returns"].dropna()
    pos2 = (returns2 > 0).sum()
    neg2 = (returns2 < 0).sum()

    def print_metrics(m, pos, neg, label, date_range):
        print(f"\n--- {label} ({date_range[0]} -> {date_range[1]}) ---")
        print(f"Total Return: {m['total_return']:.6f}")
        print(f"CAGR (illustrative): {m['cagr']:.6f}")
        print(f"Annualized Volatility: {m['annualized_volatility']:.6f}")
        print(f"Sharpe Ratio: {m['sharpe_ratio']:.4f}")
        print(f"Maximum Drawdown: {m['max_drawdown']:.6f}")
        print(f"Average Turnover: {m['average_turnover']:.6f}")
        print(f"Final Portfolio Value: {m['final_value']:.2f}")
        print(f"Positive-return days: {pos}")
        print(f"Negative-return days: {neg}")

    print_metrics(m1, pos1, neg1, "First half", first_range)
    print_metrics(m2, pos2, neg2, "Second half", second_range)

    both_positive = (m1['total_return'] > 0) and (m2['total_return'] > 0)
    if both_positive:
        conclusion = 'A) "Momentum is positive in both halves"'
    elif (m1['total_return'] > 0) or (m2['total_return'] > 0):
        conclusion = 'B) "Momentum is positive in only one half"'
    else:
        conclusion = 'C) "Momentum is negative in both halves"'

    print('\nConclusion:')
    print(conclusion)


def attribution_analysis():
    """Run per-asset attribution for the full 30-day momentum backtest.

    This re-runs the backtest (no framework changes) and computes per-asset
    contributions net of transaction costs allocated proportionally to weight changes.
    """
    # Load & resample
    loader = DataLoader(Path("data"))
    symbol_data = loader.load_all_symbols()

    resampled = {}
    for symbol, df in symbol_data.items():
        df_daily = Resampler.resample_ohlcv(df, symbol, "1D")
        resampled[symbol] = df_daily

    aligner = DataAligner(resampled)
    close_prices_daily = aligner.get_close_prices()

    # Run full-period momentum backtest
    bt = Backtester(close_prices_daily)
    res = bt.run(one_day_momentum_strategy)
    results_df = res["results_df"].copy()

    # Identify weight columns (they end with '_weight')
    weight_cols = [c for c in results_df.columns if c.endswith("_weight")]
    # Normalize column names to symbol form by stripping suffix
    symbols = [c[:-7] for c in weight_cols]

    weights_df = results_df[weight_cols].copy()
    weights_df.columns = symbols
    # Set index to timestamps for explicit alignment
    weights_df.index = pd.to_datetime(results_df['timestamp'], utc=True)

    # Asset returns from price DataFrame
    price_df = close_prices_daily.copy()
    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], utc=True)
    price_df = price_df.set_index('timestamp')
    asset_returns = price_df[symbols].pct_change()

    # Align indexes explicitly (reindex asset_returns to weights_df index)
    asset_returns = asset_returns.reindex(weights_df.index)

    # Shift weights so weights_prev aligns with returns at same index
    weights_prev = weights_df.shift(1)

    # Daily gross contribution per asset (explicit alignment)
    daily_gross = weights_prev * asset_returns

    # Compute per-period weight changes to allocate transaction costs
    weight_changes = weights_df.diff().abs()
    # transaction costs series (align with results_df index)
    tx_array = res.get("transaction_costs_per_period")
    # tx_array length n-1; prepend 0 for first period
    import numpy as _np
    tx_series = _np.concatenate(([0.0], tx_array))
    tx_series = pd.Series(tx_series, index=pd.to_datetime(results_df["timestamp"], utc=True)).astype(float)

    # Allocate transaction costs to assets proportionally to absolute weight changes
    alloc_costs = pd.DataFrame(0.0, index=weights_df.index, columns=symbols)
    for idx in weights_df.index:
        wc = weight_changes.loc[idx]
        total_wc = wc.sum()
        tx = float(tx_series.loc[idx]) if idx in tx_series.index else 0.0
        if total_wc > 0 and tx > 0:
            alloc = (wc / total_wc) * tx
            alloc_costs.loc[idx] = alloc.values
        else:
            # zero allocation
            alloc_costs.loc[idx] = 0.0

    # Net per-asset daily contributions = gross - allocated cost
    daily_net = daily_gross.subtract(alloc_costs)

    # CORRECTED summaries per asset (align weights with returns using weights_prev)
    summary = []
    for sym in symbols:
        # Use weights_prev because those are the weights that earned the day's returns
        wt_prev = weights_prev[sym]
        days_pos_weight = (wt_prev > 0).sum()
        days_neg_weight = (wt_prev < 0).sum()
        days_zero_weight = (wt_prev == 0).sum()

        # Average weight (across all days, including zeros) of weights_prev
        avg_weight_applied = wt_prev.mean()

        # Gross contributions (weights_prev * returns)
        gross_series = (wt_prev * asset_returns[sym]).fillna(0.0)
        total_gross_contrib = gross_series.sum()

        # Positive/negative contribution days but only when weight_prev != 0
        mask_exposed = wt_prev != 0
        pos_contrib_days = ((gross_series > 0) & mask_exposed).sum()
        neg_contrib_days = ((gross_series < 0) & mask_exposed).sum()

        summary.append(
            {
                "symbol": sym,
                "days_weight_pos": int(days_pos_weight),
                "days_weight_neg": int(days_neg_weight),
                "days_weight_zero": int(days_zero_weight),
                "avg_weight_applied": float(avg_weight_applied),
                "total_gross_contrib": float(total_gross_contrib),
                "pos_contrib_days": int(pos_contrib_days),
                "neg_contrib_days": int(neg_contrib_days),
            }
        )

    summary_df = pd.DataFrame(summary).set_index("symbol")

    # Portfolio-level totals
    portfolio_total_return = res["metrics"]["total_return"]
    total_tx = tx_series.sum()
    gross_total = daily_gross.sum(axis=1).sum()
    net_total = daily_net.sum(axis=1).sum()

    print('\n=== ATTRIBUTION: per-asset contributions (30-day period) ===')
    cols = ['days_weight_pos','days_weight_neg','days_weight_zero','avg_weight_applied','total_gross_contrib','pos_contrib_days','neg_contrib_days']
    print(summary_df[cols])

    # Save corrected attribution CSV
    summary_df[cols].to_csv(Path('research') / 'attribution_summary_corrected.csv')

    print('\nReconciliation:')
    print(f"Portfolio total return (from backtester): {portfolio_total_return:.6f}")
    print(f"Sum of per-asset net contributions (additive approx): {net_total:.6f}")
    print(f"Sum of per-asset gross contributions: {gross_total:.6f}")
    print(f"Total transaction costs (sum over periods): {total_tx:.6f}")

    # Top 3 contributors by total contribution
    top3 = summary_df['total_gross_contrib'].sort_values(ascending=False).head(3)
    print('\nTop 3 contributors:')
    for s, val in top3.items():
        print(f'  {s}: {val:.6f}')

    # Print exact daily weights for ADA/USD and DOGE/USD (weights_prev)
    ada_weights = weights_prev['ADA/USD']
    doge_weights = weights_prev['DOGE/USD']
    print('\nADA/USD daily weights (weights applied for each day\'s returns):')
    print(ada_weights.to_string())
    print('\nDOGE/USD daily weights (weights applied for each day\'s returns):')
    print(doge_weights.to_string())

    # Interpretation
    total_positive_assets = (summary_df['total_gross_contrib'] > 0).sum()
    if total_positive_assets >= 4:
        interpretation = 'Return broadly distributed across assets'
    else:
        interpretation = 'Return concentrated in few assets'

    print('\nInterpretation:')
    print(interpretation)




def main():
    out_path = Path('research') / 'strategy_equity_daily.csv'

    if out_path.exists():
        print(f'Found existing equity CSV at {out_path}; running diagnostics from it.')
        diagnostics_from_csv(out_path)
        # Also run half-sample robustness test
        half_sample_robustness_test()
        # Run attribution analysis
        attribution_analysis()
        return

    # If CSV does not exist, run backtests and save CSV, then run diagnostics and robustness
    print("CSV not found; running backtests to generate equity CSV.")

    loader = DataLoader(Path("data"))
    symbol_data = loader.load_all_symbols()

    # Resample each symbol to daily
    resampled = {}
    for symbol, df in symbol_data.items():
        df_daily = Resampler.resample_ohlcv(df, symbol, "1D")
        resampled[symbol] = df_daily
        print(f"  Resampled {symbol}: {len(df_daily)} daily rows")

    aligner = DataAligner(resampled)
    close_prices_daily = aligner.get_close_prices()

    rev_result = run_and_report(close_prices_daily, one_day_reversal_strategy, "One-Day Reversal")
    mom_result = run_and_report(close_prices_daily, one_day_momentum_strategy, "One-Day Momentum")

    df_rev = rev_result["results_df"][ ["timestamp", "portfolio_value"] ].rename(columns={"portfolio_value": "reversal_equity"})
    df_mom = mom_result["results_df"][ ["timestamp", "portfolio_value"] ].rename(columns={"portfolio_value": "momentum_equity"})

    merged = pd.merge(df_rev, df_mom, on="timestamp", how="outer")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Saved combined daily equity curves to: {out_path}")

    diagnostics_from_csv(out_path)
    half_sample_robustness_test()


if __name__ == "__main__":
    main()
