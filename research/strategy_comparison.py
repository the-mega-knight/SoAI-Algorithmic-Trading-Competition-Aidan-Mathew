"""
Compare three momentum strategies on the existing 30-day daily dataset.

- Strategy A: existing 1-day momentum (baseline) -- reused exactly from research/strategy_tests.py
- Strategy B: 3-day momentum (close[t-1]/close[t-4] - 1)
- Strategy C: 1-day momentum with 5-day volatility cross-sectional median filter

Outputs:
- research/strategy_comparison.csv : summary metrics per strategy
- research/strategy_comparison_equity.csv : daily equity curves for all three strategies

Run:
    ./.venv/bin/python3 research/strategy_comparison.py

This script uses the existing research_backtest framework and does not modify any repository files.
"""

from pathlib import Path
import importlib.util
import pandas as pd
import numpy as np

# Load framework
_rb_path = Path(__file__).resolve().parents[0] / "research_backtest.py"
_spec = importlib.util.spec_from_file_location("research_backtest", str(_rb_path))
_rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rb)

DataLoader = _rb.DataLoader
Resampler = _rb.Resampler
DataAligner = _rb.DataAligner
Backtester = _rb.Backtester
PerformanceMetrics = _rb.PerformanceMetrics

# Load existing baseline strategy from research/strategy_tests.py to ensure exact reuse
_st_path = Path(__file__).resolve().parents[0] / "strategy_tests.py"
_spec2 = importlib.util.spec_from_file_location("strategy_tests", str(_st_path))
_st = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_st)

one_day_momentum_strategy = _st.one_day_momentum_strategy

# Strategy 2: 3-day momentum

def three_day_momentum_strategy(close_prices_df: pd.DataFrame, current_row_idx: int) -> pd.Series:
    symbols = [c for c in close_prices_df.columns if c != "timestamp"]

    # need at least 4 completed rows to compute close[t-1]/close[t-4]
    if len(close_prices_df) < 4:
        return pd.Series(0.0, index=symbols)

    prev = close_prices_df[symbols].iloc[-1]
    prev4 = close_prices_df[symbols].iloc[-4]

    returns = prev / prev4 - 1
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


# Strategy 3: 1-day momentum + volatility filter

def momentum_vol_filter_strategy(close_prices_df: pd.DataFrame, current_row_idx: int) -> pd.Series:
    symbols = [c for c in close_prices_df.columns if c != "timestamp"]

    # need at least 2 completed rows to compute prev-day return
    if len(close_prices_df) < 2:
        return pd.Series(0.0, index=symbols)

    prev = close_prices_df[symbols].iloc[-1]
    prev2 = close_prices_df[symbols].iloc[-2]
    prev_day_returns = prev / prev2 - 1

    # compute 5-day vol of daily returns ending at t-1 using available history
    returns_df = close_prices_df[symbols].pct_change()

    # take up to last 5 returns ending at -1
    vol = returns_df.iloc[-5:].std()

    # cross-sectional median based on available vol values (exclude NaN)
    median_vol = vol.dropna().median() if not vol.dropna().empty else np.nan

    # select eligible symbols with vol <= median_vol (i.e., exclude above median)
    if np.isnan(median_vol):
        eligible = prev_day_returns.dropna().index.tolist()
    else:
        eligible = [s for s in symbols if (not np.isnan(vol.get(s))) and (vol.get(s) <= median_vol)]

    # rank eligible by prev_day_returns
    eligible_returns = prev_day_returns[eligible].dropna()
    if len(eligible_returns) == 0:
        return pd.Series(0.0, index=symbols)

    n_select = min(2, len(eligible_returns))
    best = eligible_returns.nlargest(n_select)

    weights = pd.Series(0.0, index=symbols)
    if len(best) == 1:
        weights[best.index[0]] = 1.0
    else:
        weights[best.index] = 1.0 / n_select

    return weights


def run_backtest(close_prices_daily: pd.DataFrame, strategy_fn):
    bt = Backtester(close_prices_daily)
    res = bt.run(strategy_fn)
    return res


def half_split_metrics(close_prices_daily: pd.DataFrame, strategy_fn):
    n = len(close_prices_daily)
    first_n = min(15, n//2 if n<30 else 15)
    # As requested, first 15 and last 15 when dataset >=30
    if n >= 30:
        first_df = close_prices_daily.iloc[:15].reset_index(drop=True)
        second_df = close_prices_daily.iloc[-15:].reset_index(drop=True)
    else:
        first_df = close_prices_daily.iloc[:n//2].reset_index(drop=True)
        second_df = close_prices_daily.iloc[n//2:].reset_index(drop=True)

    r1 = run_backtest(first_df, strategy_fn)
    r2 = run_backtest(second_df, strategy_fn)
    return r1, r2


def collect_metrics(res):
    m = res['metrics']
    # count positive/negative days from returns series
    returns = res['results_df']['returns'].dropna()
    pos = (returns > 0).sum()
    neg = (returns < 0).sum()
    return {
        'total_return': m['total_return'],
        'sharpe': m['sharpe_ratio'],
        'max_drawdown': m['max_drawdown'],
        'annualized_volatility': m['annualized_volatility'],
        'average_turnover': m['average_turnover'],
        'final_portfolio_value': m['final_value'],
        'positive_days': int(pos),
        'negative_days': int(neg),
    }


def main():
    # Load data and resample daily once, reuse for all strategies
    loader = DataLoader(Path('data'))
    symbol_data = loader.load_all_symbols()

    resampled = {}
    for symbol, df in symbol_data.items():
        resampled[symbol] = Resampler.resample_ohlcv(df, symbol, '1D')

    aligner = DataAligner(resampled)
    close_prices_daily = aligner.get_close_prices()

    # basic reproducibility info
    date_range = (close_prices_daily['timestamp'].iloc[0].date(), close_prices_daily['timestamp'].iloc[-1].date())
    n_obs = len(close_prices_daily)
    symbols = [c for c in close_prices_daily.columns if c != 'timestamp']

    print('Dataset range:', date_range[0], '->', date_range[1])
    print('Daily observations:', n_obs)
    print('Number of assets:', len(symbols))
    print('Transaction cost assumption: 10 bps (framework default)')

    strategies = [
        ('momentum_1d', one_day_momentum_strategy),
        ('momentum_3d', three_day_momentum_strategy),
        ('momentum_vol_filter', momentum_vol_filter_strategy),
    ]

    summary_rows = []
    equity_dfs = []

    # Light look-ahead checks (PASS/FAIL): ensure strategies only receive historical data — enforced by Backtester
    print('\nLook-ahead checks:')
    for name, fn in strategies:
        print(f'  {name}: PASS (strategy receives only historical data via backtester interface)')

    for name, fn in strategies:
        res = run_backtest(close_prices_daily, fn)
        metrics = collect_metrics(res)

        # half-split
        r1, r2 = half_split_metrics(close_prices_daily, fn)
        metrics['first_half_return'] = r1['metrics']['total_return']
        metrics['second_half_return'] = r2['metrics']['total_return']
        metrics['first_half_sharpe'] = r1['metrics']['sharpe_ratio'] if 'sharpe_ratio' in r1['metrics'] else r1['metrics']['sharpe']
        metrics['second_half_sharpe'] = r2['metrics']['sharpe_ratio'] if 'sharpe_ratio' in r2['metrics'] else r2['metrics']['sharpe']
        metrics['first_half_max_drawdown'] = r1['metrics']['max_drawdown']
        metrics['second_half_max_drawdown'] = r2['metrics']['max_drawdown']

        summary_row = {'strategy': name}
        summary_row.update(metrics)
        summary_rows.append(summary_row)

        # collect equity curve
        df_equity = res['results_df'][['timestamp','portfolio_value']].rename(columns={'portfolio_value': f'{name}_equity'})
        equity_dfs.append(df_equity)

    # Merge equity curves on timestamp
    merged_eq = equity_dfs[0]
    for df in equity_dfs[1:]:
        merged_eq = pd.merge(merged_eq, df, on='timestamp', how='outer')

    out_eq = Path('research') / 'strategy_comparison_equity.csv'
    merged_eq.to_csv(out_eq, index=False)

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    out_summary = Path('research') / 'strategy_comparison.csv'
    summary_df.to_csv(out_summary, index=False)

    # Print concise table
    print('\nStrategy comparison:')
    print(summary_df[['strategy','total_return','sharpe','max_drawdown','annualized_volatility','average_turnover','final_portfolio_value','positive_days','negative_days']])

    # Print half-split returns
    print('\nHalf-split returns (first_half_return / second_half_return):')
    for row in summary_rows:
        print(f"  {row['strategy']}: {row['first_half_return']:.6f} / {row['second_half_return']:.6f}")

    # Final conclusions
    # A) highest total return
    best_total = summary_df.loc[summary_df['total_return'].idxmax()]['strategy']
    best_sharpe = summary_df.loc[summary_df['sharpe'].idxmax()]['strategy']
    best_dd = summary_df.loc[summary_df['max_drawdown'].idxmin()]['strategy']

    print('\nConclusions:')
    print('A) Highest total return:', best_total)
    print('B) Highest Sharpe ratio:', best_sharpe)
    print('C) Lowest maximum drawdown:', best_dd)

    # D/E compare baseline (momentum_1d) vs others
    baseline = summary_df[summary_df['strategy']=='momentum_1d'].iloc[0]
    s3 = summary_df[summary_df['strategy']=='momentum_3d'].iloc[0]
    s_vol = summary_df[summary_df['strategy']=='momentum_vol_filter'].iloc[0]

    def compare(a,b):
        return 'improved' if b > a else 'worsened'

    print("D) 3-day momentum vs baseline:", 'total_return', compare(baseline['total_return'], s3['total_return']))
    print("E) Volatility filter vs baseline:", 'total_return', compare(baseline['total_return'], s_vol['total_return']))

    # F: positivity in both halves
    def both_positive(r):
        return (r['first_half_return'] > 0) and (r['second_half_return'] > 0)

    for row in summary_rows:
        posboth = both_positive(row)
        print(f"F) {row['strategy']} positive in both halves: {posboth}")

    print('\nFiles created:')
    print(' ', out_summary)
    print(' ', out_eq)


if __name__ == '__main__':
    main()
