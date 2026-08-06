"""
Transaction-cost robustness test comparing 1-day vs 3-day momentum.

Runs 6 cases:
1) 1-day, 10 bps
2) 3-day, 10 bps
3) 1-day, 20 bps
4) 3-day, 20 bps
5) 1-day, 50 bps
6) 3-day, 50 bps

Produces research/cost_robustness.csv with metrics for each case and prints concise conclusions.

Run: ./.venv/bin/python3 research/cost_robustness.py
"""
from pathlib import Path
import importlib.util
import inspect
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

# Load baseline strategy from strategy_tests.py to reuse exactly
_st_path = Path(__file__).resolve().parents[0] / "strategy_tests.py"
_spec2 = importlib.util.spec_from_file_location("strategy_tests", str(_st_path))
_st = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_st)

one_day_momentum_strategy = _st.one_day_momentum_strategy

# Implement 3-day momentum identically to prior script
def three_day_momentum_strategy(close_prices_df: pd.DataFrame, current_row_idx: int) -> pd.Series:
    symbols = [c for c in close_prices_df.columns if c != "timestamp"]
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

# Utility to run backtest with explicit transaction cost
def run_bt_with_cost(close_prices_daily, strategy_fn, cost_bps):
    bt = Backtester(close_prices_daily, transaction_cost_bps=cost_bps)
    res = bt.run(strategy_fn)
    return res

# Metric collector
def collect_metrics(res):
    m = res['metrics']
    returns = res['results_df']['returns'].dropna()
    pos = int((returns > 0).sum())
    neg = int((returns < 0).sum())
    return {
        'total_return': m['total_return'],
        'sharpe': m['sharpe_ratio'],
        'max_drawdown': m['max_drawdown'],
        'annualized_volatility': m['annualized_volatility'],
        'average_turnover': m['average_turnover'],
        'final_portfolio_value': m['final_value'],
        'positive_days': pos,
        'negative_days': neg,
    }

# Load and resample once
loader = DataLoader(Path('data'))
symbol_data = loader.load_all_symbols()
resampled = {}
for symbol, df in symbol_data.items():
    resampled[symbol] = Resampler.resample_ohlcv(df, symbol, '1D')
aligner = DataAligner(resampled)
close_prices_daily = aligner.get_close_prices()

# Validation: dataset/date range
date_range = (close_prices_daily['timestamp'].iloc[0].date(), close_prices_daily['timestamp'].iloc[-1].date())
print('Dataset range:', date_range[0], '->', date_range[1])
print('Daily observations:', len(close_prices_daily))
print('Number of assets:', len([c for c in close_prices_daily.columns if c!='timestamp']))

# Validate strategy code uses only historical rows using simple source-inspection
one_src = inspect.getsource(one_day_momentum_strategy)
three_src = inspect.getsource(three_day_momentum_strategy)

checks = {}
checks['1d_uses_iloc_-1_-2'] = ('iloc[-1]' in one_src and 'iloc[-2]' in one_src)
checks['3d_uses_iloc_-1_-4'] = ('iloc[-1]' in three_src and 'iloc[-4]' in three_src)

# Prepare cases
cases = [
    ('momentum_1d', one_day_momentum_strategy, 10),
    ('momentum_3d', three_day_momentum_strategy, 10),
    ('momentum_1d', one_day_momentum_strategy, 20),
    ('momentum_3d', three_day_momentum_strategy, 20),
    ('momentum_1d', one_day_momentum_strategy, 50),
    ('momentum_3d', three_day_momentum_strategy, 50),
]

results = []
for name, fn, cost in cases:
    # Run backtest with explicit cost
    res = run_bt_with_cost(close_prices_daily, fn, cost)
    met = collect_metrics(res)
    row = {
        'strategy': name,
        'transaction_cost_bps': cost,
    }
    row.update(met)
    results.append(row)

# Compute differences between 3d and 1d at each cost
df = pd.DataFrame(results)
# pivot for easier diff
pairs = []
for cost in [10,20,50]:
    r1 = df[(df['strategy']=='momentum_1d') & (df['transaction_cost_bps']==cost)].iloc[0]
    r3 = df[(df['strategy']=='momentum_3d') & (df['transaction_cost_bps']==cost)].iloc[0]
    diff = float(r3['total_return']) - float(r1['total_return'])
    pairs.append({'cost_bps': cost, 'total_return_diff_3d_minus_1d': diff, 'momentum_1d_return': float(r1['total_return']), 'momentum_3d_return': float(r3['total_return'])})

pairs_df = pd.DataFrame(pairs)

# Save results
out = Path('research') / 'cost_robustness.csv'
df.to_csv(out, index=False)
print('\nSaved results to', out)

# Print concise outputs
print('\nTransaction-cost robustness results:')
print(df[['strategy','transaction_cost_bps','total_return','sharpe','max_drawdown','annualized_volatility','average_turnover','final_portfolio_value','positive_days','negative_days']])

print('\nDifferences (3d - 1d) by cost:')
print(pairs_df)

# Conclusions
for _, r in pairs_df.iterrows():
    cost = int(r['cost_bps'])
    diff = r['total_return_diff_3d_minus_1d']
    print(f'At {cost} bps: 3d minus 1d total_return = {diff:.6f}')

# Answer questions
# 1. Does 3-day momentum still outperform 1-day at 10,20,50 bps?
outperform = {int(r['cost_bps']): (r['total_return_diff_3d_minus_1d']>0) for _,r in pairs_df.iterrows()}
# 2. Is 3-day momentum profitable at 50 bps?
r50 = df[(df['strategy']=='momentum_3d') & (df['transaction_cost_bps']==50)].iloc[0]
prof_50 = r50['total_return']>0

print('\nFinal answers:')
for cost in [10,20,50]:
    print(f'3-day outperforms 1-day at {cost} bps: {outperform[cost]}')
print(f'3-day profitable at 50 bps: {bool(prof_50)}')

# Does test suggest 3-day result an artifact of low costs? simple heuristic: if outperforms only at very low cost but not at higher, then yes
artifact = not (outperform[10] and outperform[20] and outperform[50])
print(f'3-day result appears to be an artifact of low costs: {artifact}')

if not all(checks.values()):
    print('\nWARNING: Source-inspection checks failed: ', checks)
else:
    print('\nSource-inspection checks passed (1-day uses iloc[-1], iloc[-2]; 3-day uses iloc[-1], iloc[-4])')

print('\nDone.')
