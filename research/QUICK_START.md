# Quick Start: Research Backtester

## TL;DR - Run It Now

```bash
cd /path/to/SoAI-2026-AI-Algorithmic-Trading-Competition-main
source .venv/bin/activate
python research/research_backtest.py
```

This takes ~2-3 minutes and outputs:
- `backtest_results.csv` - Full backtest time series
- `equity_drawdown.png` - Portfolio curve and drawdown chart
- `weights.png` - Asset weights over time
- `returns_dist.png` - Return distribution histogram

## Files in `research/`

| File | Purpose |
|------|---------|
| `research_backtest.py` | Main backtester framework (do not modify for now) |
| `download_crypto.py` | Downloads OHLCV from Coinbase (already ran) |
| `test_ccxt.py` | Connectivity test (already ran) |
| `backtest_results.csv` | Output: Full backtest results |
| `equity_drawdown.png` | Output: Equity curve & drawdown |
| `weights.png` | Output: Portfolio weights |
| `returns_dist.png` | Output: Return distribution |
| `README_BACKTEST.md` | Full documentation |

## What Just Happened

1. **Data Loading**: Loaded 8 crypto assets from CSV files
2. **Alignment**: Aligned them to a common 1-minute UTC timeline
3. **Backtesting**: Ran equal-weight strategy (12.5% per asset) for 30 days
4. **Results**: Generated metrics and plots

## Key Results (Equal-Weight Strategy)

```
Total Return:              -0.49%
CAGR:                      -6.02% (annualized for reference)
Annualized Volatility:     0.81%
Sharpe Ratio:              0.00
Max Drawdown:              -8.32%
Average Turnover:          0.00% (buy-and-hold)
Final Value:               $99,507.91 (started with $100,000)
```

The equal-weight strategy slightly underperformed due to market conditions in the 30-day window.

## Next Steps

### Option A: Test Your Own Strategy
Edit `research_backtest.py` and replace `equal_weight_strategy()`:

```python
def my_strategy(close_prices_df: pd.DataFrame, row_idx: int) -> pd.Series:
    symbols = [c for c in close_prices_df.columns if c != "timestamp"]
    
    # Your logic here (momentum, reversal, etc.)
    # Must return pd.Series with weights for each symbol
    
    return weights_series
```

Then run:
```bash
python research/research_backtest.py
```

### Option B: Test Different Data Frequency
Resample to 5-minute bars or 1-hour bars in your strategy:

```python
from research_backtest import Resampler, DataLoader, DataAligner

loader = DataLoader(Path("data"))
symbol_data = loader.load_all_symbols()

# Resample to 15-minute
symbol_data_15m = {}
for symbol, df in symbol_data.items():
    symbol_data_15m[symbol] = Resampler.resample_ohlcv(df, symbol, "15min")

aligner = DataAligner(symbol_data_15m)
close_prices = aligner.get_close_prices()

backtester = Backtester(close_prices)
results = backtester.run(my_strategy)
```

### Option C: Expand the Asset Universe
Add more symbols to `data/` by modifying `research/download_crypto.py` and re-running it.

## Example: Simple Momentum Strategy

```python
def momentum_strategy(close_prices_df: pd.DataFrame, row_idx: int) -> pd.Series:
    """Buy winners, sell losers over last 60 minutes."""
    symbols = [c for c in close_prices_df.columns if c != "timestamp"]
    
    # Get last 60 minutes of prices
    lookback = close_prices_df[symbols].tail(60)
    
    # Calculate 60-minute returns
    momentum = (lookback.iloc[-1] - lookback.iloc[0]) / lookback.iloc[0]
    
    # Normalize: long winners, short losers
    # (Negative weights = short positions)
    weights = (momentum - momentum.mean()) / momentum.std()
    
    # Ensure weights sum to zero (market-neutral) or 1 (long-only)
    weights = weights.clip(-1, 1)
    weights = weights / weights.abs().sum() * 2  # Fully invested
    
    return pd.Series(weights, index=symbols)
```

## Key Framework Features

✅ No look-ahead bias (strategy only sees past data)  
✅ Automatic forward-filling of missing prices  
✅ Proper transaction cost accounting  
✅ Annualized Sharpe ratio calculation  
✅ Maximum drawdown tracking  
✅ Weights and returns tracking  
✅ Multiple plot outputs  

## Common Questions

**Q: Why is the equal-weight strategy down -0.49%?**  
A: Market conditions over the 30-day window. The crypto market was relatively flat with slight negative bias.

**Q: Can I trade shorter timeframes (e.g., 1-second)?**  
A: The data is 1-minute candles. You can't construct sub-minute strategies from this data.

**Q: Can I add machine learning?**  
A: Yes, but not yet. Build robust statistical strategies first. ML comes later.

**Q: Can I use this for live trading?**  
A: No. This is research-only. The real strategy in `strategies/strategy.py` will use Lumibot for live execution.

**Q: How do I avoid overfitting?**  
A: Use different time windows for development vs. testing. Never optimize parameters on the data you'll evaluate on.

---

For full documentation, see `README_BACKTEST.md`.
