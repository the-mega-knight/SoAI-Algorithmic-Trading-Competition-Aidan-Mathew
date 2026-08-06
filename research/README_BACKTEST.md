# Research Backtester Framework

A clean, reusable backtesting engine for quantitative research on cryptocurrency trading strategies.

## Overview

The framework is designed to:
- Load and validate OHLCV data from multiple symbols
- Align disparate symbol timelines to a common 1-minute grid
- Resample 1-minute data to arbitrary intervals (5min, 15min, 1H, 1D, etc.)
- Backtest custom strategies with proper forward-looking bias prevention
- Calculate comprehensive performance metrics
- Generate visualization plots

**No machine learning is used. No parameters are optimized. Strategy logic is completely separate from the backtesting engine.**

---

## Architecture

### Core Classes

#### `DataLoader`
Loads and validates OHLCV CSV files from the `data/` directory.

```python
loader = DataLoader(data_dir=Path("data"))
symbol_data = loader.load_all_symbols()  # Returns Dict[str, DataFrame]
```

- Automatically skips `EXAMPLE_1m_spot.csv`
- Validates required columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`
- Reports duplicate timestamps and missing data
- Parses timestamps as UTC

#### `DataAligner`
Aligns multiple symbols to a common 1-minute timeline.

```python
aligner = DataAligner(symbol_data)
close_prices = aligner.get_close_prices()  # DataFrame with timestamp + close prices
```

- Creates union of all timestamps across all symbols
- Forward-fills missing prices (handles symbols that don't trade every minute)
- Returns a clean, aligned DataFrame ready for backtesting

#### `Resampler`
Resamples 1-minute OHLCV data to different timeframes.

```python
ohlcv_5min = Resampler.resample_ohlcv(df_1min, symbol="BTC/USD", timeframe="5min")
ohlcv_1h = Resampler.resample_ohlcv(df_1min, symbol="BTC/USD", timeframe="60min")
ohlcv_1d = Resampler.resample_ohlcv(df_1min, symbol="BTC/USD", timeframe="1D")
```

Supports any pandas resample frequency: `5min`, `15min`, `60min`, `1D`, `1W`, etc.

#### `PerformanceMetrics`
Static utility functions for calculating metrics and statistics.

```python
# Returns calculations
returns = PerformanceMetrics.returns(prices)
log_returns = PerformanceMetrics.log_returns(prices)

# Volatility
rolling_vol = PerformanceMetrics.rolling_volatility(returns, window=20)
annual_vol = PerformanceMetrics.annualized_volatility(returns)

# Risk metrics
sharpe = PerformanceMetrics.sharpe_ratio(returns, risk_free_rate=0.0)
max_dd = PerformanceMetrics.max_drawdown(equity)
dd_series = PerformanceMetrics.drawdown_series(equity)

# Performance
total_ret = PerformanceMetrics.total_return(prices)
cagr = PerformanceMetrics.cagr(prices, years=1.0)

# Portfolio
turnover = PerformanceMetrics.turnover(weights_df)
tran_cost = PerformanceMetrics.transaction_costs(weights_t, weights_t_prev, cost_bps=10)

# Descriptive
stats = PerformanceMetrics.descriptive_stats(df, symbol)
```

#### `Backtester`
Core backtesting engine that applies a strategy and calculates performance.

```python
backtester = Backtester(
    close_prices=close_prices,
    initial_capital=100_000,
    transaction_cost_bps=10  # 10 basis points per trade
)

results = backtester.run(strategy_function)
```

**Critical**: The strategy receives only data up to the current point in time (no look-ahead bias).

#### `Plotter`
Generates research-quality plots.

```python
Plotter.plot_equity_and_drawdown(results_df, title="My Strategy")
Plotter.plot_weights(results_df, symbols, title="Portfolio Weights")
Plotter.plot_returns_distribution(results_df, title="Returns Distribution")
```

---

## How to Run

### Basic Run with Equal-Weight Strategy

```bash
cd /path/to/SoAI-2026-AI-Algorithmic-Trading-Competition-main
source .venv/bin/activate
python research/research_backtest.py
```

This demonstrates the framework with a trivial equal-weight strategy. It will:
1. Load all `*_1m_spot.csv` files from `data/`
2. Align them to a common timeline
3. Print descriptive statistics for each symbol
4. Run a backtest with equal weights across all 8 symbols
5. Print performance metrics
6. Save results to `research/backtest_results.csv`
7. Generate three PNG plots

**Execution time**: ~2-3 minutes on a modern Mac

---

## Output Files

After running `research_backtest.py`:

### `research/backtest_results.csv`
Main backtest results with one row per 1-minute candle:

```
timestamp,portfolio_value,ADA/USD_weight,AVAX/USD_weight,...,returns,drawdown
2026-07-06 09:43:00+00:00,100000.0,0.125,0.125,...,NaN,0.0
2026-07-06 09:44:00+00:00,99906.08,0.125,0.125,...,-0.000939,-0.000939
...
```

- **timestamp**: UTC datetime
- **portfolio_value**: Portfolio equity at this time
- **{SYMBOL}_weight**: Portfolio weight for each symbol (0.125 = 12.5%)
- **returns**: Period return
- **drawdown**: Current drawdown from peak

### `research/equity_drawdown.png`
Two-panel plot:
- **Top**: Portfolio equity curve over time
- **Bottom**: Drawdown series (% below peak)

### `research/weights.png`
Time series of portfolio weights for each symbol.

### `research/returns_dist.png`
Histogram of 1-minute returns with mean and median lines.

---

## Writing Custom Strategies

Strategies are simple functions that take a (partial) DataFrame and return portfolio weights:

```python
def my_strategy(close_prices_df: pd.DataFrame, row_idx: int) -> pd.Series:
    """
    Compute portfolio weights for the next period.
    
    Args:
        close_prices_df: DataFrame with all data up to (not including) row_idx
                         Columns: timestamp, symbol1, symbol2, ...
        row_idx: Current row index (for reference)
    
    Returns:
        pd.Series with index=symbol_names and values=weights (sum=1.0)
        Example: pd.Series([0.5, 0.3, 0.2], index=["BTC/USD", "ETH/USD", "SOL/USD"])
    """
    symbols = [c for c in close_prices_df.columns if c != "timestamp"]
    
    # Example: Buy if momentum is positive
    prices = close_prices_df[symbols].iloc[-20:]  # Last 20 candles
    returns = prices.pct_change().mean()
    
    # Long winners, short losers (simplified momentum)
    weights = returns - returns.mean()
    weights = weights / weights.abs().sum()  # Normalize to sum to 1
    weights = weights.clip(-0.5, 0.5)  # Limit position sizes
    weights = weights / weights.abs().sum()  # Re-normalize
    
    return pd.Series(weights, index=symbols)
```

### Key Rules for Strategy Functions

1. **NO LOOK-AHEAD BIAS**: The function receives `close_prices_df[:row_idx]` (not including current row)
2. **Return normalized weights**: Must sum to 1.0 (or close to it)
3. **Handle all symbols**: Return a weight for each symbol in the index
4. **Vectorized NumPy/Pandas**: Use NumPy/Pandas operations, not loops where possible

### Using Custom Strategy

```python
from research_backtest import Backtester, DataLoader, DataAligner
from pathlib import Path

# Load and align
loader = DataLoader(Path("data"))
symbol_data = loader.load_all_symbols()
aligner = DataAligner(symbol_data)
close_prices = aligner.get_close_prices()

# Run backtest
backtester = Backtester(close_prices, initial_capital=100_000)
results = backtester.run(my_strategy)

# Access results
print(results["metrics"])
results_df = results["results_df"]
```

---

## Performance Metrics Explained

### Total Return
`(final_value - initial_capital) / initial_capital`

Cumulative return over the entire period. For our 30-day backtest, this is typically small.

### CAGR (Compound Annual Growth Rate)
Annualized return assuming the 30-day performance continued for a full year. **Do not trust CAGR on short backtest periods** — it's just for reference.

### Annualized Volatility
Standard deviation of returns × √252 (trading days per year).

### Sharpe Ratio
`(mean_excess_return / std_excess_return) × √252`

Return per unit of risk. Higher is better. Values > 1 are generally good; > 2 is excellent.

### Max Drawdown
Largest peak-to-trough decline in equity. Our equal-weight strategy shows -8.32% max drawdown.

### Average Turnover
Average absolute weight change per period. 0% means buy-and-hold. Higher turnover = more trading.

### Average Transaction Cost
Impact of transaction costs on returns. At 10 bps per trade with 0% turnover, transaction cost is zero.

---

## Data Quality Notes

- **Alignment**: Not all symbols trade at every 1-minute mark. The framework forward-fills missing prices.
- **Coverage**: Different symbols have different data coverage:
  - BTC/USD, ETH/USD, SOL/USD: 43,200 candles (~30 days)
  - ADA/USD: 35,165 candles (~24.4 days)
  - AVAX/USD: 18,601 candles (~12.9 days)
- **Common timeline**: The union of all timestamps is 43,198 minutes (30 days starting 2026-07-06 09:43 UTC)

---

## Extending the Framework

### Add Resampling to Your Strategy

```python
from research_backtest import Resampler

# Resample 1-minute data to 15-minute bars
symbol_data_15m = {}
for symbol, df_1m in symbol_data.items():
    symbol_data_15m[symbol] = Resampler.resample_ohlcv(df_1m, symbol, "15min")

# Rebuild aligned data from 15-minute bars
aligner_15m = DataAligner(symbol_data_15m)
close_prices_15m = aligner_15m.get_close_prices()

backtester = Backtester(close_prices_15m, initial_capital=100_000)
results = backtester.run(my_strategy)
```

### Add Custom Metrics

```python
from research_backtest import PerformanceMetrics

results_df = results["results_df"]
daily_returns = results_df.set_index("timestamp")["returns"].resample("1D").sum()

# Calculate daily Sharpe
daily_sharpe = PerformanceMetrics.sharpe_ratio(daily_returns)

# Calculate VaR (Value at Risk, 95% confidence)
var_95 = results_df["returns"].quantile(0.05)
```

### Sector Rotation or Multi-Asset

The framework handles any number of assets. You can add forex, stocks, or commodities as long as you have OHLCV data:

```
data/EURUSD_1m_spot.csv
data/AAPL_1m_spot.csv
data/GC_1m_spot.csv  # Gold futures
```

The backtester will automatically discover and align them.

---

## Common Pitfalls

1. **Look-ahead bias**: Don't use `close_prices_df.iloc[row_idx]` in your strategy. Use only `close_prices_df.iloc[:row_idx]`.

2. **Weights don't sum to 1**: The backtester doesn't enforce this—if you return weights that don't sum to 1, the portfolio will be over/under-invested.

3. **Fractional position sizes**: You can use fractional weights (e.g., 0.01 for 1% position). Transaction costs apply proportionally.

4. **NaN handling**: The framework handles NaN prices via forward-filling, but any custom indicators must handle NaN properly.

5. **Overfitting**: This is research-only. Do NOT optimize parameters on the same data you'll test on. Use walk-forward analysis.

---

## Performance Tips

- **Vectorize**: Use NumPy/Pandas operations. Loops are slow.
- **Cache data**: Load data once, reuse across multiple backtests.
- **Profile**: Use Python's `cProfile` to identify bottlenecks.
- **Smaller universes**: Start with 2-3 symbols before scaling to 8.

---

## Questions?

- Check the example `equal_weight_strategy()` function for a baseline
- Read the docstrings in `research_backtest.py`
- Run with print statements to debug your strategy

---

**Last updated**: 2026-08-05  
**Framework version**: 1.0  
**Python**: 3.9+  
**Dependencies**: pandas, numpy, matplotlib, scipy
