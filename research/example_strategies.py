"""
Example Research Strategies

This file shows how to write strategies for the research_backtest.py framework.
Copy these functions and modify them for your own research.

KEY RULE: No look-ahead bias!
Your strategy receives ONLY data up to (not including) the current row.
"""

import numpy as np
import pandas as pd


def equal_weight_strategy(close_prices_df: pd.DataFrame, row_idx: int) -> pd.Series:
    """
    Equal-weight portfolio (buy-and-hold baseline).
    
    This is the simplest possible strategy: invest equal amounts in each asset.
    It's useful as a baseline to compare more complex strategies against.
    """
    symbols = [c for c in close_prices_df.columns if c != "timestamp"]
    n_symbols = len(symbols)
    weights = pd.Series(1.0 / n_symbols, index=symbols)
    return weights


def momentum_strategy(close_prices_df: pd.DataFrame, row_idx: int) -> pd.Series:
    """
    Momentum strategy: Buy winners, sell losers.
    
    This strategy looks at 60-minute returns and overweights assets that
    have recently outperformed the average.
    """
    symbols = [c for c in close_prices_df.columns if c != "timestamp"]
    
    # Ensure we have enough history
    if len(close_prices_df) < 60:
        # Not enough data yet, return equal weight
        n_symbols = len(symbols)
        return pd.Series(1.0 / n_symbols, index=symbols)
    
    # Get last 60 minutes of prices
    lookback_prices = close_prices_df[symbols].tail(60)
    
    # Calculate 60-minute returns for each symbol
    start_price = lookback_prices.iloc[0]
    end_price = lookback_prices.iloc[-1]
    returns_60m = (end_price - start_price) / start_price
    
    # Normalize: deviation from mean return
    avg_return = returns_60m.mean()
    momentum = returns_60m - avg_return
    
    # Convert to weights
    # Positive momentum -> long positions
    # Negative momentum -> short positions (or underweight)
    if momentum.std() > 0:
        weights = momentum / momentum.std()
    else:
        weights = momentum
    
    # Clip extreme positions and normalize
    weights = weights.clip(-1.0, 1.0)
    
    # Ensure weights sum to 1 (fully invested)
    abs_sum = weights.abs().sum()
    if abs_sum > 0:
        weights = weights / abs_sum
    else:
        # No momentum detected, revert to equal weight
        n_symbols = len(symbols)
        weights = pd.Series(1.0 / n_symbols, index=symbols)
    
    return pd.Series(weights, index=symbols)


def reversal_strategy(close_prices_df: pd.DataFrame, row_idx: int) -> pd.Series:
    """
    Reversal strategy: Buy losers, sell winners.
    
    Mean-reversion assumes assets that have underperformed will bounce back.
    This is the opposite of momentum.
    """
    symbols = [c for c in close_prices_df.columns if c != "timestamp"]
    
    if len(close_prices_df) < 60:
        n_symbols = len(symbols)
        return pd.Series(1.0 / n_symbols, index=symbols)
    
    # Get last 60 minutes of prices
    lookback_prices = close_prices_df[symbols].tail(60)
    
    # Calculate 60-minute returns
    start_price = lookback_prices.iloc[0]
    end_price = lookback_prices.iloc[-1]
    returns_60m = (end_price - start_price) / start_price
    
    # REVERSED: Underweighting winners, overweighting losers
    avg_return = returns_60m.mean()
    reversal = -(returns_60m - avg_return)  # Note the negative sign
    
    if reversal.std() > 0:
        weights = reversal / reversal.std()
    else:
        weights = reversal
    
    weights = weights.clip(-1.0, 1.0)
    abs_sum = weights.abs().sum()
    if abs_sum > 0:
        weights = weights / abs_sum
    else:
        n_symbols = len(symbols)
        weights = pd.Series(1.0 / n_symbols, index=symbols)
    
    return pd.Series(weights, index=symbols)


def volatility_adjusted_strategy(close_prices_df: pd.DataFrame, row_idx: int) -> pd.Series:
    """
    Volatility-adjusted strategy: Higher volatility = lower allocation.
    
    Risk-parity approach: Allocate more capital to lower-volatility assets
    to achieve equal risk contribution from each holding.
    """
    symbols = [c for c in close_prices_df.columns if c != "timestamp"]
    
    if len(close_prices_df) < 20:
        n_symbols = len(symbols)
        return pd.Series(1.0 / n_symbols, index=symbols)
    
    # Calculate rolling volatility over last 20 periods
    lookback_prices = close_prices_df[symbols].tail(20)
    returns = lookback_prices.pct_change().dropna()
    
    if len(returns) < 2:
        n_symbols = len(symbols)
        return pd.Series(1.0 / n_symbols, index=symbols)
    
    volatilities = returns.std()
    
    # Inverse volatility: Lower vol = higher weight
    inv_vol = 1.0 / (volatilities + 1e-8)  # Avoid division by zero
    
    # Normalize to sum to 1
    weights = inv_vol / inv_vol.sum()
    
    return pd.Series(weights, index=symbols)


def channel_breakout_strategy(close_prices_df: pd.DataFrame, row_idx: int) -> pd.Series:
    """
    Channel breakout strategy: Buy on breakouts, sell on breakdowns.
    
    Identifies 20-period high/low channels and takes positions on breakouts.
    """
    symbols = [c for c in close_prices_df.columns if c != "timestamp"]
    
    if len(close_prices_df) < 20:
        n_symbols = len(symbols)
        return pd.Series(1.0 / n_symbols, index=symbols)
    
    lookback_prices = close_prices_df[symbols].tail(20)
    current_price = lookback_prices.iloc[-1]
    
    # Calculate 20-period high and low
    period_high = lookback_prices.max()
    period_low = lookback_prices.min()
    
    # Identify breakouts
    breakout_signal = pd.Series(index=symbols)
    for symbol in symbols:
        price = current_price[symbol]
        high = period_high[symbol]
        low = period_low[symbol]
        
        if price >= high * 0.95:  # Near or at 20-period high
            breakout_signal[symbol] = 1.0  # Bullish breakout
        elif price <= low * 1.05:  # Near or at 20-period low
            breakout_signal[symbol] = -1.0  # Bearish breakdown
        else:
            breakout_signal[symbol] = 0.0  # No clear signal
    
    # Normalize to weights
    abs_sum = breakout_signal.abs().sum()
    if abs_sum > 0:
        weights = breakout_signal / abs_sum
    else:
        n_symbols = len(symbols)
        weights = pd.Series(1.0 / n_symbols, index=symbols)
    
    return pd.Series(weights, index=symbols)


def random_walk_strategy(close_prices_df: pd.DataFrame, row_idx: int) -> pd.Series:
    """
    Random walk strategy: Random weights (for testing framework only).
    
    This is purely for framework validation. Performance will be close to zero
    after accounting for transaction costs.
    """
    symbols = [c for c in close_prices_df.columns if c != "timestamp"]
    
    # Generate random weights
    random_weights = np.random.dirichlet(np.ones(len(symbols)))
    
    return pd.Series(random_weights, index=symbols)


def pairs_trading_strategy(close_prices_df: pd.DataFrame, row_idx: int) -> pd.Series:
    """
    Pairs trading: Identify divergent pairs and trade the spread.
    
    This is a simplified example. Long the underperformer, short the outperformer
    when their returns diverge significantly.
    """
    symbols = [c for c in close_prices_df.columns if c != "timestamp"]
    
    if len(close_prices_df) < 20:
        n_symbols = len(symbols)
        return pd.Series(1.0 / n_symbols, index=symbols)
    
    # Calculate recent returns
    lookback_prices = close_prices_df[symbols].tail(20)
    returns_20 = (lookback_prices.iloc[-1] - lookback_prices.iloc[0]) / lookback_prices.iloc[0]
    
    # Find best and worst performers
    best = returns_20.idxmax()
    worst = returns_20.idxmin()
    
    # Create weights: long worst (reversal bet), short best (mean reversion bet)
    weights = pd.Series(0.0, index=symbols)
    weights[worst] = 0.5
    weights[best] = -0.5
    
    return weights


def composite_strategy(close_prices_df: pd.DataFrame, row_idx: int) -> pd.Series:
    """
    Composite strategy: Combine multiple signals.
    
    This demonstrates how to blend multiple strategies:
    - 50% momentum
    - 30% reversal
    - 20% volatility-adjusted
    """
    symbols = [c for c in close_prices_df.columns if c != "timestamp"]
    
    if len(close_prices_df) < 60:
        n_symbols = len(symbols)
        return pd.Series(1.0 / n_symbols, index=symbols)
    
    # Get individual strategy weights
    w_momentum = momentum_strategy(close_prices_df, row_idx)
    w_reversal = reversal_strategy(close_prices_df, row_idx)
    w_vol = volatility_adjusted_strategy(close_prices_df, row_idx)
    
    # Combine with different weights
    weights = (
        0.50 * w_momentum +
        0.30 * w_reversal +
        0.20 * w_vol
    )
    
    # Normalize
    abs_sum = weights.abs().sum()
    if abs_sum > 0:
        weights = weights / abs_sum
    else:
        n_symbols = len(symbols)
        weights = pd.Series(1.0 / n_symbols, index=symbols)
    
    return weights


# =====================================================================
# Usage Example
# =====================================================================

if __name__ == "__main__":
    """
    To use these strategies in research_backtest.py:
    
    1. Copy this file or import from it
    2. In research_backtest.py, change the main() function:
    
        # results = backtester.run(equal_weight_strategy)  # Old
        results = backtester.run(momentum_strategy)         # New
        
    3. Run: python research/research_backtest.py
    """
    
    print("Strategy definitions loaded.")
    print("See research_backtest.py to select which strategy to use.")
