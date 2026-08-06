"""
Research Backtester Framework

A clean, reusable backtesting engine for research purposes.
No strategy logic is implemented here—this is purely the framework.
"""

import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

# Suppress warnings
warnings.filterwarnings("ignore")

# =====================================================================
# Data Loading and Validation
# =====================================================================


class DataLoader:
    """Loads and validates OHLCV data from CSV files."""

    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = Path(data_dir)

    def load_symbol(self, filename: str) -> pd.DataFrame:
        """Load a single OHLCV CSV file."""
        path = self.data_dir / filename

        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Validate columns
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"Missing columns in {filename}: {missing}")

        # Check for duplicates
        dup_count = df["timestamp"].duplicated().sum()
        if dup_count > 0:
            print(f"  WARNING: {filename} has {dup_count} duplicate timestamps")

        # Check for missing data
        missing = df.isnull().sum()
        if missing.any():
            print(f"  WARNING: {filename} has missing values:\n{missing[missing > 0]}")

        df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    def load_all_symbols(self) -> Dict[str, pd.DataFrame]:
        """Load all *_1m_spot.csv files from data directory."""
        symbol_data = {}

        for csv_file in sorted(self.data_dir.glob("*_1m_spot.csv")):
            # Skip EXAMPLE file
            if "EXAMPLE" in csv_file.name:
                continue

            # Extract symbol from filename: BTC_USD_1m_spot.csv -> BTC/USD
            parts = csv_file.stem.split("_")
            if len(parts) >= 2:
                symbol = f"{parts[0]}/{parts[1]}"
            else:
                continue

            try:
                df = self.load_symbol(csv_file.name)
                symbol_data[symbol] = df
                print(f"  Loaded {symbol}: {len(df):,} candles")
            except Exception as e:
                print(f"  ERROR loading {csv_file.name}: {e}")

        return symbol_data


# =====================================================================
# Data Alignment and Resampling
# =====================================================================


class DataAligner:
    """Aligns multiple symbols to a common timeline."""

    def __init__(self, symbol_data: Dict[str, pd.DataFrame]):
        self.symbol_data = symbol_data

    def get_common_timeline(self) -> pd.DatetimeIndex:
        """Get the union of all timestamps across all symbols."""
        all_times = set()
        for df in self.symbol_data.values():
            all_times.update(df["timestamp"])
        return pd.DatetimeIndex(sorted(all_times), tz="UTC").sort_values()

    def align_symbols(self) -> pd.DataFrame:
        """Create a single aligned DataFrame with all symbols."""
        # Start with common timeline
        timeline = self.get_common_timeline()
        aligned = pd.DataFrame({"timestamp": timeline})

        # For each symbol, merge OHLCV data
        for symbol, df in self.symbol_data.items():
            # Rename columns to include symbol
            renamed = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
            renamed.columns = [
                "timestamp",
                f"{symbol}_open",
                f"{symbol}_high",
                f"{symbol}_low",
                f"{symbol}_close",
                f"{symbol}_volume",
            ]

            aligned = aligned.merge(renamed, on="timestamp", how="left")

        aligned = aligned.sort_values("timestamp").reset_index(drop=True)
        return aligned

    def get_close_prices(self) -> pd.DataFrame:
        """Extract close prices for all symbols, forward-filling missing values."""
        aligned = self.align_symbols()
        close_cols = [c for c in aligned.columns if c.endswith("_close")]

        df = aligned[["timestamp"] + close_cols].copy()

        # Rename columns to symbol names
        rename_map = {col: col.replace("_close", "") for col in close_cols}
        df = df.rename(columns=rename_map)

        # Forward-fill missing values (prices don't trade at every minute across all symbols)
        df = df.fillna(method="ffill")

        # Drop any remaining NaN rows at the start
        df = df.dropna()
        df = df.reset_index(drop=True)

        return df


# =====================================================================
# Resampling
# =====================================================================


class Resampler:
    """Resample OHLCV data to different timeframes."""

    @staticmethod
    def resample_ohlcv(df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Resample 1-minute OHLCV data to a different timeframe.

        Args:
            df: DataFrame with timestamp and OHLCV columns for the symbol
            symbol: Symbol name (e.g., "BTC/USD")
            timeframe: Target timeframe ("5min", "15min", "60min", "1D", etc.)

        Returns:
            Resampled DataFrame with timestamp, open, high, low, close, volume
        """
        # Create a copy with timestamp as index
        data = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        data = data.set_index("timestamp")

        # Resample
        resampled = pd.DataFrame()
        resampled["open"] = data["open"].resample(timeframe).first()
        resampled["high"] = data["high"].resample(timeframe).max()
        resampled["low"] = data["low"].resample(timeframe).min()
        resampled["close"] = data["close"].resample(timeframe).last()
        resampled["volume"] = data["volume"].resample(timeframe).sum()

        # Drop rows with missing OHLCV (due to gaps)
        resampled = resampled.dropna()

        resampled = resampled.reset_index()
        return resampled


# =====================================================================
# Performance Metrics and Utilities
# =====================================================================


class PerformanceMetrics:
    """Calculate performance metrics and statistics."""

    TRADING_DAYS_PER_YEAR = 252

    @staticmethod
    def returns(prices: pd.Series) -> pd.Series:
        """Calculate simple returns from prices."""
        return prices.pct_change()

    @staticmethod
    def log_returns(prices: pd.Series) -> pd.Series:
        """Calculate log returns from prices."""
        return np.log(prices / prices.shift(1))

    @staticmethod
    def rolling_volatility(
        returns: pd.Series, window: int = 20
    ) -> pd.Series:
        """Calculate rolling volatility (annualized)."""
        vol = returns.rolling(window).std() * np.sqrt(252)
        return vol

    @staticmethod
    def annualized_volatility(returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        return returns.std() * np.sqrt(252)

    @staticmethod
    def sharpe_ratio(
        returns: pd.Series, risk_free_rate: float = 0.0
    ) -> float:
        """Calculate Sharpe ratio (annualized)."""
        excess_returns = returns - (risk_free_rate / 252)
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    @staticmethod
    def max_drawdown(equity: pd.Series) -> float:
        """Calculate maximum drawdown."""
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        return drawdown.min()

    @staticmethod
    def drawdown_series(equity: pd.Series) -> pd.Series:
        """Calculate drawdown at each point."""
        running_max = equity.expanding().max()
        return (equity - running_max) / running_max

    @staticmethod
    def total_return(prices: pd.Series) -> float:
        """Calculate total return."""
        return (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]

    @staticmethod
    def cagr(prices: pd.Series, years: float) -> float:
        """Calculate CAGR."""
        if years <= 0:
            return 0.0
        total_ret = prices.iloc[-1] / prices.iloc[0]
        return total_ret ** (1 / years) - 1

    @staticmethod
    def turnover(weights: pd.DataFrame) -> float:
        """Calculate average turnover from weight changes."""
        if len(weights) < 2:
            return 0.0
        weight_changes = weights.diff().abs().sum(axis=1)
        return weight_changes.mean()

    @staticmethod
    def transaction_costs(
        weights_t: pd.Series, weights_t_prev: pd.Series, cost_bps: float = 10
    ) -> float:
        """Calculate transaction costs from weight changes."""
        # cost_bps: basis points (1 bps = 0.01%)
        trades = (weights_t - weights_t_prev).abs().sum() / 2
        cost = trades * (cost_bps / 10_000)
        return cost

    @staticmethod
    def descriptive_stats(df: pd.DataFrame, symbol: str) -> Dict:
        """Calculate descriptive statistics for a symbol."""
        # Original dataframes have columns: timestamp, open, high, low, close, volume
        # Not symbol-prefixed
        close_col = "close"
        volume_col = "volume"

        stats_dict = {
            "symbol": symbol,
            "observations": len(df),
            "start_date": df["timestamp"].iloc[0],
            "end_date": df["timestamp"].iloc[-1],
        }

        if close_col in df.columns:
            prices = df[close_col]
            returns = PerformanceMetrics.returns(prices)
            stats_dict["annualized_vol"] = PerformanceMetrics.annualized_volatility(returns)
            stats_dict["missing_pct"] = prices.isnull().sum() / len(prices) * 100
        else:
            stats_dict["annualized_vol"] = 0.0
            stats_dict["missing_pct"] = 100.0

        if volume_col in df.columns:
            stats_dict["avg_daily_volume"] = df[volume_col].mean()

        return stats_dict


# =====================================================================
# Backtester Engine
# =====================================================================


class Backtester:
    """
    Standalone backtesting engine.
    Strategy logic is provided via a callback function.
    """

    def __init__(
        self,
        close_prices: pd.DataFrame,
        initial_capital: float = 100_000,
        transaction_cost_bps: float = 10,
    ):
        """
        Initialize backtester.

        Args:
            close_prices: DataFrame with timestamp + close prices for each symbol
            initial_capital: Starting capital
            transaction_cost_bps: Transaction cost in basis points
        """
        self.close_prices = close_prices
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps

        # Extract symbols from column names (everything except 'timestamp')
        self.symbols = [c for c in close_prices.columns if c != "timestamp"]

    def run(
        self, strategy: Callable[[pd.DataFrame, int], pd.Series]
    ) -> Dict:
        """
        Run backtest with a given strategy.

        Args:
            strategy: Callable that takes (close_prices_df, current_row_idx) -> weights_series
                     Must return a pd.Series with index=symbols and values=portfolio weights
                     Weights should sum to 1.0 (fully invested)
                     Strategy has access ONLY to data up to current_row_idx (no look-ahead)

        Returns:
            Dictionary with backtest results
        """
        n_periods = len(self.close_prices)
        n_assets = len(self.symbols)

        # Initialize tracking arrays
        portfolio_values = np.zeros(n_periods)
        portfolio_values[0] = self.initial_capital

        weights_over_time = np.zeros((n_periods, n_assets))
        turnover_per_period = np.zeros(n_periods - 1)
        transaction_costs_per_period = np.zeros(n_periods - 1)

        # Starting weights (uniform or zero)
        prev_weights = np.zeros(n_assets)

        # Loop through time
        for t in range(1, n_periods):
            # Strategy computes weights based on data up to t-1
            # But it receives indices, so pass a view of data up to t-1
            data_so_far = self.close_prices.iloc[:t].copy()

            weights_series = strategy(data_so_far, t - 1)
            weights = weights_series.values

            # Store weights
            weights_over_time[t - 1] = weights

            # Calculate transaction costs
            tran_cost = PerformanceMetrics.transaction_costs(
                pd.Series(weights, index=self.symbols),
                pd.Series(prev_weights, index=self.symbols),
                cost_bps=self.transaction_cost_bps,
            )
            transaction_costs_per_period[t - 1] = tran_cost

            # Calculate turnover
            turnover = np.abs(weights - prev_weights).sum() / 2
            turnover_per_period[t - 1] = turnover

            # Get returns from t-1 to t
            prev_prices = self.close_prices[self.symbols].iloc[t - 1].values
            curr_prices = self.close_prices[self.symbols].iloc[t].values
            price_returns = (curr_prices - prev_prices) / prev_prices

            # Portfolio return = weighted sum of asset returns - transaction costs
            portfolio_return = np.dot(weights, price_returns) - tran_cost

            # Update portfolio value
            portfolio_values[t] = portfolio_values[t - 1] * (1 + portfolio_return)

            prev_weights = weights.copy()

        # Store final weights
        weights_over_time[-1] = prev_weights

        # Create results DataFrame
        results_df = pd.DataFrame({
            "timestamp": self.close_prices["timestamp"],
            "portfolio_value": portfolio_values,
        })

        # Add weights for each symbol
        for i, symbol in enumerate(self.symbols):
            results_df[f"{symbol}_weight"] = weights_over_time[:, i]

        # Calculate returns and drawdown
        results_df["returns"] = results_df["portfolio_value"].pct_change()
        results_df["drawdown"] = PerformanceMetrics.drawdown_series(
            results_df["portfolio_value"]
        )

        # Calculate performance metrics
        total_return = PerformanceMetrics.total_return(results_df["portfolio_value"])
        n_years = (results_df["timestamp"].iloc[-1] - results_df["timestamp"].iloc[0]).days / 365.25
        cagr = PerformanceMetrics.cagr(results_df["portfolio_value"], n_years)
        annualized_vol = PerformanceMetrics.annualized_volatility(
            results_df["returns"].dropna()
        )
        sharpe = PerformanceMetrics.sharpe_ratio(
            results_df["returns"].dropna(), risk_free_rate=0.0
        )
        max_dd = results_df["drawdown"].min()
        avg_turnover = turnover_per_period.mean()
        avg_tran_costs = transaction_costs_per_period.mean()

        return {
            "results_df": results_df,
            "metrics": {
                "total_return": total_return,
                "cagr": cagr,
                "annualized_volatility": annualized_vol,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_dd,
                "average_turnover": avg_turnover,
                "average_transaction_cost": avg_tran_costs,
                "n_years": n_years,
                "final_value": portfolio_values[-1],
            },
            "turnover_per_period": turnover_per_period,
            "transaction_costs_per_period": transaction_costs_per_period,
        }


# =====================================================================
# Plotting
# =====================================================================


class Plotter:
    """Generate research plots."""

    @staticmethod
    def plot_equity_and_drawdown(
        results_df: pd.DataFrame, title: str = "Portfolio Equity & Drawdown"
    ):
        """Plot equity curve and drawdown."""
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 8), sharex=True
        )

        # Equity curve
        ax1.plot(
            results_df["timestamp"],
            results_df["portfolio_value"],
            linewidth=2,
            color="blue",
        )
        ax1.set_ylabel("Portfolio Value ($)", fontsize=11)
        ax1.set_title(title, fontsize=13, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        # Drawdown
        ax2.fill_between(
            results_df["timestamp"],
            results_df["drawdown"],
            0,
            alpha=0.5,
            color="red",
        )
        ax2.set_ylabel("Drawdown", fontsize=11)
        ax2.set_xlabel("Date", fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1%}"))

        # Format x-axis
        ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_weights(results_df: pd.DataFrame, symbols: list, title: str = "Portfolio Weights"):
        """Plot portfolio weights over time."""
        weight_cols = [c for c in results_df.columns if "_weight" in c]

        fig, ax = plt.subplots(figsize=(14, 6))

        for col in weight_cols:
            symbol = col.replace("_weight", "")
            ax.plot(
                results_df["timestamp"],
                results_df[col],
                label=symbol,
                linewidth=1.5,
                alpha=0.7,
            )

        ax.set_ylabel("Weight", fontsize=11)
        ax.set_xlabel("Date", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1%}"))

        # Format x-axis
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_returns_distribution(
        results_df: pd.DataFrame, title: str = "Returns Distribution"
    ):
        """Plot returns distribution."""
        returns = results_df["returns"].dropna()

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(returns, bins=50, alpha=0.7, color="blue", edgecolor="black")
        ax.axvline(returns.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {returns.mean():.4f}")
        ax.axvline(returns.median(), color="green", linestyle="--", linewidth=2, label=f"Median: {returns.median():.4f}")

        ax.set_xlabel("Return", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return fig


# =====================================================================
# Example: Dummy Strategy (Equal Weight)
# =====================================================================


def equal_weight_strategy(close_prices_df: pd.DataFrame, row_idx: int) -> pd.Series:
    """
    Example: Equal weight across all symbols.
    This strategy has no look-ahead bias: it only looks at data up to row_idx.
    """
    symbols = [c for c in close_prices_df.columns if c != "timestamp"]
    n_symbols = len(symbols)
    weights = pd.Series(1.0 / n_symbols, index=symbols)
    return weights


# =====================================================================
# Main: Run a demonstration
# =====================================================================


def main():
    print("=" * 70)
    print("RESEARCH BACKTESTER FRAMEWORK")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    loader = DataLoader(Path("data"))
    symbol_data = loader.load_all_symbols()

    if not symbol_data:
        print("ERROR: No symbol data loaded. Exiting.")
        return

    # Align symbols
    print(f"\n[2] Aligning {len(symbol_data)} symbols to common timeline...")
    aligner = DataAligner(symbol_data)
    close_prices = aligner.get_close_prices()
    print(f"  Common timeline: {len(close_prices):,} timestamps")
    print(f"  Date range: {close_prices['timestamp'].iloc[0]} → {close_prices['timestamp'].iloc[-1]}")

    # Descriptive statistics
    print(f"\n[3] Descriptive statistics:")
    for symbol in sorted(symbol_data.keys()):
        df = symbol_data[symbol]
        stats_dict = PerformanceMetrics.descriptive_stats(df, symbol)
        print(f"\n  {symbol}:")
        print(f"    Observations: {stats_dict['observations']:,}")
        print(f"    Date Range: {stats_dict['start_date']} → {stats_dict['end_date']}")
        print(f"    Annualized Vol: {stats_dict['annualized_vol']:.2%}")
        print(f"    Missing Data: {stats_dict['missing_pct']:.2f}%")
        if "avg_daily_volume" in stats_dict:
            print(f"    Avg Volume: {stats_dict['avg_daily_volume']:.2f}")

    # Run backtest with equal-weight strategy
    print(f"\n[4] Running backtest (Equal Weight strategy)...")
    backtester = Backtester(close_prices, initial_capital=100_000)
    results = backtester.run(equal_weight_strategy)

    # Print metrics
    print("\n[5] Performance Metrics:")
    metrics = results["metrics"]
    print(f"  Total Return: {metrics['total_return']:.2%}")
    print(f"  CAGR: {metrics['cagr']:.2%}")
    print(f"  Annualized Volatility: {metrics['annualized_volatility']:.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"  Average Turnover: {metrics['average_turnover']:.2%}")
    print(f"  Average Transaction Cost: {metrics['average_transaction_cost']:.4%}")
    print(f"  Final Value: ${metrics['final_value']:,.2f}")

    # Save results
    print(f"\n[6] Saving results...")
    results_df = results["results_df"]
    output_path = Path("research/backtest_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"  Results saved to {output_path}")

    # Create plots
    print(f"\n[7] Creating plots...")
    
    fig1 = Plotter.plot_equity_and_drawdown(results_df, title="Equal Weight Strategy - Equity & Drawdown")
    fig1.savefig("research/equity_drawdown.png", dpi=150, bbox_inches="tight")
    print(f"  Plot saved to research/equity_drawdown.png")

    fig2 = Plotter.plot_weights(results_df, backtester.symbols, title="Equal Weight Strategy - Portfolio Weights")
    fig2.savefig("research/weights.png", dpi=150, bbox_inches="tight")
    print(f"  Plot saved to research/weights.png")

    fig3 = Plotter.plot_returns_distribution(results_df, title="Equal Weight Strategy - Returns Distribution")
    fig3.savefig("research/returns_dist.png", dpi=150, bbox_inches="tight")
    print(f"  Plot saved to research/returns_dist.png")

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
