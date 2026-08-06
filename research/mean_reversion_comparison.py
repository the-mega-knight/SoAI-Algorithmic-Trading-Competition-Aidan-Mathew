"""
Compare mean reversion variants on the historical crypto dataset.

This script uses the existing research framework for resampling, alignment, and
backtesting, while reading the 1-minute historical data from data/historical/.
"""

from pathlib import Path
import importlib.util
import pandas as pd
import numpy as np

# Load research framework
_rb_path = Path(__file__).resolve().parents[0] / "research_backtest.py"
_spec = importlib.util.spec_from_file_location("research_backtest", str(_rb_path))
_rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rb)

Resampler = _rb.Resampler
DataAligner = _rb.DataAligner
Backtester = _rb.Backtester
PerformanceMetrics = _rb.PerformanceMetrics

# Load existing baseline strategies
_es_path = Path(__file__).resolve().parents[0] / "example_strategies.py"
_spec2 = importlib.util.spec_from_file_location("example_strategies", str(_es_path))
_es = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_es)

BUY_AND_HOLD = _es.equal_weight_strategy

RESULTS_PATH = Path(__file__).resolve().parents[0] / "mean_reversion_results.csv"


def _load_historical_symbol(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns in {path.name}: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
    if df["timestamp"].isnull().any():
        raise ValueError(f"Invalid timestamps in {path.name}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _build_daily_close_prices(data_dir: Path) -> pd.DataFrame:
    symbol_data = {}
    for csv_file in sorted(data_dir.glob("*_1m_spot.csv")):
        if "EXAMPLE" in csv_file.name:
            continue
        parts = csv_file.stem.split("_")
        if len(parts) < 2:
            continue
        symbol = f"{parts[0]}/{parts[1]}"
        symbol_data[symbol] = _load_historical_symbol(csv_file)

    if not symbol_data:
        raise RuntimeError(f"No historical CSV files found in {data_dir}")

    print("\nLoaded symbols and row counts:")
    for symbol, df in symbol_data.items():
        print(f"  {symbol}: {len(df):,} rows")

    resampled = {}
    for symbol, df in symbol_data.items():
        daily = Resampler.resample_ohlcv(df, symbol, "1D")
        resampled[symbol] = daily
        if daily["timestamp"].duplicated().any():
            raise RuntimeError(f"Duplicate timestamps after resampling for {symbol}")

    aligner = DataAligner(resampled)
    close_prices_daily = aligner.get_close_prices()
    if close_prices_daily["timestamp"].duplicated().any():
        raise RuntimeError("Duplicate timestamps found after aligning daily data")

    return close_prices_daily


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _run_backtest(strategy_fn, close_prices_daily: pd.DataFrame) -> dict:
    bt = Backtester(close_prices_daily, transaction_cost_bps=10)
    return bt.run(strategy_fn)


def _collect_metrics(res: dict) -> dict:
    results = res["results_df"].copy()
    returns = results["returns"].dropna()
    return {
        "strategy": res.get("strategy", "unknown"),
        "total_return": res["metrics"]["total_return"],
        "cagr": res["metrics"]["cagr"],
        "sharpe_ratio": res["metrics"]["sharpe_ratio"],
        "annualized_volatility": res["metrics"]["annualized_volatility"],
        "max_drawdown": res["metrics"]["max_drawdown"],
        "average_turnover": res["metrics"]["average_turnover"],
        "final_portfolio_value": res["metrics"]["final_value"],
        "positive_days": int((returns > 0).sum()),
        "negative_days": int((returns < 0).sum()),
    }


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _momentum_3d_returns(close_prices_df: pd.DataFrame) -> pd.Series:
    symbols = [c for c in close_prices_df.columns if c != "timestamp"]
    if len(close_prices_df) < 4:
        return pd.Series({s: np.nan for s in symbols})
    closes = close_prices_df[symbols]
    return closes.iloc[-1] / closes.iloc[-4] - 1


def baseline_mean_reversion(close_prices_df: pd.DataFrame, current_row_idx: int) -> pd.Series:
    symbols = [c for c in close_prices_df.columns if c != "timestamp"]
    if len(close_prices_df) < 4:
        return pd.Series(0.0, index=symbols)
    momentum = close_prices_df[symbols].iloc[-1] / close_prices_df[symbols].iloc[-4] - 1
    worst = momentum.nsmallest(2).dropna()
    weights = pd.Series(0.0, index=symbols)
    if len(worst) == 0:
        return weights
    if len(worst) == 1:
        weights[worst.index[0]] = 1.0
    else:
        weights[worst.index] = 0.5
    return weights


def oversold_filter(close_prices_df: pd.DataFrame, current_row_idx: int) -> pd.Series:
    symbols = [c for c in close_prices_df.columns if c != "timestamp"]
    if len(close_prices_df) < 4:
        return pd.Series(0.0, index=symbols)
    momentum = close_prices_df[symbols].iloc[-1] / close_prices_df[symbols].iloc[-4] - 1
    qualified = momentum[momentum < -0.05].sort_values()
    weights = pd.Series(0.0, index=symbols)
    if len(qualified) == 0:
        return weights
    if len(qualified) == 1:
        weights[qualified.index[0]] = 0.5
        return weights
    weights[qualified.index[:2]] = 0.5
    return weights


def volatility_filter(close_prices_df: pd.DataFrame, current_row_idx: int) -> pd.Series:
    symbols = [c for c in close_prices_df.columns if c != "timestamp"]
    if len(close_prices_df) < 9:
        return pd.Series(0.0, index=symbols)

    closes = close_prices_df[symbols]
    momentum = closes.iloc[-1] / closes.iloc[-4] - 1
    returns = closes.pct_change()
    vol5 = returns.rolling(5, min_periods=5).std().iloc[-1]
    median_vol = vol5.median()
    eligible = vol5[vol5 < median_vol].dropna().index.tolist()
    if len(eligible) == 0:
        return pd.Series(0.0, index=symbols)

    momentum = momentum[eligible].dropna().sort_values()
    worst = momentum.nsmallest(2)
    weights = pd.Series(0.0, index=symbols)
    if len(worst) == 0:
        return weights
    if len(worst) == 1:
        weights[worst.index[0]] = 0.5
    else:
        weights[worst.index] = 0.5
    return weights


def btc_regime_filter(close_prices_df: pd.DataFrame, current_row_idx: int) -> pd.Series:
    symbols = [c for c in close_prices_df.columns if c != "timestamp"]
    if len(close_prices_df) < 11:
        return pd.Series(0.0, index=symbols)
    closes = close_prices_df[symbols]
    btc = "BTC/USD"
    if btc not in closes.columns:
        return pd.Series(0.0, index=symbols)
    btc_close = closes[btc].iloc[-1]
    btc_sma10 = _sma(closes[btc], 10).iloc[-1]
    if pd.isna(btc_sma10) or btc_close <= btc_sma10:
        return pd.Series(0.0, index=symbols)
    return baseline_mean_reversion(close_prices_df, current_row_idx)


def rsi_filter(close_prices_df: pd.DataFrame, current_row_idx: int) -> pd.Series:
    symbols = [c for c in close_prices_df.columns if c != "timestamp"]
    if len(close_prices_df) < 18:
        return pd.Series(0.0, index=symbols)
    closes = close_prices_df[symbols]
    momentum = closes.iloc[-1] / closes.iloc[-4] - 1
    rsi_values = closes.apply(lambda col: _rsi(col, 14)).iloc[-1]
    qualified = rsi_values[rsi_values < 35].dropna().index.tolist()
    if len(qualified) == 0:
        return pd.Series(0.0, index=symbols)
    momentum = momentum[qualified].dropna().sort_values()
    worst = momentum.nsmallest(2)
    weights = pd.Series(0.0, index=symbols)
    if len(worst) == 0:
        return weights
    if len(worst) == 1:
        weights[worst.index[0]] = 0.5
    else:
        weights[worst.index] = 0.5
    return weights


def _run_strategy(name: str, fn, close_prices_daily: pd.DataFrame) -> dict:
    res = _run_backtest(fn, close_prices_daily)
    metrics = _collect_metrics(res)
    metrics["strategy"] = name
    return metrics


def main() -> None:
    data_dir = Path(__file__).resolve().parents[0].parent / "data" / "historical"
    print(f"Loading historical data from: {data_dir}")
    close_prices_daily = _build_daily_close_prices(data_dir)
    first = close_prices_daily["timestamp"].iloc[0].date()
    last = close_prices_daily["timestamp"].iloc[-1].date()
    print(f"Detected historical date range: {first} -> {last}")
    print(f"Daily observations: {len(close_prices_daily)}")
    print(f"Assets: {[c for c in close_prices_daily.columns if c != 'timestamp']}\n")

    strategies = [
        ("buy_and_hold", BUY_AND_HOLD),
        ("baseline_mean_reversion", baseline_mean_reversion),
        ("oversold_filter", oversold_filter),
        ("volatility_filter", volatility_filter),
        ("btc_regime_filter", btc_regime_filter),
        ("rsi_filter", rsi_filter),
    ]

    records = []
    for name, fn in strategies:
        print(f"Running strategy: {name}")
        metrics = _run_strategy(name, fn, close_prices_daily)
        records.append(metrics)

    summary_df = pd.DataFrame(records)
    summary_df = summary_df[
        [
            "strategy",
            "total_return",
            "cagr",
            "sharpe_ratio",
            "annualized_volatility",
            "max_drawdown",
            "average_turnover",
            "final_portfolio_value",
            "positive_days",
            "negative_days",
        ]
    ]
    summary_df = summary_df.sort_values(by="total_return", ascending=False).reset_index(drop=True)
    summary_df.to_csv(RESULTS_PATH, index=False)

    print("\nStrategy ranking by total return:")
    print(summary_df.to_string(index=False))

    best_return = summary_df.iloc[0]["strategy"]
    best_sharpe = summary_df.loc[summary_df["sharpe_ratio"].idxmax()]["strategy"]
    best_dd = summary_df.loc[summary_df["max_drawdown"].idxmin()]["strategy"]
    buyhold_ret = summary_df.loc[summary_df["strategy"] == "buy_and_hold", "total_return"].iloc[0]
    beat_bh = summary_df[summary_df["total_return"] > buyhold_ret]
    beat_bh_names = beat_bh["strategy"].tolist()
    profitable = summary_df[summary_df["total_return"] > 0]["strategy"].tolist()

    print(f"\nBest total return: {best_return}")
    print(f"Best Sharpe: {best_sharpe}")
    print(f"Lowest drawdown: {best_dd}")
    print(f"Strategies beating Buy & Hold: {beat_bh_names if beat_bh_names else 'None'}")
    print(f"Profitable strategies after costs: {profitable if profitable else 'None'}")


if __name__ == "__main__":
    main()
