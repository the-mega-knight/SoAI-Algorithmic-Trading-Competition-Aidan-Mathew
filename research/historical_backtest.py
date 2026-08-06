"""
Historical backtest using the downloaded 8-asset crypto minute dataset.

This script reuses the existing research framework in research/research_backtest.py
and the existing 3-day momentum strategy from research/strategy_comparison.py.

It reads CSVs from data/historical/, resamples to daily OHLCV, aligns the assets,
then runs the 3-day momentum strategy and a buy-and-hold equal-weight benchmark
over the same daily date range.
"""

from pathlib import Path
import importlib.util
import pandas as pd

# Load the research framework module
_rb_path = Path(__file__).resolve().parents[0] / "research_backtest.py"
_spec = importlib.util.spec_from_file_location("research_backtest", str(_rb_path))
_rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rb)

DataLoader = _rb.DataLoader
Resampler = _rb.Resampler
DataAligner = _rb.DataAligner
Backtester = _rb.Backtester
PerformanceMetrics = _rb.PerformanceMetrics

# Load the existing research strategy definitions
_sc_path = Path(__file__).resolve().parents[0] / "strategy_comparison.py"
_spec2 = importlib.util.spec_from_file_location("strategy_comparison", str(_sc_path))
_sc = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_sc)

# Load the buy-and-hold baseline strategy from example_strategies
_es_path = Path(__file__).resolve().parents[0] / "example_strategies.py"
_spec3 = importlib.util.spec_from_file_location("example_strategies", str(_es_path))
_es = importlib.util.module_from_spec(_spec3)
_spec3.loader.exec_module(_es)

THREE_DAY_STRATEGY = _sc.three_day_momentum_strategy
BUY_AND_HOLD_STRATEGY = _es.equal_weight_strategy

SUMMARY_PATH = Path(__file__).resolve().parents[0] / "historical_backtest_summary.csv"
EQUITY_PATH = Path(__file__).resolve().parents[0] / "historical_equity_curves.csv"


def _load_historical_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns in {path.name}: {missing}")

    # Timestamps are stored in epoch milliseconds.
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
        df = _load_historical_csv(csv_file)
        symbol_data[symbol] = df

    if len(symbol_data) == 0:
        raise RuntimeError(f"No CSV files loaded from {data_dir}")

    print("\nLoaded symbols and row counts:")
    for symbol, df in symbol_data.items():
        dup_count = df["timestamp"].duplicated().sum()
        print(f"  {symbol}: {len(df):,} rows", end="")
        if dup_count > 0:
            print(f" ({dup_count} duplicate timestamps detected)", end="")
        print()

    resampled = {}
    for symbol, df in symbol_data.items():
        daily = Resampler.resample_ohlcv(df, symbol, "1D")
        resampled[symbol] = daily
        dup_count = daily["timestamp"].duplicated().sum()
        if dup_count > 0:
            raise RuntimeError(f"Duplicate timestamps after resampling for {symbol}")

    aligner = DataAligner(resampled)
    close_prices_daily = aligner.get_close_prices()

    if close_prices_daily["timestamp"].duplicated().any():
        raise RuntimeError("Duplicate timestamps found in daily aligned data")

    return close_prices_daily


def _print_data_checks(close_prices_daily: pd.DataFrame) -> None:
    first = close_prices_daily["timestamp"].iloc[0].date()
    last = close_prices_daily["timestamp"].iloc[-1].date()
    print(f"\nDetected historical date range: {first} -> {last}")
    print(f"Daily observations: {len(close_prices_daily)}")
    print(f"Assets in daily dataset: {[c for c in close_prices_daily.columns if c != 'timestamp']}\n")
    dup_count = close_prices_daily["timestamp"].duplicated().sum()
    print(f"Duplicate daily timestamps: {dup_count}")
    if dup_count > 0:
        raise RuntimeError("Daily aligned data contains duplicate timestamps")


def _run_backtest(strategy_fn, close_prices_daily: pd.DataFrame) -> dict:
    bt = Backtester(close_prices_daily, transaction_cost_bps=10)
    return bt.run(strategy_fn)


def _collect_metrics(res: dict) -> dict:
    res_df = res["results_df"].copy()
    returns = res_df["returns"].dropna()
    pos_days = int((returns > 0).sum())
    neg_days = int((returns < 0).sum())
    metrics = res["metrics"].copy()
    metrics.update(
        {
            "positive_days": pos_days,
            "negative_days": neg_days,
        }
    )
    return metrics


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    data_dir = Path(__file__).resolve().parents[0].parent / "data" / "historical"
    print(f"Loading historical data from: {data_dir}")

    close_prices_daily = _build_daily_close_prices(data_dir)
    _print_data_checks(close_prices_daily)

    date_range = (
        close_prices_daily["timestamp"].iloc[0].date(),
        close_prices_daily["timestamp"].iloc[-1].date(),
    )
    n_obs = len(close_prices_daily)

    print("\nRunning backtests with 10 bps transaction costs and default initial capital")
    momentum_res = _run_backtest(THREE_DAY_STRATEGY, close_prices_daily)
    buyhold_res = _run_backtest(BUY_AND_HOLD_STRATEGY, close_prices_daily)

    momentum_metrics = _collect_metrics(momentum_res)
    buyhold_metrics = _collect_metrics(buyhold_res)

    momentum_metrics["strategy"] = "momentum_3d"
    buyhold_metrics["strategy"] = "buy_and_hold"

    summary_df = pd.DataFrame([momentum_metrics, buyhold_metrics])
    summary_df = summary_df[
        [
            "strategy",
            "total_return",
            "cagr",
            "sharpe_ratio",
            "annualized_volatility",
            "max_drawdown",
            "average_turnover",
            "final_value",
            "positive_days",
            "negative_days",
        ]
    ]

    eq_df = pd.DataFrame({
        "timestamp": momentum_res["results_df"]["timestamp"],
        "momentum_3d_equity": momentum_res["results_df"]["portfolio_value"],
        "buy_and_hold_equity": buyhold_res["results_df"]["portfolio_value"],
    })

    # Save outputs
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    eq_df.to_csv(EQUITY_PATH, index=False)

    print("\nBacktest summary")
    print(f"Date range: {date_range[0]} -> {date_range[1]}")
    print(f"Daily observations: {n_obs}")
    print("\nMomentum 3-day metrics:")
    print(f"  Total return: {_format_pct(momentum_metrics['total_return'])}")
    print(f"  CAGR: {_format_pct(momentum_metrics['cagr'])}")
    print(f"  Sharpe ratio: {momentum_metrics['sharpe_ratio']:.4f}")
    print(f"  Annualized volatility: {_format_pct(momentum_metrics['annualized_volatility'])}")
    print(f"  Maximum drawdown: {_format_pct(momentum_metrics['max_drawdown'])}")
    print(f"  Average turnover: {_format_pct(momentum_metrics['average_turnover'])}")
    print(f"  Final portfolio value: ${momentum_metrics['final_value']:,.2f}")
    print(f"  Positive-return days: {momentum_metrics['positive_days']}")
    print(f"  Negative-return days: {momentum_metrics['negative_days']}")

    print("\nBuy & Hold metrics:")
    print(f"  Total return: {_format_pct(buyhold_metrics['total_return'])}")
    print(f"  CAGR: {_format_pct(buyhold_metrics['cagr'])}")
    print(f"  Sharpe ratio: {buyhold_metrics['sharpe_ratio']:.4f}")
    print(f"  Annualized volatility: {_format_pct(buyhold_metrics['annualized_volatility'])}")
    print(f"  Maximum drawdown: {_format_pct(buyhold_metrics['max_drawdown'])}")
    print(f"  Average turnover: {_format_pct(buyhold_metrics['average_turnover'])}")
    print(f"  Final portfolio value: ${buyhold_metrics['final_value']:,.2f}")
    print(f"  Positive-return days: {buyhold_metrics['positive_days']}")
    print(f"  Negative-return days: {buyhold_metrics['negative_days']}")

    diff = momentum_metrics["total_return"] - buyhold_metrics["total_return"]
    print("\nComparison:")
    print(f"  Strategy return: {_format_pct(momentum_metrics['total_return'])}")
    print(f"  Buy & Hold return: {_format_pct(buyhold_metrics['total_return'])}")
    print(f"  Difference: {_format_pct(diff)}")
    print(f"  Did momentum outperform? {'Yes' if diff > 0 else 'No'}")

    print(f"\nSaved summary to: {SUMMARY_PATH}")
    print(f"Saved daily equity curves to: {EQUITY_PATH}")


if __name__ == "__main__":
    main()
