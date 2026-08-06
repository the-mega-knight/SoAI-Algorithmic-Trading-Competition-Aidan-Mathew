"""
Compare the current 3-day momentum strategy and new variants on the historical crypto dataset.

This script uses the existing research framework for resampling, alignment, and backtesting.
"""

from pathlib import Path
import importlib.util
import pandas as pd

# Load research framework
_rb_path = Path(__file__).resolve().parents[0] / "research_backtest.py"
_spec = importlib.util.spec_from_file_location("research_backtest", str(_rb_path))
_rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rb)

Resampler = _rb.Resampler
DataAligner = _rb.DataAligner
Backtester = _rb.Backtester
PerformanceMetrics = _rb.PerformanceMetrics

# Load existing strategy definitions
_sc_path = Path(__file__).resolve().parents[0] / "strategy_comparison.py"
_spec2 = importlib.util.spec_from_file_location("strategy_comparison", str(_sc_path))
_sc = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_sc)

# Load buy-and-hold baseline
_es_path = Path(__file__).resolve().parents[0] / "example_strategies.py"
_spec3 = importlib.util.spec_from_file_location("example_strategies", str(_es_path))
_es = importlib.util.module_from_spec(_spec3)
_spec3.loader.exec_module(_es)

THREE_DAY_STRATEGY = _sc.three_day_momentum_strategy
BUY_AND_HOLD = _es.equal_weight_strategy

SUMMARY_PATH = Path(__file__).resolve().parents[0] / "strategy_variant_comparison.csv"
EQUITY_PATH = Path(__file__).resolve().parents[0] / "strategy_variant_equity.csv"


def _load_historical_csv(path: Path) -> pd.DataFrame:
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
    print(f"Assets: {[c for c in close_prices_daily.columns if c != 'timestamp']}\n")
    dup_count = close_prices_daily["timestamp"].duplicated().sum()
    print(f"Duplicate daily timestamps: {dup_count}")
    if dup_count > 0:
        raise RuntimeError("Duplicate timestamps found in aligned daily dataset")


def _run_backtest(strategy_fn, close_prices_daily: pd.DataFrame) -> dict:
    bt = Backtester(close_prices_daily, transaction_cost_bps=10)
    return bt.run(strategy_fn)


def _collect_metrics(res: dict) -> dict:
    results = res["results_df"].copy()
    returns = results["returns"].dropna()
    return {
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


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def _volatility(series: pd.Series, window: int) -> pd.Series:
    return series.pct_change().rolling(window, min_periods=window).std()


def _top_n_positive(momentum, n: int) -> list[str]:
    positive = momentum[momentum > 0].dropna()
    if len(positive) == 0:
        return []
    return positive.nlargest(n).index.tolist()


def trend_filter_strategy(close_prices_df: pd.DataFrame, current_row_idx: int) -> pd.Series:
    symbols = [c for c in close_prices_df.columns if c != "timestamp"]
    if len(close_prices_df) < 20:
        return pd.Series(0.0, index=symbols)

    closes = close_prices_df[symbols]
    prev = closes.iloc[-1]
    prev4 = closes.iloc[-4]
    momentum = prev / prev4 - 1

    sma20 = _sma(closes, 20).iloc[-1]
    eligible = [s for s in symbols if not pd.isna(sma20[s]) and prev[s] > sma20[s]]
    if len(eligible) == 0:
        return pd.Series(0.0, index=symbols)

    momentum = momentum[eligible].dropna()
    if len(momentum) == 0:
        return pd.Series(0.0, index=symbols)

    best = momentum.nlargest(2)
    weights = pd.Series(0.0, index=symbols)
    if len(best) == 0:
        return weights
    if len(best) == 1:
        weights[best.index[0]] = 1.0
    else:
        weights[best.index] = 1.0 / len(best)
    return weights


def top2_momentum_strategy(close_prices_df: pd.DataFrame, current_row_idx: int) -> pd.Series:
    symbols = [c for c in close_prices_df.columns if c != "timestamp"]
    if len(close_prices_df) < 4:
        return pd.Series(0.0, index=symbols)

    closes = close_prices_df[symbols]
    prev = closes.iloc[-1]
    prev4 = closes.iloc[-4]
    momentum = prev / prev4 - 1
    best = _top_n_positive(momentum, 2)
    weights = pd.Series(0.0, index=symbols)
    if len(best) == 1:
        weights[best[0]] = 1.0
    elif len(best) > 1:
        weights[best] = 1.0 / len(best)
    return weights


def top3_momentum_strategy(close_prices_df: pd.DataFrame, current_row_idx: int) -> pd.Series:
    symbols = [c for c in close_prices_df.columns if c != "timestamp"]
    if len(close_prices_df) < 4:
        return pd.Series(0.0, index=symbols)

    closes = close_prices_df[symbols]
    prev = closes.iloc[-1]
    prev4 = closes.iloc[-4]
    momentum = prev / prev4 - 1
    best = _top_n_positive(momentum, 3)
    weights = pd.Series(0.0, index=symbols)
    if len(best) == 1:
        weights[best[0]] = 1.0
    elif len(best) > 1:
        weights[best] = 1.0 / len(best)
    return weights


def volatility_scaling_strategy(close_prices_df: pd.DataFrame, current_row_idx: int) -> pd.Series:
    symbols = [c for c in close_prices_df.columns if c != "timestamp"]
    if len(close_prices_df) < 24:
        return pd.Series(0.0, index=symbols)

    closes = close_prices_df[symbols]
    prev = closes.iloc[-1]
    prev4 = closes.iloc[-4]
    momentum = prev / prev4 - 1
    best = momentum.nlargest(2).dropna()
    if len(best) == 0:
        return pd.Series(0.0, index=symbols)

    returns = closes.pct_change()
    vol20 = returns.rolling(20, min_periods=20).std().iloc[-1]

    eligible = [s for s in best.index if not pd.isna(vol20[s]) and vol20[s] > 0]
    if len(eligible) == 0:
        weights = pd.Series(0.0, index=symbols)
        weights[best.index] = 1.0 / len(best)
        return weights

    inv_vol = pd.Series({s: 1.0 / vol20[s] for s in eligible})
    normalized = inv_vol / inv_vol.sum()
    weights = pd.Series(0.0, index=symbols)
    for s in eligible:
        weights[s] = normalized[s]
    return weights


def btc_market_filter_strategy(close_prices_df: pd.DataFrame, current_row_idx: int) -> pd.Series:
    symbols = [c for c in close_prices_df.columns if c != "timestamp"]
    if len(close_prices_df) < 21:
        return pd.Series(0.0, index=symbols)

    closes = close_prices_df[symbols]
    btc_symbol = "BTC/USD"
    if btc_symbol not in closes.columns:
        return pd.Series(0.0, index=symbols)

    btc_prev = closes[btc_symbol].iloc[-1]
    btc_past = closes[btc_symbol].iloc[-21]
    btc_return = btc_prev / btc_past - 1
    if btc_return <= 0:
        return pd.Series(0.0, index=symbols)

    return THREE_DAY_STRATEGY(close_prices_df, current_row_idx)


def _run_and_record(name: str, strategy_fn, close_prices_daily: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    res = _run_backtest(strategy_fn, close_prices_daily)
    metrics = _collect_metrics(res)
    metrics["strategy"] = name
    equity_df = pd.DataFrame({
        "timestamp": res["results_df"]["timestamp"],
        f"{name}_equity": res["results_df"]["portfolio_value"],
    })
    return metrics, equity_df


def main() -> None:
    data_dir = Path(__file__).resolve().parents[0].parent / "data" / "historical"
    print(f"Loading historical data from: {data_dir}")
    close_prices_daily = _build_daily_close_prices(data_dir)
    _print_data_checks(close_prices_daily)

    strategies = [
        ("buy_and_hold", BUY_AND_HOLD),
        ("momentum_3d", THREE_DAY_STRATEGY),
        ("trend_filter", trend_filter_strategy),
        ("top2_momentum", top2_momentum_strategy),
        ("top3_momentum", top3_momentum_strategy),
        ("volatility_scaling", volatility_scaling_strategy),
        ("btc_market_filter", btc_market_filter_strategy),
    ]

    summary_rows = []
    equity_frames = []

    for name, fn in strategies:
        print(f"\nRunning strategy: {name}")
        metrics, equity_df = _run_and_record(name, fn, close_prices_daily)
        summary_rows.append(metrics)
        equity_frames.append(equity_df)

    summary_df = pd.DataFrame(summary_rows)
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

    merged_eq = equity_frames[0]
    for df in equity_frames[1:]:
        merged_eq = pd.merge(merged_eq, df, on="timestamp", how="outer")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    merged_eq.to_csv(EQUITY_PATH, index=False)

    print("\nStrategy ranking by total return:")
    print(summary_df.to_string(index=False))

    best_return = summary_df.iloc[0]["strategy"]
    best_sharpe = summary_df.loc[summary_df["sharpe_ratio"].idxmax()]["strategy"]
    best_dd = summary_df.loc[summary_df["max_drawdown"].idxmin()]["strategy"]

    buyhold_return = summary_df.loc[summary_df["strategy"] == "buy_and_hold", "total_return"].iloc[0]
    best_strategy_return = summary_df.iloc[0]["total_return"]
    diff = best_strategy_return - buyhold_return
    outperform = diff > 0

    print(f"\nBest strategy by return: {best_return}")
    print(f"Best strategy by Sharpe: {best_sharpe}")
    print(f"Lowest drawdown: {best_dd}")
    print(f"Did any strategy outperform Buy & Hold? {'Yes' if outperform else 'No'}")
    if outperform:
        print(f"Outperformance vs Buy & Hold: {_format_pct(diff)}")

    print(f"\nSaved summary to: {SUMMARY_PATH}")
    print(f"Saved equity curves to: {EQUITY_PATH}")


if __name__ == "__main__":
    main()
