import yfinance as yf
import os

os.makedirs("data", exist_ok=True)

symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

for symbol in symbols:
    # 10y instead of 2y: daily bars don't have the short intraday-history
    # limits minute bars do, so pulling more history is free and lets us
    # test the strategy across multiple market regimes (see regime_test.py).
    df = yf.download(symbol, period="10y", interval="1d", auto_adjust=True, progress=False)
    df = df.reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.rename(columns={
        "Date": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    out_path = f"data/{symbol}_daily.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
