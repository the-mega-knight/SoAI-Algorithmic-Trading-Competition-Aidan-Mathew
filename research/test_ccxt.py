import ccxt


def main():
    exchange = ccxt.coinbase()

    print(f"Exchange: {exchange.name}")

    # Load the exchange's market definitions.
    exchange.load_markets()

    symbol = "BTC/USD"

    print(f"Checking: {symbol}")

    # Fetch the most recent 5 one-minute candles.
    candles = exchange.fetch_ohlcv(
        symbol,
        timeframe="1m",
        limit=5,
    )

    print("\nLatest candles:")
    print("timestamp            open       high       low        close      volume")

    for candle in candles:
        timestamp, open_, high, low, close, volume = candle

        print(
            f"{timestamp}  "
            f"{open_:9.2f}  "
            f"{high:9.2f}  "
            f"{low:9.2f}  "
            f"{close:9.2f}  "
            f"{volume:.6f}"
        )


if __name__ == "__main__":
    main()
