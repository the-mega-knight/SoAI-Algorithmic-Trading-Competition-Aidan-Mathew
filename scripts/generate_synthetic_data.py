"""
Synthetic 1-minute OHLCV generator for LOCAL DEV/backtesting only.
Vectorized regime-switching returns + an Ornstein-Uhlenbeck pull back
toward the start price (half-life ~15 days) so the walk stays in a
plausible crypto-volatility range instead of drifting to absurd levels.
Not used for official scoring -- purely for exercising strategy code.
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(7)
OUT = Path("data")
OUT.mkdir(parents=True, exist_ok=True)

MINUTES_PER_DAY = 1440
DAYS = 60
N = MINUTES_PER_DAY * DAYS

CONFIGS = [
    ("BTC", 62000.0, 0.55),
    ("ETH", 3400.0, 0.70),
    ("SOL", 145.0, 0.95),
]

def build_regime_mask(n, rng):
    kinds = np.array(["trend_up", "trend_down", "chop", "high_vol"])
    weights = [0.28, 0.22, 0.35, 0.15]
    labels = np.empty(n, dtype=object)
    i = 0
    while i < n:
        length = int(rng.integers(1000, 4000))
        length = min(length, n - i)
        kind = rng.choice(kinds, p=weights)
        labels[i:i + length] = kind
        i += length
    return labels

def simulate(start_price, base_vol, n, rng):
    minute_vol = base_vol / np.sqrt(252 * MINUTES_PER_DAY)
    labels = build_regime_mask(n, rng)

    drift_map = {"trend_up": 0.00006, "trend_down": -0.00006, "chop": 0.0, "high_vol": 0.0}
    vol_map = {"trend_up": 1.0, "trend_down": 1.1, "chop": 0.55, "high_vol": 2.0}
    mu = np.vectorize(drift_map.get)(labels).astype(float)
    sig_mult = np.vectorize(vol_map.get)(labels).astype(float)

    shocks = rng.normal(0.0, 1.0, size=n) * (minute_vol * sig_mult)

    # OU mean reversion on log-price toward log(start_price), half-life ~15 days
    half_life_minutes = 15 * MINUTES_PER_DAY
    theta = np.log(2) / half_life_minutes
    log_price0 = np.log(start_price)

    log_price = np.empty(n)
    x = log_price0
    steps = mu + shocks  # base random-walk step before mean reversion
    # Vectorized OU via cumulative trick is not exact with regime drift, so
    # do a fast numpy loop (still ~86k iters, trivial compared to a Python
    # object-heavy loop since it's pure float ops).
    for t in range(n):
        x += theta * (log_price0 - x) + steps[t]
        log_price[t] = x

    close = np.exp(log_price)
    open_ = np.roll(close, 1)
    open_[0] = start_price
    intrabar_noise = np.abs(rng.normal(0, minute_vol * 0.5, size=n))
    high = np.maximum(open_, close) * (1 + intrabar_noise)
    low = np.minimum(open_, close) * (1 - intrabar_noise)
    volume = rng.lognormal(mean=np.log(50), sigma=0.8, size=n) * (start_price / 100)

    return open_, high, low, close, volume

start = pd.Timestamp("2026-06-06T00:00:00Z")
timestamps = pd.date_range(start, periods=N, freq="1min", tz="UTC")

for name, price0, vol in CONFIGS:
    o, h, l, c, v = simulate(price0, vol, N, rng)
    df = pd.DataFrame({
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "timestamp": timestamps,
    })
    out_path = OUT / f"{name}_1m_spot.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {len(df)} rows, {df['timestamp'].min()} -> {df['timestamp'].max()}, "
          f"price {df['close'].iloc[0]:.2f} -> {df['close'].iloc[-1]:.2f}, "
          f"min {df['close'].min():.2f} max {df['close'].max():.2f}")

print("DONE")
