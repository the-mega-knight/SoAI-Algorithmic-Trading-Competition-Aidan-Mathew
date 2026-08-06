# SoAI 2026 AI Algorithmic Trading Competition — Official Template

**Build. Deploy. Trade. Compete.**

This repository is the **official starter template** for the
[SoAI 2026 AI Algorithmic Trading Competition](https://www.soc-ai.org/events/intelligencex-2026),
held alongside **IntelligenceX 2026 — The Global Quantum × AI Frontier**
(24–26 September 2026, Singapore).

Fork or clone this repo, implement your strategy in
[`strategies/strategy.py`](strategies/strategy.py), backtest it locally with
[`backtest.py`](backtest.py), and submit your GitHub repository link before the
deadline. The official trading run is executed by the IntelligenceX technical
team in a standardized environment, so every submission is evaluated on a
level playing field.

---

## 1) Competition at a Glance

| Item | Value |
| --- | --- |
| Code submission deadline | **25 July 2026, 23:59:59 SGT (UTC+8)** |
| Verification & test run | 26–28 July 2026 (SGT) |
| Official trading period | 1 August 2026, 00:00:00 → 31 August 2026, 23:59:59 (SGT) |
| Winners announcement | 6 September 2026 (SGT) |
| Primary evaluation metric | **Terminal Return** (final portfolio return after full liquidation at the end of the trading window) |
| Total prize pool | **SGD 3,000** |
| Eligibility | Open worldwide; individuals or teams of 2–5 |
| Contact | info@soc-ai.org |

> ⚠️ **Prize condition (strict):** cash prizes are awarded only to winners who
> present **in person** at IntelligenceX 2026 in Singapore. No remote
> disbursement or money transfer will be arranged.

For the full call for participation, see the conference page:
<https://www.soc-ai.org/events/intelligencex-2026>.

---

## 2) Repository Layout

```text
SoAI-2026-AI-Algorithmic-Trading-Competition/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── backtest.py                       # local backtest entrypoint (pandas / CSV)
├── strategies/
│   ├── strategy.py                   # YOUR strategy (official entrypoint)
│   ├── params.py                     # shared parameters for local backtests
│   ├── example_strategy_1.py         # reference: daily DCA into SPY
│   └── example_strategy_2.py         # reference: one-shot buy & hold
└── data/
    └── EXAMPLE_1m_spot.csv           # placeholder minute-bar CSV
```

Files the official environment relies on:

- **`strategies/strategy.py`** — must define a class named `Strategy` that
  subclasses `lumibot.strategies.Strategy`. This is what the organizers import
  and run.
- **`requirements.txt`** — must list every Python dependency your strategy
  needs. The organizers install from this file.
- **`README.md`** — keep a short, accurate description of your approach so
  reviewers can reproduce it.

Everything else (the `backtest.py` harness, the `data/` folder, the example
strategies) is provided for **your local development only** and is not used
by the official evaluation.

---

## 3) Setup

### Prerequisites

- Python 3.10 or newer
- Git
- Internet access for `pip install`

### Create and activate a virtual environment

#### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

#### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your strategy needs additional libraries (e.g. PyTorch, XGBoost, TA-Lib),
add them to `requirements.txt` with pinned or compatible versions so the
official environment can reproduce your build.

---

## 4) Write Your Strategy

1. Open [`strategies/strategy.py`](strategies/strategy.py). The class
   `Strategy` already inherits from
   [`lumibot.strategies.Strategy`](https://lumibot.lumiwealth.com/) and is the
   single entrypoint expected by the official environment.
2. Implement the two key lifecycle methods:
   - **`initialize(self)`** — runs once before trading begins. Set
     `self.sleeptime` to the cadence at which your strategy wakes up,
     declare the assets you can trade, store any model artifacts or
     hyperparameters, and configure risk limits.
   - **`on_trading_iteration(self)`** — runs every `sleeptime` step. Read
     current portfolio state, pull market data via
     `self.get_last_price(...)` / `self.get_historical_prices(...)`, compute
     signals or model predictions, translate them into target weights, and
     submit orders with `self.create_order(...)` + `self.submit_order(...)`.
3. Use [`strategies/example_strategy_1.py`](strategies/example_strategy_1.py)
   and [`strategies/example_strategy_2.py`](strategies/example_strategy_2.py)
   as concrete reference patterns.

> ⏱️ **Supported trading frequencies.** The official execution environment
> accepts **minute-, hourly-, and daily-level** strategies — for example
> `self.sleeptime = "1M"`, `"5M"`, `"15M"`, `"60M"`, or `"1D"`. **Sub-minute
> (tick / second-level) scheduling is NOT supported** and will be rejected
> during verification. Design your strategy around one of the allowed
> cadences.

Useful Lumibot documentation:

- Lifecycle methods: <https://lumibot.lumiwealth.com/lifecycle_methods.html>
- Strategy methods: <https://lumibot.lumiwealth.com/strategy_methods.html>
- Strategy properties (cash, portfolio value, sleeptime, …):
  <https://lumibot.lumiwealth.com/strategy_properties.html>
- Entities (Asset, Order, Position): <https://lumibot.lumiwealth.com/entities.html>

> The competition evaluates **Terminal Return** only. Risk management,
> drawdown control, and execution-cost awareness are still strongly
> encouraged — a strategy that blows up before liquidation will not finish
> on the podium.

---

## 5) Asset Universe & Data Sources

### 🌐 Unrestricted asset universe

We believe in the power of AI to find signal in the noise. For SoAI 2026
the trading universe is **completely open** — from blue chips to penny
stocks, from BTC to meme coins, if it has data, you can trade it.

A symbol is eligible if it satisfies **both** of the following:

1. It is reachable through one of the data adapters the official
   execution engine uses:
   - **CCXT** — any spot trading pair listed on a [CCXT](https://github.com/ccxt/ccxt)-supported
     exchange (Binance, OKX, Bybit, Coinbase, Kraken, …). This covers
     blue-chip coins (BTC, ETH, SOL, BNB), altcoins (LDO, OP, ARB, INJ),
     meme coins (DOGE, SHIB, PEPE, WIF), and stable-coin pairs.
   - **Massive** — the production US-equities feed used by the organizers.
     Covers the full US public market: large caps (AAPL, MSFT, NVDA,
     TSLA), every ETF (SPY, QQQ, SMH, ARKK, TQQQ, UVXY, …), Chinese ADRs
     (BABA, PDD, TSM, ASML), and small / micro-cap / penny stocks.
2. **You can source its 1-minute historical data** for your local
   backtest.

### 📡 Official data source providers (August 2026 run)

During the verification run (26–28 July 2026) and the official trading
window (1–31 August 2026, SGT), every submission receives bars from:

- **CCXT** for crypto spot pairs.
- **Massive** for US equities and ETFs.

Bars are minute-resolution OHLCV — see *§8 Evaluation & Fairness →
Official data feed* for the full feed contract.

### ⚠️ Caveats and known pitfalls

Trading the long tail of the asset universe is exciting — and
unforgiving. Plan for the following gotchas before submission:

1. **Volume-aware slippage on the official engine.** The IntelligenceX
   execution engine caps each submitted child order at a fraction of the
   bar's real historical minute-volume. **Orders larger than the
   available liquidity will not fill** — back-test P&L from a model that
   "buys" a million dollars of an illiquid penny stock will not
   materialize on the live run. Size your trades against realistic
   per-minute volume.
2. **Local backtest ≠ official slippage.** The bundled `backtest.py`
   uses Lumibot's flat fee / slippage primitives by default (configurable
   at the top of the file). The official engine layers stricter,
   volume-aware constraints on top — treat your local results as an
   optimistic upper bound.
3. **Free 1-minute history is scarce for US equities.** CCXT happily
   serves years of minute bars for crypto, but free US-equity feeds
   (e.g. Yahoo) typically expose only a few days of intraday data. If
   you trade equities locally, you may need paid coverage (Polygon,
   Alpaca premium, DataBento, etc.). The organizers handle data during
   the August run — you only need historical data for local development.
4. **Survivorship bias.** Penny stocks and small-cap altcoins delist or
   go to zero frequently. The historical datasets you pull in 2026 will
   be missing many tickers that were tradable earlier — train your
   models with that in mind. The official trading universe is whatever
   is *live and listed* during August 2026.

---

## 6) Local Backtesting

This template ships with a ready-to-run **Pandas / CSV** backtest. Lumibot
also supports several other backtest modes — pick whichever fits the data
you already have access to. The official competition score is **not**
computed from any local backtest; this section is purely for your own
development.

Top-level reference: [Lumibot backtesting overview](https://lumibot.lumiwealth.com/backtesting.html).

### 5.1 Default mode — Pandas (CSV minute bars)

The bundled harness uses Lumibot's
[`PandasDataBacktesting`](https://lumibot.lumiwealth.com/backtesting.pandas.html)
mode against minute-bar CSVs stored in [`data/`](data/).

#### Data format

Each symbol you want to backtest must have a CSV at
`data/{SYMBOL}_1m_spot.csv` with the following columns:

| Column | Description |
| --- | --- |
| `open`, `high`, `low`, `close` | Minute-bar OHLC prices |
| `volume` | Traded volume |
| `timestamp` | Bar close timestamp, ISO-8601 with timezone (UTC) |

The repository ships with one placeholder file
(`data/EXAMPLE_1m_spot.csv`, constant price = 400, covering 1–29 Aug 2026)
so the harness runs end-to-end out of the box. Replace it (or add more
files) with realistic data of your choosing for local development —
historical CSVs are **not** used by the official evaluation.

> 📌 **The shape of [`data/EXAMPLE_1m_spot.csv`](data/EXAMPLE_1m_spot.csv)
> is canonical.** It reflects exactly what the official environment will
> feed your strategy: minute-resolution OHLCV bars per symbol, nothing
> else (no order-book depth, no macro factors, no alternative data). See
> [§8 — Evaluation & Fairness](#8-evaluation--fairness) for the full
> data-feed contract.

#### Choose your universe

Edit [`strategies/params.py`](strategies/params.py) to list the symbols you
want the local backtest to load:

```python
STOCK_SLEEVE_SYMBOLS = ["EXAMPLE"]   # add your tickers
CRYPTO_SLEEVE_SYMBOLS = []           # add crypto symbols (e.g. "BTC")
STOCK_BENCH = "EXAMPLE"              # benchmark line on the tearsheet
```

#### Run

```bash
python backtest.py
```

The harness prints the date window it is testing and produces Lumibot's
standard backtest output: a tearsheet HTML, a trades CSV, an indicators
CSV, and a logs CSV — see
[Files Generated from Backtesting](https://lumibot.lumiwealth.com/backtesting.html#files-generated-from-backtesting)
for the full list. Adjust budget, fees, slippage, and the date window at
the top of [`backtest.py`](backtest.py).

### 5.2 Other Lumibot backtest modes

Lumibot supports five additional backtest modes besides the Pandas/CSV
default — pick one that matches the asset class and data source you
prefer:

| Mode | Best for | Cost | Docs |
| --- | --- | --- | --- |
| **Yahoo** | Daily stock backtests, zero-setup smoke tests | Free | [Yahoo](https://lumibot.lumiwealth.com/backtesting.yahoo.html) |
| **Pandas** *(default here)* | Any data you can provide as CSVs | Free | [Pandas](https://lumibot.lumiwealth.com/backtesting.pandas.html) |
| **Polygon.io** | Intraday stocks / options / crypto | Free + paid tiers | [Polygon](https://lumibot.lumiwealth.com/backtesting.polygon.html) |
| **DataBento** | High-quality stocks / futures / options | Paid | [DataBento](https://lumibot.lumiwealth.com/backtesting.databento.html) |
| **ThetaData** | Stocks / options / index, intraday | Subscription | [ThetaData](https://lumibot.lumiwealth.com/backtesting.thetadata.html) |
| **Interactive Brokers (REST)** | Futures, crypto via IBKR Gateway | IBKR account | [IBKR REST](https://lumibot.lumiwealth.com/backtesting.interactive_brokers_rest.html) |

The quickest alternative to the CSV mode is **Yahoo**, which fetches daily
data from Yahoo! Finance with zero API keys. Save the snippet below as
`backtest_yahoo.py` next to `backtest.py` and run it:

```python
from datetime import datetime

from lumibot.backtesting import YahooDataBacktesting

from strategies.strategy import Strategy

Strategy.run_backtest(
    YahooDataBacktesting,
    backtesting_start=datetime(2026, 1, 1),
    backtesting_end=datetime(2026, 8, 29),
    budget=1_000_000,
    benchmark_asset="SPY",
)
```

> Yahoo backtests are **daily-only**, so set `self.sleeptime = "1D"`
> (or larger) inside your strategy's `initialize` when using this mode.

### 5.3 Fetching market data

You have three practical options:

1. **Bring your own CSVs.** Drop minute-bar files into `data/` using the
   format described in section 5.1. This works offline and is the default
   path supported by `backtest.py`.
2. **Let Lumibot pull data on the fly.** The Yahoo / Polygon / DataBento /
   ThetaData / IBKR REST modes pull data for you — follow the per-mode
   docs above for the API keys or accounts each one needs.
3. **Build a one-off downloader.** Write a small script using
   [`yfinance`](https://pypi.org/project/yfinance/),
   [`ccxt`](https://pypi.org/project/ccxt/), or any vendor SDK that
   writes `{SYMBOL}_1m_spot.csv` files into `data/`, then run
   `python backtest.py`. The example CSV bundled with the template shows
   the exact columns and timestamp format expected.

Whatever you pick, remember the **fairness rule**: the data your local
backtest sees does not influence the official score — every submission is
re-executed in the organizers' standardized environment over the official
trading window.

---

## 7) Submission Requirements

To be considered for the competition you must:

1. Push your full project to a **public GitHub repository** (or share access
   with the organizers).
2. Keep `strategies/strategy.py` as the official entrypoint — it must define
   a class `Strategy(lumibot.strategies.Strategy)` and be importable as
   `from strategies.strategy import Strategy`.
3. List **every** runtime dependency in `requirements.txt`. The organizers'
   environment will install from this file; missing or pinned-incompatible
   dependencies will cause your submission to be skipped.
4. Make sure your code is **fully reproducible**:
   - No hard-coded absolute paths.
   - No interactive prompts.
   - No reliance on local files outside the repo (anything you need must be
     committed or downloadable from a stable public URL during install).
5. Keep secrets, API keys, and personal credentials **out of the repo**
   (`.gitignore` already excludes `.env`).
6. Update this `README.md` with a short description of your approach so
   reviewers can understand and reproduce it.
7. Submit the repository link before the deadline:
   **25 July 2026, 23:59:59 SGT (UTC+8)**.

### Submission Checklist

- [ ] `strategies/strategy.py` contains a runnable `Strategy` class.
- [ ] `python backtest.py` runs end-to-end on a clean clone (after
      `pip install -r requirements.txt`).
- [ ] `requirements.txt` lists all dependencies with compatible versions.
- [ ] README describes the approach in plain language.
- [ ] No secrets, no `.env`, no large binary blobs committed.
- [ ] Repository link submitted via the official registration form.

---

## 8) Evaluation & Fairness

- **Centralized execution.** Every submission is run by the IntelligenceX
  technical team in a standardized environment over the official trading
  window. This removes latency advantages, hardware differences, and
  execution bias between participants.
- **Primary metric: Terminal Return** — the final portfolio return after
  full liquidation at **31 August 2026, 23:59:59 SGT**.
- **Only results generated by the official system are valid.** Self-reported
  backtest numbers do not count toward the leaderboard.

### Supported trading frequencies

The official environment supports **minute-, hourly-, and daily-level**
strategies, controlled via `self.sleeptime` (e.g. `"1M"`, `"5M"`, `"15M"`,
`"60M"`, `"1D"`). **Sub-minute (tick / second-level) scheduling is not
supported** — submissions that attempt it will be rejected during
verification (26–28 July 2026, SGT).

### Official data feed

Your strategy is fed **OHLCV bars only** (open, high, low, close, volume)
during the official run. There is **no order-book / Level-2 depth, no
tick data, no macroeconomic series, no news feed, and no alternative-data
source**. Bars are delivered at minute resolution per symbol; you can
choose to resample to hourly or daily inside your strategy.

Bars are sourced via **CCXT** (crypto spot pairs) and **Massive** (US
equities and ETFs) — see [§5 — Asset Universe & Data Sources](#5-asset-universe--data-sources)
for the full tradable universe and the liquidity / data caveats that come
with it.

The file [`data/EXAMPLE_1m_spot.csv`](data/EXAMPLE_1m_spot.csv) shows the
**exact shape** of the production feed — treat it as the canonical
example when you design your features, your data-loading code, and your
training pipeline.

> 🛡️ Because every team sees the same minimal information set, the
> winning strategies are the ones that stay **robust across regimes**
> (trending, mean-reverting, choppy, low-volume). Aim for a model that
> behaves well at all times rather than one that depends on rich but
> fragile signals — those signals are not available here. 🙂

---

## 9) Troubleshooting

- **`ModuleNotFoundError`** — make sure your virtual environment is active
  and `pip install -r requirements.txt` has succeeded.
- **`No valid CSV data loaded from data/`** — confirm your CSV filenames
  match `{SYMBOL}_1m_spot.csv` and that the columns include
  `open, high, low, close, volume, timestamp`.
- **`No overlapping datetime range across loaded symbols`** — your CSVs do
  not share an overlapping time window. Either widen the data or adjust
  `BACKTEST_START` / `BACKTEST_END` in `backtest.py`.
- **Strategy import errors during official verification (26–28 July 2026)** —
  re-test from a clean clone, fix any path / dependency issues, and push the
  fix before the verification window closes.

---

## 10) Contact & Links

- Conference website: <https://www.soc-ai.org/events/intelligencex-2026>
- Enquiries: **info@soc-ai.org**
- Lumibot documentation: <https://lumibot.lumiwealth.com/>
- Lumibot backtesting guide: <https://lumibot.lumiwealth.com/backtesting.html>
- Lumibot code examples: <https://lumibot.lumiwealth.com/code_examples.html>

---

## 11) License

This template is released under the [MIT License](LICENSE). Please review the license terms before
redistribution.
