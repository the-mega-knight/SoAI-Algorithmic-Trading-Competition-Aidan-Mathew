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

## Our Approach: Low-Turnover Momentum Core with Volatility/Gap Risk Throttle

**Team**: Aidan & Mathew

**One-sentence hypothesis**: among a basket of large-cap US tech stocks,
names showing sustained positive 30-day momentum tend to keep outperforming
long enough to be worth holding through short-term noise, so the strategy
stays close to fully invested and rebalances only weekly, while a separate
daily-checked volatility/gap throttle cuts exposure to any name showing an
abnormal move.

**Universe**: AAPL, MSFT, GOOGL, AMZN, NVDA. Five liquid, large-cap US
equities, chosen for data availability and familiarity.

**Why this design, and not daily trend-following**: two earlier versions of
this strategy (a crypto momentum version, then a daily SMA-crossover version
on this same equity universe) were both validated against real historical
data and both underperformed simple buy-and-hold by a wide margin. The
common cause was heavy turnover from daily binary entry/exit trading, mostly
whipsaw on short-term noise rather than genuine trend changes, with fees
eating a large share of the return. Every version tested pointed the same
direction: staying invested consistently beat trading in and out. This
version is built around that result.

**Cadence**: checked once per day (`self.sleeptime = "1D"`), but the target
allocation is only *recomputed* every 5 trading days. The risk throttle
(below) still runs every day regardless of the rebalance schedule.

**Rules (fully mechanical, no discretionary judgment calls)**:
- **Entry signal**: 30-day rate of change > 0
- **Weighting**: proportional to each qualifying name's own ROC (not
  inverse-volatility, which was found to penalize exactly the strongest
  movers), capped at 40% of the portfolio per name and 98% total exposure
- **Risk throttle** (checked daily, OHLCV-only): halves a name's weight if
  its 20-day realized volatility exceeds 1.8x its own 60-day baseline, and
  cuts it to 30% for 3 trading days after a >5% overnight gap. This is a
  reactive, OHLCV-only stand-in for earnings/shock awareness. The official
  feed provides OHLCV bars only, no news or calendar data, so a strategy
  that depended on knowing earnings dates in advance would not be reliably
  reproducible in the official environment.
- **Risk control**: a portfolio-level drawdown circuit breaker. If the
  portfolio falls more than 25% from its running peak, flatten everything
  and pause new entries for 5 trading days. When cooldown ends, the peak
  resets to the recovery value rather than the pre-crash high, so the
  breaker cannot permanently lock the strategy out of the market after a
  single drawdown (a bug caught and fixed during validation).
- **No-trade band**: only rebalance a name when the target position
  differs from the current one by more than 2% of portfolio value

**Local validation on real historical data** (`vector_validate.py`, a
from-scratch pandas re-implementation of the same rules, cross-checked
against `python backtest.py` running the real Lumibot engine):
- **+73.1% terminal return, Sharpe 1.70, Sortino 2.75, max drawdown -15.9%**
  over a real ~436-day window (real daily OHLCV pulled via yfinance). A
  separate full 2-year Lumibot run confirmed the same direction: +107%
  total return, Sharpe 1.95, max drawdown -15.0%, against a +46% AAPL-only
  benchmark over the same period.
- Positive in **all three** sub-period thirds (+17.4%, +35.0%, +7.3%), not
  dependent on one lucky stretch.
- The parameter grid around these settings (30-day momentum lookback,
  5-day rebalance) forms a genuine plateau of strong Sharpe ratios rather
  than an isolated spike, and doubling transaction fees only costs about
  4 points of return. Both are signs the result isn't curve-fit to this
  specific window.
- Beats 4 of the 5 individual buy-and-hold benchmarks over the same window
  (AAPL +39.5%, MSFT +21.4%, AMZN +39.1%, NVDA +51.7%), with a much
  shallower drawdown than any single concentrated position would carry. It
  does not beat GOOGL's standalone +123.9% buy-and-hold, which used no
  diversification or risk control at all.

**Known limitations, stated plainly**:
- All validation above is against **historical** data ending in early
  August 2026. The official scored window (16 August - 15 September 2026)
  has not happened yet at the time of writing, and nothing guarantees it
  resembles the backtested period.
- The drawdown circuit breaker has not fired in real-data validation (max
  drawdown stayed at -15.9%, short of the 25% threshold), so whether it
  works as intended in an actual severe drawdown is still unconfirmed on
  real data.
- Five large-cap tech names are meaningfully correlated with each other;
  the strategy does not diversify across sectors or asset classes even
  though the competition permits a much wider universe (any CCXT crypto
  pair, or the full US equity universe via Massive).
- `requirements.txt` should be pinned to exact installed versions
  (`pip freeze`) before final submission, per the competition's stated
  reproducibility requirement.

---

## 1) Competition at a Glance

| Item | Value |
| --- | --- |
| Code submission deadline | **9 August 2026, 23:59:59 SGT (UTC+8)** |
| Verification & test run | 10–12 August 2026 (SGT) |
| Official trading period | 16 August 2026, 00:00:00 → 15 September 2026, 23:59:59 (SGT) |
| Winners announcement | 18 September 2026 (SGT) |
| Primary evaluation metric | **Terminal Return** (final portfolio value after full liquidation at the end of the trading window) |
| Total prize pool | **SGD 3,000** |
| Eligibility | Open worldwide; individuals or teams of 1–5 |
| Contact | info@soc-ai.org |

> ⚠️ **Prize condition (strict):** cash prizes are awarded only to winners who
> present **in person** at IntelligenceX 2026 in Singapore. No remote
> disbursement or money transfer will be arranged.

For the full call for participation, see the conference page:
<https://www.soc-ai.org/events/intelligencex-2026>.

---

## 2) Repository Layout

```text
SoAI-Algorithmic-Trading-Competition-Aidan-Mathew/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── backtest.py                       # local backtest entrypoint (pandas / CSV, daily bars)
├── vector_validate.py                # independent pandas re-implementation for fast iteration
├── strategies/
│   ├── strategy.py                   # OUR strategy (official entrypoint)
│   ├── params.py                     # shared parameters for local backtests
│   ├── example_strategy_1.py         # reference: daily DCA into SPY
│   └── example_strategy_2.py         # reference: one-shot buy & hold
├── scripts/
│   └── fetch_stock_data.py           # pulls real daily OHLCV via yfinance into data/
└── data/
    └── {SYMBOL}_daily.csv            # real daily OHLCV per symbol (gitignored; regenerate locally)
```

Files the official environment relies on:

- **`strategies/strategy.py`** — must define a class named `Strategy` that
  subclasses `lumibot.strategies.Strategy`. This is what the organizers import
  and run.
- **`requirements.txt`** — must list every Python dependency your strategy
  needs. The organizers install from this file.
- **`README.md`** — keep a short, accurate description of your approach so
  reviewers can reproduce it.

Everything else (the `backtest.py` / `vector_validate.py` harnesses, the
`data/` folder, the example strategies) is provided for **your local
development only** and is not used by the official evaluation.

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
stocks, from BTC to meme coins, if it has data, you can trade it. This
repo's strategy deliberately trades a small five-stock universe (see
"Our Approach" above) rather than the full available universe, as a
conscious build-time and reliability tradeoff.

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
2. **You can source its historical data** for your local backtest.

### 📡 Official data source providers (August–September 2026 run)

During the verification run (10–12 August 2026) and the official trading
window (16 August – 15 September 2026, SGT), every submission receives
bars from:

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
3. **Free 1-minute history is scarce for US equities.** Free equity feeds
   (e.g. Yahoo/yfinance) typically expose only a few days of intraday
   data, which is why this repo's local backtest uses daily bars instead
   (see §6). The organizers handle minute-resolution data during the
   official run — you only need historical data for local development.
4. **Survivorship bias.** Penny stocks and small-cap altcoins delist or
   go to zero frequently. The historical datasets you pull in 2026 will
   be missing many tickers that were tradable earlier — train your
   models with that in mind. The official trading universe is whatever
   is *live and listed* during the trading window.

---

## 6) Local Backtesting

This repo ships two local validation paths: a **Pandas / CSV** Lumibot
backtest (`backtest.py`) and a **from-scratch pandas re-implementation**
(`vector_validate.py`) used for fast parameter iteration and stress-testing
without needing a full Lumibot install. The official competition score is
**not** computed from either; this section is purely for your own
development.

Top-level reference: [Lumibot backtesting overview](https://lumibot.lumiwealth.com/backtesting.html).

### 6.1 `backtest.py` — Pandas (CSV daily bars)

The bundled harness uses Lumibot's
[`PandasDataBacktesting`](https://lumibot.lumiwealth.com/backtesting.pandas.html)
mode against **daily-bar** CSVs stored in [`data/`](data/) (this repo's
strategy trades on a daily cadence, so daily bars are sufficient locally;
the official feed itself remains minute-resolution per §8).

#### Data format

Each symbol you want to backtest must have a CSV at
`data/{SYMBOL}_daily.csv` with the following columns:

| Column | Description |
| --- | --- |
| `open`, `high`, `low`, `close` | Daily OHLC prices |
| `volume` | Traded volume |
| `timestamp` | Bar date, ISO-8601 |

Run `python scripts/fetch_stock_data.py` to pull real data via `yfinance`
for the symbols listed in `strategies/params.py` (data/ is gitignored, so
a fresh clone needs to regenerate it before backtesting).

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

### 6.2 `vector_validate.py` — fast pandas re-implementation

```bash
python vector_validate.py
```

Runs the same decision rules as `strategies/strategy.py`, independently
re-derived in plain pandas, against the same daily CSVs. Prints a baseline
run, a parameter sensitivity grid, stress tests (2x fees, circuit breaker
disabled), sub-period robustness, and a buy-and-hold benchmark comparison,
the full workflow from `trading-backtest-methodology`. Useful for
iterating quickly since it doesn't require a Lumibot install.

### 6.3 Other Lumibot backtest modes

Lumibot supports several other backtest modes — pick one that matches the
asset class and data source you prefer:

| Mode | Best for | Cost | Docs |
| --- | --- | --- | --- |
| **Yahoo** | Daily stock backtests, zero-setup smoke tests | Free | [Yahoo](https://lumibot.lumiwealth.com/backtesting.yahoo.html) |
| **Pandas** *(default here)* | Any data you can provide as CSVs | Free | [Pandas](https://lumibot.lumiwealth.com/backtesting.pandas.html) |
| **Polygon.io** | Intraday stocks / options / crypto | Free + paid tiers | [Polygon](https://lumibot.lumiwealth.com/backtesting.polygon.html) |
| **DataBento** | High-quality stocks / futures / options | Paid | [DataBento](https://lumibot.lumiwealth.com/backtesting.databento.html) |
| **ThetaData** | Stocks / options / index, intraday | Subscription | [ThetaData](https://lumibot.lumiwealth.com/backtesting.thetadata.html) |
| **Interactive Brokers (REST)** | Futures, crypto via IBKR Gateway | IBKR account | [IBKR REST](https://lumibot.lumiwealth.com/backtesting.interactive_brokers_rest.html) |

### 6.4 Fetching market data

This repo uses `scripts/fetch_stock_data.py` (yfinance, daily bars, no API
key needed). Whatever source you use, remember the **fairness rule**: the
data your local backtest sees does not influence the official score —
every submission is re-executed in the organizers' standardized
environment over the official trading window.

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
   **9 August 2026, 23:59:59 SGT (UTC+8)**, via the competition site's
   submission form and registration flow (separate from pushing to GitHub).

### Submission Checklist

- [x] `strategies/strategy.py` contains a runnable `Strategy` class.
- [x] `python backtest.py` runs end-to-end on a clean clone (after
      `pip install -r requirements.txt` and `python scripts/fetch_stock_data.py`).
- [ ] `requirements.txt` lists all dependencies with pinned versions
      (currently unpinned, run `pip freeze` before final submission).
- [x] README describes the approach in plain language.
- [ ] No secrets, no `.env`, no large binary blobs committed.
- [ ] Repository link submitted via the official registration form.

---

## 8) Evaluation & Fairness

- **Centralized execution.** Every submission is run by the IntelligenceX
  technical team in a standardized environment over the official trading
  window. This removes latency advantages, hardware differences, and
  execution bias between participants.
- **Primary metric: Terminal Return** — the final portfolio value after
  full liquidation at **15 September 2026, 23:59:59 SGT**.
- **Only results generated by the official system are valid.** Self-reported
  backtest numbers do not count toward the leaderboard.

### Supported trading frequencies

The official environment supports **minute-, hourly-, and daily-level**
strategies, controlled via `self.sleeptime` (e.g. `"1M"`, `"5M"`, `"15M"`,
`"60M"`, `"1D"`). **Sub-minute (tick / second-level) scheduling is not
supported** — submissions that attempt it will be rejected during
verification (10–12 August 2026, SGT).

### Official data feed

Your strategy is fed **OHLCV bars only** (open, high, low, close, volume)
during the official run. There is **no order-book / Level-2 depth, no
tick data, no macroeconomic series, no news feed, and no alternative-data
source**. Bars are delivered at minute resolution per symbol; you can
choose to resample to hourly or daily inside your strategy (this repo's
strategy requests daily bars directly via `get_historical_prices(symbol,
N, "day")`, one of the explicitly supported cadences).

Bars are sourced via **CCXT** (crypto spot pairs) and **Massive** (US
equities and ETFs) — see [§5 — Asset Universe & Data Sources](#5-asset-universe--data-sources)
for the full tradable universe and the liquidity / data caveats that come
with it.

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
  match `{SYMBOL}_daily.csv` and that the columns include
  `open, high, low, close, volume, timestamp`. Run
  `python scripts/fetch_stock_data.py` if `data/` is empty (it's gitignored).
- **`No overlapping datetime range across loaded symbols`** — your CSVs do
  not share an overlapping time window. Either widen the data or adjust
  `BACKTEST_START` / `BACKTEST_END` in `backtest.py`.
- **Strategy import errors during official verification (10–12 August 2026)** —
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
