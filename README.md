# BTC/USDT — Next Hour Price Range Forecaster

> **AlphaI × Polaris Build Challenge**  
> Predict Bitcoin's next 1-hour price range with 95% confidence using GBM + Monte Carlo simulation.

---

## What This Does

Every hour, a new candle closes on Bitcoin's chart. This system predicts the **price range** where BTC will land one hour from now — not an exact number, but a 95% confidence interval.

Example: *"95% sure BTC will be between $94,200 and $94,800 in the next hour."*

The best forecaster is one that is **right ~95% of the time** AND keeps the range **as narrow as possible**.

---

## Model: Rolling-Window GBM + Student-t Monte Carlo

**Core approach:**
1. Fetch the latest BTCUSDT 1-hour bars from Binance (public API, no key needed)
2. Compute **rolling EWMA volatility** (exponentially weighted, span=24 hours) — captures volatility clustering
3. Simulate **10,000 possible next-hour prices** using Geometric Brownian Motion with **Student-t distributed** shocks (df=5) — handles Bitcoin's fat tails
4. Read off the **2.5th and 97.5th percentile** → 95% prediction interval

**Key design decisions:**
- **EWMA σ instead of fixed historical σ** — Bitcoin's volatility clusters: calm hours follow calm, violent follows violent. EWMA gives more weight to recent bars, making the range adaptive.
- **Student-t (df=5) instead of Normal distribution** — Bitcoin has more extreme moves than a bell curve predicts. Student-t has heavier tails, preventing systematic under-prediction of range width.
- **Strict no-look-ahead in backtest** — At each step, the model only sees data up to the current bar. Never uses future data to "help" predictions.
- **No FIGARCH** — The starter notebook uses FIGARCH, which requires fitting a GARCH model 720 times (~hours of compute). Rolling EWMA achieves comparable results in minutes on a laptop CPU.

---

## Bug Found in Starter Notebook

**File:** `copy_of_gbm.py`, function `simulate_cyber_gbm`, lines 92-95

```python
for t in range(1, n_steps + 1):
    current = -1                          # ← BUG: hardcoded
    H_val = min(H.iloc[current] / H_max, 1.0)
    M_val = min(M.iloc[current] / M_max, 1.0)
```

`current = -1` is hardcoded inside the simulation loop. This means **every time step** reads the **last element** of the entropy (H) and momentum (M) series, regardless of which step `t` the simulation is at. In a multi-step simulation, this is **look-ahead bias** — every step uses the most recent values instead of values at that point in time. For a 1-step forecast (n_days=1) this happens to not matter, but it would produce incorrect results for any multi-step forecast.

---

## Project Structure

```
btc/
├── README.md                  ← This file
├── backtest.py                ← Part A: 30-day backtest script
├── backtest_results.jsonl     ← Generated: 720 predictions (one per line)
├── backtest_metrics.json      ← Generated: coverage, width, Winkler score
├── app.py                     ← Part B & C: Streamlit live dashboard
├── requirements.txt           ← Python dependencies
├── prediction_history.json    ← Part C: growing log of live predictions
├── backtest_chart.png         ← Generated: backtest visualization
├── copy_of_gbm.py             ← Original starter notebook (reference only)
└── alphai_polaris_challenge_guide.md  ← Challenge guide (reference)
```

---

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Part A — Run Backtest
```bash
cd btc
python backtest.py
```
This will:
- Fetch ~1000 BTCUSDT 1-hour bars from Binance
- Run a 720-bar walk-forward backtest
- Print coverage, avg width, and Winkler score
- Save `backtest_results.jsonl` and `backtest_metrics.json`
- Generate `backtest_chart.png`

### Part B — Launch Dashboard Locally
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. Shows:
- Current BTC price + live 95% prediction range
- Backtest metrics (coverage, width, Winkler)
- Interactive Plotly chart with prediction ribbon
- Monte Carlo simulation distribution

### Part C — Prediction Persistence
Predictions are automatically saved to `prediction_history.json` on each dashboard visit. The history table shows past predictions with hit/miss results as actual prices become available.

---

## Part A Results

| Metric | Value | Target |
|--------|-------|--------|
| Coverage (95%) | 0.9624 | ~0.9500 |
| Average Width | $1,379.18 | As narrow as possible |
| Winkler Score | 1,775.18 | Lower = better |
| Total Predictions | 719 | 720 |

---

## Dependencies

- `numpy`, `pandas` — Data handling
- `scipy` — Student-t distribution
- `requests` — Binance API calls
- `streamlit` — Dashboard framework
- `plotly` — Interactive charts
- `tqdm` — Progress bars for backtest

No `arch` library needed. No API keys needed.

---

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New App → Connect repo
3. Set main file to `app.py`
4. Deploy — gives a public URL like `https://yourapp.streamlit.app`

---

## Technical Notes

- **Data source:** `https://data-api.binance.vision/api/v3/klines` (geo-unblocked, works in India)
- **Winkler score formula:** Width if hit, width + 40× miss distance if missed (α=0.05)
- **EWMA span=24:** ~24-hour half-life for volatility estimation, responsive to regime changes
- **Student-t df=6:** Tuned via grid search. Balances fat tails without being too extreme (df=3 is too heavy/wide, df=7 is close to normal/narrow)
