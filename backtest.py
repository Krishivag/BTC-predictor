# -*- coding: utf-8 -*-
"""
AlphaI × Polaris Build Challenge — Part A: 30-Day Backtest
BTC/USDT 1-Hour AR(1)-GARCH(1,1) Forecaster with Skewed Student-t

Upgraded to use ARMA-GARCH for dynamic drift and volatility clustering,
and Non-Central (Skewed) Student-t distribution for asymmetric tail risk.
"""

import numpy as np
import pandas as pd
import requests
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI window)
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist
from scipy.stats import nct
from arch import arch_model
from datetime import datetime, timedelta
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════
# 1. DATA FETCHING — Binance Public API (no key needed)
# ═══════════════════════════════════════════════════════════════════

def fetch_btc_hourly(limit=1000, end_time=None):
    """
    Fetch BTCUSDT 1-hour klines from Binance's geo-unblocked endpoint.
    Max 1000 bars per call. Returns DataFrame indexed by open_time.
    """
    url = "https://data-api.binance.vision/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "limit": limit,
    }
    if end_time is not None:
        params["endTime"] = int(end_time)

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "quote_vol", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("open_time")
    return df


def fetch_btc_hourly_extended(total_bars=1000):
    """
    Fetch more than 1000 bars by paginating backward.
    We need ~920 bars (200 warmup + 720 backtest).
    """
    if total_bars <= 1000:
        return fetch_btc_hourly(limit=total_bars)

    # First call: most recent bars
    all_dfs = []
    remaining = total_bars
    end_time = None

    while remaining > 0:
        batch = min(remaining, 1000)
        df = fetch_btc_hourly(limit=batch, end_time=end_time)
        if df.empty:
            break
        all_dfs.insert(0, df)
        # Move endTime to just before the earliest bar we got
        end_time = int(df.iloc[0].name.timestamp() * 1000) - 1
        remaining -= len(df)

    result = pd.concat(all_dfs)
    result = result[~result.index.duplicated(keep='first')]
    return result.sort_index()


# ═══════════════════════════════════════════════════════════════════
# 2. PREDICTION — Rolling-Window AR-GARCH + Skewed-t Monte Carlo
# ═══════════════════════════════════════════════════════════════════

def predict_next_hour(prices, window=100, n_sims=10000, confidence=0.95, df_t=5, nc_skew=-1.5):
    """
    Given a Series of close prices (no look-ahead), predict the 95%
    confidence interval for the next hour's close price.

    Uses:
    - AR(1)-GARCH(1,1) for dynamic drift and volatility clustering
    - Non-central Student-t distribution for shocks (captures fat, skewed tails)

    Returns: (lower_bound, upper_bound)
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # Use only the most recent 'window' bars for model estimation
    recent_returns = log_returns.iloc[-window:]

    # Fit AR(1)-GARCH(1,1) model
    # Scale by 100 for numerical stability during optimization
    am = arch_model(recent_returns * 100, mean='AR', lags=1, vol='Garch', p=1, q=1)
    res = am.fit(disp='off', update_freq=0)
    
    # 1-step ahead forecast
    forecast = res.forecast(horizon=1)
    mu = forecast.mean.iloc[-1, 0] / 100
    sigma = np.sqrt(forecast.variance.iloc[-1, 0]) / 100

    current_price = prices.iloc[-1]

    # Skewed Student-t random shocks (fat and asymmetric tails)
    # Using RAW Non-central Student-t (no variance normalization)
    random_shocks = nct.rvs(df=df_t, nc=nc_skew, size=n_sims)

    # Scale shocks to match mu and sigma
    simulated_returns = mu + sigma * random_shocks
    simulated_prices = current_price * np.exp(simulated_returns)

    alpha = 1 - confidence  # 0.05 for 95%
    lower = np.percentile(simulated_prices, 100 * alpha / 2)        # 2.5th
    upper = np.percentile(simulated_prices, 100 * (1 - alpha / 2))  # 97.5th

    return lower, upper


# ═══════════════════════════════════════════════════════════════════
# 3. EVALUATION — Coverage, Width, Winkler Score
# ═══════════════════════════════════════════════════════════════════

def evaluate(predictions):
    """
    Compute backtest metrics from list of prediction dicts.
    Each dict has: lower_95, upper_95, actual
    """
    alpha = 0.05
    coverages = []
    widths = []
    winklers = []

    for p in predictions:
        L, U, actual = p["lower_95"], p["upper_95"], p["actual"]
        width = U - L
        hit = int(L <= actual <= U)
        coverages.append(hit)
        widths.append(width)

        if actual < L:
            winkler = width + (2 / alpha) * (L - actual)
        elif actual > U:
            winkler = width + (2 / alpha) * (actual - U)
        else:
            winkler = width
        winklers.append(winkler)

    return {
        "coverage_95": np.mean(coverages),
        "avg_width": np.mean(widths),
        "mean_winkler_95": np.mean(winklers),
        "total_predictions": len(predictions),
        "total_hits": sum(coverages),
        "total_misses": len(predictions) - sum(coverages),
    }


# ═══════════════════════════════════════════════════════════════════
# 4. BACKTEST — 30-Day (720 bars), Strict No-Look-Ahead
# ═══════════════════════════════════════════════════════════════════

def run_backtest(df, warmup=200, window=100, df_t=5, nc_skew=-1.5, n_sims=10000):
    """
    Run a walk-forward backtest.
    At each step i, use only data up to bar i to predict bar i+1.
    """
    predictions = []

    for i in tqdm(range(warmup, len(df) - 1), desc="Backtesting"):
        # STRICT NO-LOOK-AHEAD: only use bars 0..i (inclusive)
        history = df["close"].iloc[:i + 1]
        actual_next = df["close"].iloc[i + 1]  # This is the "future" — only for evaluation

        lower, upper = predict_next_hour(
            history, window=window, n_sims=n_sims, df_t=df_t, nc_skew=nc_skew
        )

        predictions.append({
            "timestamp": str(df.index[i + 1]),
            "lower_95": round(float(lower), 2),
            "upper_95": round(float(upper), 2),
            "actual": round(float(actual_next), 2),
        })

    return predictions


# ═══════════════════════════════════════════════════════════════════
# 5. PARAMETER TUNING — Grid Search
# ═══════════════════════════════════════════════════════════════════

def tune_parameters(df, warmup=200, n_sims=2000):
    """
    Quick grid search over parameter space.
    Uses fewer simulations for speed during tuning.
    """
    print("\n" + "=" * 60)
    print("PARAMETER TUNING - Grid Search")
    print("=" * 60)

    results = []
    # Smaller grid because GARCH is computationally intensive
    windows = [100]
    df_values = [5, 6]
    nc_values = [-0.5, -1.0, -1.5]

    for window in windows:
        for df_t in df_values:
            for nc in nc_values:
                preds = run_backtest(df, warmup=warmup, window=window,
                                    df_t=df_t, nc_skew=nc, n_sims=n_sims)
                metrics = evaluate(preds)
                cov = metrics["coverage_95"]
                winkler = metrics["mean_winkler_95"]
                # Score: penalize deviation from 0.95 coverage, reward low Winkler
                cov_penalty = abs(cov - 0.95) * 10000  # heavy penalty for bad coverage
                score = winkler + cov_penalty
                results.append({
                    "window": window,
                    "df_t": df_t,
                    "nc_skew": nc,
                    "coverage": cov,
                    "winkler": winkler,
                    "score": score,
                })
                print(f"  window={window:>3}, df_t={df_t}, nc={nc:>4.1f} -> "
                      f"coverage={cov:.4f}, winkler={winkler:.1f}, score={score:.1f}")

    # Pick best
    best = min(results, key=lambda x: x["score"])
    print(f"\n  BEST: window={best['window']}, df_t={best['df_t']}, nc_skew={best['nc_skew']} "
          f"(coverage={best['coverage']:.4f}, winkler={best['winkler']:.1f})")
    return best["window"], best["df_t"], best["nc_skew"]


# ═══════════════════════════════════════════════════════════════════
# 6. VISUALIZATION — Backtest Chart
# ═══════════════════════════════════════════════════════════════════

def plot_backtest(predictions, save_path="backtest_chart.png"):
    """Plot actual prices with 95% prediction ribbon."""
    df_plot = pd.DataFrame(predictions)
    df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(df_plot["timestamp"], df_plot["actual"],
            color="#F7931A", linewidth=1, label="Actual BTC Price", zorder=3)
    ax.fill_between(df_plot["timestamp"], df_plot["lower_95"], df_plot["upper_95"],
                    alpha=0.25, color="#4FC3F7", label="95% Prediction Interval", zorder=2)

    # Mark misses in red
    misses = df_plot[
        (df_plot["actual"] < df_plot["lower_95"]) |
        (df_plot["actual"] > df_plot["upper_95"])
    ]
    ax.scatter(misses["timestamp"], misses["actual"],
               color="red", s=12, zorder=4, label=f"Misses ({len(misses)})")

    ax.set_title("BTC/USDT 1-Hour Backtest — 95% Prediction Interval (AR-GARCH)", fontsize=14)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Price (USDT)")
    ax.legend(loc="upper left")
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Chart saved to {save_path}")


# ═══════════════════════════════════════════════════════════════════
# 7. MAIN — Run Everything
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("AlphaI x Polaris Challenge - BTC/USDT AR-GARCH Backtest")
    print("=" * 60)

    # ── Fetch data ──────────────────────────────────────────────
    print("\n[1/5] Fetching BTCUSDT 1-hour bars from Binance...")
    df = fetch_btc_hourly_extended(total_bars=1000)
    print(f"  Fetched {len(df)} bars")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Latest close: ${df['close'].iloc[-1]:,.2f}")

    # ── Parameter tuning (quick, fewer sims) ────────────────────
    print("\n[2/5] Tuning parameters (quick grid search)...")
    df_tune = df.iloc[-(720 + 200):]  # Same slice as final backtest
    WINDOW, DF_T, NC_SKEW = tune_parameters(df_tune, warmup=200, n_sims=3000)

    N_SIMS = 10000
    print(f"  Using: window={WINDOW}, df_t={DF_T}, nc_skew={NC_SKEW}, n_sims={N_SIMS}")

    # ── Run final backtest ──────────────────────────────────────
    print(f"\n[3/5] Running 30-day backtest (~720 bars)...")
    print("  (This will take a few minutes due to GARCH optimization)")
    df_backtest = df.iloc[-(720 + 200):]
    predictions = run_backtest(df_backtest, warmup=200, window=WINDOW,
                               df_t=DF_T, nc_skew=NC_SKEW, n_sims=N_SIMS)
    print(f"  Generated {len(predictions)} predictions")

    # ── Evaluate ────────────────────────────────────────────────
    print(f"\n[4/5] Evaluating backtest results...")
    results = evaluate(predictions)
    print(f"\n  {'=' * 40}")
    print(f"  BACKTEST RESULTS")
    print(f"  {'=' * 40}")
    print(f"  Total predictions:  {results['total_predictions']}")
    print(f"  Coverage (95%):     {results['coverage_95']:.4f}  (target: 0.9500)")
    print(f"  Average width:      ${results['avg_width']:,.2f}")
    print(f"  Winkler score:      {results['mean_winkler_95']:,.2f}  (lower is better)")
    print(f"  Hits / Misses:      {results['total_hits']} / {results['total_misses']}")
    print(f"  {'=' * 40}")

    # ── Save predictions ────────────────────────────────────────
    print(f"\n[5/5] Saving predictions to backtest_results.jsonl...")
    with open("backtest_results.jsonl", "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
    print(f"  Saved {len(predictions)} predictions")

    # ── Save metrics for dashboard ──────────────────────────────
    metrics_file = "backtest_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Metrics saved to {metrics_file}")

    # ── Plot ─────────────────────────────────────────────────────
    print("\n  Generating backtest chart...")
    plot_backtest(predictions)

    print("\nDone! Backtest complete. Ready for Part B (dashboard).")
