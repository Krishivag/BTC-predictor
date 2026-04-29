# -*- coding: utf-8 -*-
"""
AlphaI × Polaris Build Challenge — Part B & C: Live Dashboard
BTC/USDT 1-Hour GBM Forecaster with Prediction Persistence

Deploy: streamlit run app.py
Host:   Streamlit Community Cloud (free) — push to GitHub, connect repo
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
from scipy.stats import t as t_dist
from scipy.stats import nct
from arch import arch_model
import plotly.graph_objects as go
import json
import os
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════════════════
# CONFIG — Update these after running backtest.py
# ═══════════════════════════════════════════════════════════════════

# Model parameters (from backtest tuning)
WINDOW = 100        # rolling window for volatility estimation
DF_T = 6            # Student-t degrees of freedom (tuned via grid search)
NC_SKEW = -1.0      # Skew parameter for Non-central Student-t
N_SIMS = 10000      # number of Monte Carlo simulations

# Part A backtest metrics — FILL THESE IN after running backtest.py
# These are displayed as headline numbers on the dashboard
BACKTEST_METRICS = {
    "coverage_95": 0.0000,       # ← Update after backtest
    "avg_width": 0.00,           # ← Update after backtest
    "mean_winkler_95": 0.00,     # ← Update after backtest
}

# Try to load metrics from file (auto-populated by backtest.py)
METRICS_FILE = "backtest_metrics.json"
if os.path.exists(METRICS_FILE):
    with open(METRICS_FILE) as f:
        saved = json.load(f)
        BACKTEST_METRICS["coverage_95"] = saved.get("coverage_95", 0)
        BACKTEST_METRICS["avg_width"] = saved.get("avg_width", 0)
        BACKTEST_METRICS["mean_winkler_95"] = saved.get("mean_winkler_95", 0)

# Persistence file for Part C
HISTORY_FILE = "prediction_history.json"


# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="BTC/USDT — Next Hour Forecast",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Dark theme enhancements */
    .stMetric {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(100, 200, 255, 0.15);
    }
    .stMetric label {
        color: #8892b0 !important;
        font-size: 0.85rem !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #ccd6f6 !important;
        font-size: 1.8rem !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
    }
    div[data-testid="stHorizontalBlock"] > div {
        padding: 4px;
    }
    /* Header styling */
    h1 {
        background: linear-gradient(90deg, #F7931A, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem !important;
    }
    /* Divider */
    hr {
        border-color: rgba(100, 200, 255, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# DATA FETCHING — Binance Public API
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)  # Cache for 5 minutes, then refresh
def fetch_data(limit=500):
    """Fetch latest BTCUSDT 1-hour bars from Binance."""
    url = "https://data-api.binance.vision/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1h", "limit": limit}
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
    return df.set_index("open_time")


# ═══════════════════════════════════════════════════════════════════
# MODEL — Same as backtest.py (consistency is critical)
# ═══════════════════════════════════════════════════════════════════

def predict(prices, window=WINDOW, n_sims=N_SIMS, df_t=DF_T, nc_skew=NC_SKEW):
    """
    Predict 95% confidence interval for the next hour's BTC price.
    Uses rolling-window AR-GARCH with Skewed Student-t Monte Carlo simulation.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    recent = log_returns.iloc[-window:]

    # Fit AR(1)-GARCH(1,1)
    am = arch_model(recent * 100, mean='AR', lags=1, vol='Garch', p=1, q=1)
    res = am.fit(disp='off', update_freq=0)
    
    # 1-step ahead forecast
    forecast = res.forecast(horizon=1)
    mu = forecast.mean.iloc[-1, 0] / 100
    sigma = np.sqrt(forecast.variance.iloc[-1, 0]) / 100

    current = prices.iloc[-1]

    # Skewed Student-t shocks (raw, no variance normalization)
    shocks = nct.rvs(df=df_t, nc=nc_skew, size=n_sims)

    # GBM: mu + sigma * Z (as per guide)
    sim_returns = mu + sigma * shocks
    sim_prices = current * np.exp(sim_returns)

    lower = np.percentile(sim_prices, 2.5)
    upper = np.percentile(sim_prices, 97.5)
    return lower, upper, sim_prices


# ═══════════════════════════════════════════════════════════════════
# PART C — Prediction Persistence
# ═══════════════════════════════════════════════════════════════════

def save_prediction(lower, upper, current_price):
    """Save current prediction to history file."""
    history = load_history()

    # Don't save duplicate if last prediction was < 5 min ago
    now = datetime.now(timezone.utc)
    if history:
        last_ts = history[-1]["timestamp"]
        last_time = datetime.fromisoformat(last_ts)
        # Ensure timezone-aware comparison
        if last_time.tzinfo is None:
            last_time = last_time.replace(tzinfo=timezone.utc)
        if (now - last_time).total_seconds() < 300:
            return  # Skip -- too recent

    history.append({
        "timestamp": now.isoformat(),
        "current_price": round(float(current_price), 2),
        "lower_95": round(float(lower), 2),
        "upper_95": round(float(upper), 2),
        "actual": None,  # Filled in on next visit
    })

    # Keep only last 500 predictions to prevent file bloat
    history = history[-500:]

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def load_history():
    """Load prediction history from file."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def backfill_actuals(history, df):
    """
    For past predictions, fill in the actual price that was observed
    one hour after the prediction timestamp.
    """
    updated = False
    for entry in history:
        if entry["actual"] is not None:
            continue
        pred_time = pd.to_datetime(entry["timestamp"])
        # Strip timezone info to match df.index (which is tz-naive from Binance)
        if pred_time.tzinfo is not None:
            pred_time = pred_time.tz_convert(None)
        # Find the next hourly close after prediction
        future_bars = df[df.index > pred_time]
        if not future_bars.empty:
            entry["actual"] = round(float(future_bars["close"].iloc[0]), 2)
            updated = True

    if updated:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)

    return history


# ═══════════════════════════════════════════════════════════════════
# DASHBOARD UI
# ═══════════════════════════════════════════════════════════════════

# ── Title ────────────────────────────────────────────────────────
st.title("₿ BTC/USDT — Next Hour Forecast")
st.caption("Powered by AR-GARCH + Skewed Student-t Monte Carlo Simulation  |  AlphaI × Polaris Build Challenge")

# ── Fetch data & predict ─────────────────────────────────────────
try:
    df = fetch_data(limit=500)
    current_price = df["close"].iloc[-1]
    lower, upper, sim_prices = predict(df["close"])

    # Part C: Save prediction & backfill actuals
    save_prediction(lower, upper, current_price)
    history = load_history()
    history = backfill_actuals(history, df)

except Exception as e:
    st.error(f"⚠️ Error fetching data from Binance: {e}")
    st.info("This may be a temporary network issue. Refresh the page to retry.")
    st.stop()

# ── Headline Numbers ─────────────────────────────────────────────
st.markdown("### 📊 Live Prediction")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Current BTC Price", f"${current_price:,.2f}")
with col2:
    st.metric("Lower Bound (95%)", f"${lower:,.2f}")
with col3:
    st.metric("Upper Bound (95%)", f"${upper:,.2f}")
with col4:
    width = upper - lower
    st.metric("Range Width", f"${width:,.2f}")

st.divider()

# ── Backtest Metrics ─────────────────────────────────────────────
st.markdown("### 📈 Backtest Performance (30-Day, ~720 Bars)")
m1, m2, m3 = st.columns(3)

with m1:
    cov = BACKTEST_METRICS["coverage_95"]
    delta_cov = f"{'✓ On target' if 0.93 <= cov <= 0.97 else '⚠ Needs tuning'}"
    st.metric("Coverage", f"{cov:.4f}", delta_cov)
with m2:
    st.metric("Avg Range Width", f"${BACKTEST_METRICS['avg_width']:,.2f}")
with m3:
    st.metric("Winkler Score", f"{BACKTEST_METRICS['mean_winkler_95']:,.2f}", "Lower = Better")

st.divider()

# ── Chart: Last 50 Bars + Prediction Ribbon ──────────────────────
st.markdown("### 🕐 Last 50 Bars + Next-Hour Prediction (AR-GARCH)")
df_plot = df.tail(50).copy()

fig = go.Figure()

# Price line (Bitcoin orange)
fig.add_trace(go.Scatter(
    x=df_plot.index,
    y=df_plot["close"],
    mode="lines+markers",
    name="BTC Close",
    line=dict(color="#F7931A", width=2.5),
    marker=dict(size=3, color="#F7931A"),
))

# High/Low range as subtle background
fig.add_trace(go.Scatter(
    x=df_plot.index,
    y=df_plot["high"],
    mode="lines",
    line=dict(width=0),
    showlegend=False,
    hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=df_plot.index,
    y=df_plot["low"],
    mode="lines",
    line=dict(width=0),
    fill="tonexty",
    fillcolor="rgba(247, 147, 26, 0.05)",
    showlegend=False,
    hoverinfo="skip",
))

# Next-hour prediction ribbon
next_time = df_plot.index[-1] + pd.Timedelta(hours=1)
fig.add_shape(
    type="rect",
    x0=df_plot.index[-1], x1=next_time,
    y0=lower, y1=upper,
    fillcolor="rgba(79, 195, 247, 0.25)",
    line=dict(color="rgba(79, 195, 247, 0.6)", width=1),
)

# Prediction annotation
fig.add_annotation(
    x=next_time, y=(lower + upper) / 2,
    text=f"<b>95% Range</b><br>${lower:,.0f} – ${upper:,.0f}",
    showarrow=True, arrowhead=2, arrowcolor="#4FC3F7",
    font=dict(size=11, color="#4FC3F7"),
    bgcolor="rgba(26, 26, 46, 0.8)",
    bordercolor="#4FC3F7",
    borderwidth=1,
    borderpad=6,
)

# Current price marker
fig.add_trace(go.Scatter(
    x=[df_plot.index[-1]],
    y=[current_price],
    mode="markers",
    marker=dict(size=10, color="#FFD700", symbol="diamond"),
    name=f"Current: ${current_price:,.2f}",
))

fig.update_layout(
    xaxis_title="Time (UTC)",
    yaxis_title="Price (USDT)",
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font=dict(color="#ccd6f6", family="Inter, sans-serif"),
    height=500,
    margin=dict(l=60, r=30, t=30, b=50),
    legend=dict(
        bgcolor="rgba(26, 26, 46, 0.8)",
        bordercolor="rgba(100, 200, 255, 0.2)",
        borderwidth=1,
        font=dict(size=11),
    ),
    xaxis=dict(
        gridcolor="rgba(100, 200, 255, 0.06)",
        showgrid=True,
    ),
    yaxis=dict(
        gridcolor="rgba(100, 200, 255, 0.06)",
        showgrid=True,
        tickformat="$,.0f",
    ),
    hovermode="x unified",
)

st.plotly_chart(fig, width='stretch')

# ── Simulation Distribution ──────────────────────────────────────
with st.expander("📊 Monte Carlo Simulation Distribution", expanded=False):
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=sim_prices,
        nbinsx=80,
        marker_color="rgba(79, 195, 247, 0.6)",
        marker_line=dict(color="rgba(79, 195, 247, 0.8)", width=0.5),
        name="Simulated Prices",
    ))
    fig_dist.add_vline(x=lower, line_dash="dash", line_color="red",
                       annotation_text=f"2.5th: ${lower:,.0f}")
    fig_dist.add_vline(x=upper, line_dash="dash", line_color="red",
                       annotation_text=f"97.5th: ${upper:,.0f}")
    fig_dist.add_vline(x=current_price, line_color="#FFD700",
                       annotation_text=f"Current: ${current_price:,.0f}")
    fig_dist.update_layout(
        xaxis_title="Simulated Next-Hour Price (USDT)",
        yaxis_title="Frequency",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#ccd6f6"),
        height=350,
        showlegend=False,
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    st.caption(f"Distribution of {N_SIMS:,} Monte Carlo simulated next-hour prices. "
               f"Skewed-t(df={DF_T}, nc={NC_SKEW}) shocks with AR(1)-GARCH(1,1) dynamic volatility.")

st.divider()

# ── Part C: Prediction History ───────────────────────────────────
st.markdown("### 📜 Prediction History (Part C — Persistence)")

if history:
    # Build display table
    hist_df = pd.DataFrame(history[-20:][::-1])  # Show last 20, newest first
    hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M UTC")

    # Add hit/miss column where actuals are available
    def check_hit(row):
        if row["actual"] is None:
            return "⏳ Pending"
        if row["lower_95"] <= row["actual"] <= row["upper_95"]:
            return "✅ Hit"
        return "❌ Miss"

    hist_df["result"] = hist_df.apply(check_hit, axis=1)

    # Format for display
    display_df = hist_df[["timestamp", "current_price", "lower_95", "upper_95", "actual", "result"]].copy()
    display_df.columns = ["Prediction Time", "Price at Prediction", "Lower 95%", "Upper 95%", "Actual (Next Hr)", "Result"]

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Summary stats for filled-in predictions
    filled = [h for h in history if h["actual"] is not None]
    if filled:
        hits = sum(1 for h in filled if h["lower_95"] <= h["actual"] <= h["upper_95"])
        live_cov = hits / len(filled)
        st.caption(f"Live coverage: {hits}/{len(filled)} = {live_cov:.1%} | "
                   f"Total predictions logged: {len(history)}")
    else:
        st.caption(f"Total predictions logged: {len(history)} | Actuals will be filled on next visit.")
else:
    st.info("No prediction history yet. Predictions are saved on each visit and accumulate over time.")

st.divider()

# ── Footer ───────────────────────────────────────────────────────
st.markdown("---")
col_f1, col_f2 = st.columns(2)
with col_f1:
    st.caption(
        f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  \n"
        f"Data source: Binance Public API (`data-api.binance.vision`)  \n"
        f"Model: Rolling AR(1)-GARCH(1,1) + Skewed Student-t(df={DF_T}, nc={NC_SKEW}) ({N_SIMS:,} sims)"
    )
with col_f2:
    st.caption(
        "**Technical Notes:**  \n"
        "• AR-GARCH for dynamic drift and volatility clustering  \n"
        "• Non-central Student-t shocks for asymmetric fat tails  \n"
        "• Strict no-look-ahead in backtest  \n"
        "• Predictions saved to `prediction_history.json`"
    )
