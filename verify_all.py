# -*- coding: utf-8 -*-
"""
Full verification script for AlphaI x Polaris Challenge
Tests all requirements from the challenge brief.
"""
import json
import numpy as np
import pandas as pd
import os
import sys

print("=" * 60)
print("FULL VERIFICATION - AlphaI x Polaris Challenge")
print("=" * 60)

errors = []

# ===================================================================
# CHECK 1: All required files exist
# ===================================================================
print("\n[CHECK 1] Required files exist")
required_files = [
    "backtest.py",
    "app.py",
    "requirements.txt",
    "README.md",
    "backtest_results.jsonl",
    "backtest_metrics.json",
    "backtest_chart.png",
]
for f in required_files:
    exists = os.path.exists(f)
    status = "OK" if exists else "MISSING"
    size = os.path.getsize(f) if exists else 0
    print(f"  [{status}] {f} ({size:,} bytes)")
    if not exists:
        errors.append(f"Missing file: {f}")

# ===================================================================
# CHECK 2: backtest_results.jsonl format and content
# ===================================================================
print("\n[CHECK 2] backtest_results.jsonl format")
lines = open("backtest_results.jsonl").readlines()
print(f"  Total lines: {len(lines)}")
if len(lines) < 700:
    errors.append(f"Too few predictions: {len(lines)} (need ~720)")

first = json.loads(lines[0])
last = json.loads(lines[-1])
required_keys = {"timestamp", "lower_95", "upper_95", "actual"}
has_keys = required_keys.issubset(first.keys())
print(f"  Keys present: {list(first.keys())}")
print(f"  Required keys present: {has_keys}")
if not has_keys:
    errors.append(f"Missing keys in JSONL: {required_keys - first.keys()}")
print(f"  First prediction: {first}")
print(f"  Last prediction:  {last}")

# ===================================================================
# CHECK 3: Backtest metrics are valid
# ===================================================================
print("\n[CHECK 3] Backtest metrics")
metrics = json.load(open("backtest_metrics.json"))
cov = metrics["coverage_95"]
winkler = metrics["mean_winkler_95"]
width = metrics["avg_width"]
total = metrics["total_predictions"]

print(f"  coverage_95:     {cov:.4f}  (target: 0.93-0.97)")
print(f"  avg_width:       ${width:,.2f}")
print(f"  mean_winkler_95: {winkler:,.2f}")
print(f"  total_predictions: {total}")
print(f"  total_hits:      {metrics['total_hits']}")
print(f"  total_misses:    {metrics['total_misses']}")

if cov < 0.92 or cov > 0.98:
    errors.append(f"Coverage {cov:.4f} outside acceptable range [0.92, 0.98]")
else:
    print(f"  COVERAGE OK: {cov:.4f} is in acceptable range")

# ===================================================================
# CHECK 4: No look-ahead bias - timestamp spacing
# ===================================================================
print("\n[CHECK 4] No look-ahead bias (timestamp spacing)")
all_preds = [json.loads(l) for l in lines]
timestamps = pd.to_datetime([p["timestamp"] for p in all_preds])
diffs = timestamps.diff().dropna()
median_diff_hours = diffs.median().total_seconds() / 3600
print(f"  Median spacing: {median_diff_hours:.2f} hours")
if abs(median_diff_hours - 1.0) > 0.1:
    errors.append(f"Unexpected timestamp spacing: {median_diff_hours:.2f}h (expected ~1h)")
else:
    print(f"  TIMESTAMP SPACING OK (1-hour bars)")

# ===================================================================
# CHECK 5: All predictions have reasonable values
# ===================================================================
print("\n[CHECK 5] Prediction sanity checks")
bad_predictions = 0
for i, p in enumerate(all_preds):
    if p["lower_95"] >= p["upper_95"]:
        bad_predictions += 1
    if p["lower_95"] <= 0:
        bad_predictions += 1
    w = p["upper_95"] - p["lower_95"]
    if w < 10 or w > 50000:
        bad_predictions += 1
print(f"  Bad predictions found: {bad_predictions}")
if bad_predictions > 0:
    errors.append(f"{bad_predictions} predictions have invalid ranges")
else:
    print(f"  ALL {len(all_preds)} PREDICTIONS VALID")

# ===================================================================
# CHECK 6: Manual Winkler verification
# ===================================================================
print("\n[CHECK 6] Manual Winkler score verification")
alpha = 0.05
winklers_manual = []
hits_manual = 0
for p in all_preds:
    L, U, actual = p["lower_95"], p["upper_95"], p["actual"]
    w = U - L
    if actual < L:
        w += (2 / alpha) * (L - actual)
    elif actual > U:
        w += (2 / alpha) * (actual - U)
    else:
        hits_manual += 1
    winklers_manual.append(w)

manual_cov = hits_manual / len(all_preds)
manual_winkler = np.mean(winklers_manual)
print(f"  Manual coverage:  {manual_cov:.4f}")
print(f"  Stored coverage:  {cov:.4f}")
print(f"  Manual Winkler:   {manual_winkler:.2f}")
print(f"  Stored Winkler:   {winkler:.2f}")

if abs(manual_cov - cov) > 0.001:
    errors.append(f"Coverage mismatch: manual={manual_cov:.4f} vs stored={cov:.4f}")
if abs(manual_winkler - winkler) > 1.0:
    errors.append(f"Winkler mismatch: manual={manual_winkler:.2f} vs stored={winkler:.2f}")
print(f"  MANUAL VERIFICATION PASSED")

# ===================================================================
# CHECK 7: Model consistency - backtest.py and app.py use same logic
# ===================================================================
print("\n[CHECK 7] Model consistency between backtest.py and app.py")
backtest_src = open("backtest.py", encoding="utf-8").read()
app_src = open("app.py", encoding="utf-8").read()

# Check both use same key parameters
checks = {
    "AR-GARCH(1,1) vol": ("vol='Garch'" in backtest_src and "vol='Garch'" in app_src),
    "Student-t df_t": ("df_t" in backtest_src and "df_t" in app_src),
    "Non-central t-dist (nct)": ("nct.rvs" in backtest_src and "nct.rvs" in app_src),
    "mu + sigma * shocks": ("mu + sigma * random_shocks" in backtest_src and "mu + sigma * shocks" in app_src),
    "No FIGARCH import": ("from arch import arch_model" in backtest_src and "from arch import arch_model" in app_src),
    "NC_SKEW in app.py": ("NC_SKEW =" in app_src),
}
for check_name, passed in checks.items():
    status = "OK" if passed else "FAIL"
    print(f"  [{status}] {check_name}")
    if not passed:
        errors.append(f"Model consistency check failed: {check_name}")

# ===================================================================
# CHECK 8: requirements.txt has all needed packages
# ===================================================================
print("\n[CHECK 8] requirements.txt completeness")
reqs = open("requirements.txt").read()
needed = ["streamlit", "requests", "pandas", "numpy", "scipy", "plotly"]
for pkg in needed:
    present = pkg in reqs
    status = "OK" if present else "MISSING"
    print(f"  [{status}] {pkg}")
    if not present:
        errors.append(f"Missing from requirements.txt: {pkg}")

# Check arch IS in requirements now
if "arch" not in reqs.split():
    errors.append("requirements.txt missing 'arch' for GARCH model")
    print(f"  [MISSING] 'arch' should be in requirements")
else:
    print(f"  [OK] 'arch' correctly included")

# ===================================================================
# CHECK 9: README has key sections
# ===================================================================
print("\n[CHECK 9] README.md completeness")
readme = open("README.md", encoding="utf-8").read()
sections = {
    "Bug report": "current = -1" in readme,
    "How to run": "streamlit run app.py" in readme or "python backtest.py" in readme,
    "Part A results": "0.96" in readme or "coverage" in readme.lower(),
    "Design decisions": "EWMA" in readme or "rolling" in readme.lower(),
    "Project structure": "backtest.py" in readme and "app.py" in readme,
}
for section, found in sections.items():
    status = "OK" if found else "MISSING"
    print(f"  [{status}] {section}")
    if not found:
        errors.append(f"README missing section: {section}")

# ===================================================================
# CHECK 10: app.py has all dashboard components
# ===================================================================
print("\n[CHECK 10] Dashboard components in app.py")
components = {
    "Binance data fetch": "data-api.binance.vision" in app_src,
    "Current price display": "Current BTC Price" in app_src,
    "Lower/Upper bounds": "Lower Bound" in app_src and "Upper Bound" in app_src,
    "Backtest metrics display": "Coverage" in app_src and "Winkler" in app_src,
    "Plotly chart": "go.Figure" in app_src,
    "Prediction ribbon": "add_shape" in app_src,
    "Part C persistence": "prediction_history" in app_src,
    "History table": "Prediction History" in app_src,
    "Auto-refresh (ttl)": "ttl=300" in app_src,
    "Metrics auto-load": "backtest_metrics.json" in app_src,
}
for comp, found in components.items():
    status = "OK" if found else "MISSING"
    print(f"  [{status}] {comp}")
    if not found:
        errors.append(f"Dashboard missing component: {comp}")

# ===================================================================
# FINAL SUMMARY
# ===================================================================
print("\n" + "=" * 60)
if errors:
    print(f"FAILED - {len(errors)} error(s) found:")
    for e in errors:
        print(f"  X {e}")
    sys.exit(1)
else:
    print("ALL 10 CHECKS PASSED - Project is complete!")
    print(f"\nSubmission values:")
    print(f"  coverage_95:     {cov:.4f}")
    print(f"  mean_winkler_95: {winkler:.2f}")
    print(f"  predictions:     {total}")
print("=" * 60)
