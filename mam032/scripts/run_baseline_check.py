#!/usr/bin/env python3
"""Compare the trained models with a simple 7-day same-hour median rule.

The rule needs no training: for each hour it takes the median of the same hour
on the last 7 days. We score it on exactly the same test days as the models and
save a small table, baseline_comparison.csv, for the Model Lab screen.
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import get_dataset
from src.uncertainty import seasonal_median_forecast


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["uci_household", "xky_2025"], default="uci_household")
    args = parser.parse_args()
    spec = get_dataset(args.dataset)
    hourly = pd.read_csv(spec.processed_path, parse_dates=["timestamp"]).set_index("timestamp")["energy_kwh"]
    test = pd.read_csv(spec.artifact_dir / "test_predictions.csv", parse_dates=["forecast_origin", "target_timestamp"])

    # Step 1: score the three trained models from the saved test predictions.
    rows = []
    for model, group in test.groupby("model"):
        error = group["actual_kwh"] - group["predicted_kwh"]
        rows.append({"model": model, "mae_kwh": error.abs().mean(), "rmse_kwh": float(np.sqrt((error ** 2).mean()))})

    # Step 2: build the simple baseline for the same test origins.
    pieces = []
    for origin in sorted(test["forecast_origin"].unique()):
        forecast = seasonal_median_forecast(hourly, pd.Timestamp(origin))
        forecast["actual_kwh"] = hourly.reindex(pd.DatetimeIndex(forecast["target_timestamp"])).to_numpy()
        pieces.append(forecast)
    baseline = pd.concat(pieces).dropna()
    error = baseline["actual_kwh"] - baseline["baseline_kwh"]
    rows.append({"model": "7-day same-hour median", "mae_kwh": error.abs().mean(),
                 "rmse_kwh": float(np.sqrt((error ** 2).mean()))})

    # Step 3: add easy-to-read columns and save.
    table = pd.DataFrame(rows)
    naive_mae = float(table.loc[table["model"] == "Seasonal Naive", "mae_kwh"].iloc[0])
    mean_actual = float(test["actual_kwh"].mean())
    table["mae_vs_naive"] = table["mae_kwh"] / naive_mae          # below 1.0 means better than Seasonal Naive
    table["mae_percent_of_mean"] = 100 * table["mae_kwh"] / mean_actual
    table = table.sort_values("mae_kwh").round(4)
    out = spec.artifact_dir / "baseline_comparison.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print("saved", out)


if __name__ == "__main__":
    main()
