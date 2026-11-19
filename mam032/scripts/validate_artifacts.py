#!/usr/bin/env python3
"""Fail fast when a demo artifact is missing, inconsistent, or unusable."""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import validate_hourly_csv
from src.evaluation import regression_metrics
from src.forecasting import forecast_24h, load_model_bundle


def validate_dataset(dataset_id: str) -> None:
    from src.datasets import get_dataset
    spec = get_dataset(dataset_id)
    artifact_dir = spec.artifact_dir
    required = [
        artifact_dir / "model_bundle.joblib",
        artifact_dir / "test_metrics.csv",
        artifact_dir / "test_predictions.csv",
        artifact_dir / "permutation_importance.csv",
        artifact_dir / "data_audit.json",
        spec.demo_path,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing required artifacts: " + ", ".join(missing))
    bundle = load_model_bundle(required[0], expected_dataset_id=dataset_id)
    metadata = json.loads((artifact_dir / "training_metadata.json").read_text())
    if metadata.get("dataset_id", "uci_household") != dataset_id:
        raise SystemExit("Training metadata belongs to a different dataset.")
    demo = validate_hourly_csv(required[-1], min_history=169)
    valid_origins = demo.index[168:-24]
    if (valid_origins.hour == 23).any():
        valid_origins = valid_origins[valid_origins.hour == 23]
    origin = valid_origins[-1]
    result = forecast_24h(demo, origin, bundle)
    if len(result) != 24 or result["predicted_kwh"].isna().any():
        raise SystemExit("Forecast artifact validation failed.")
    audit = json.loads((artifact_dir / "data_audit.json").read_text())
    print(f"PASS [{dataset_id}]: 24 aligned predictions generated from origin {origin}.")
    print(f"PASS: source audit contains {audit['source_rows']:,} source readings.")
    predictions = pd.read_csv(required[2], parse_dates=["forecast_origin", "target_timestamp"])
    scores = pd.read_csv(required[1]).set_index("model")
    comparison_keys = None
    for name, group in predictions.groupby("model"):
        group = group.sort_values(["forecast_origin", "horizon"])
        keys = group[["forecast_origin", "target_timestamp", "horizon", "actual_kwh"]].reset_index(drop=True)
        if comparison_keys is None:
            comparison_keys = keys
        elif not comparison_keys.equals(keys):
            raise SystemExit("Models were not evaluated on identical targets.")
        if not (group.groupby("forecast_origin").size() == 24).all():
            raise SystemExit("A test origin does not contain all 24 horizons.")
        if not (group["forecast_origin"] >= pd.Timestamp(bundle["trained_through"])).all():
            raise SystemExit("Test origins overlap the fitted model's training targets.")
        recomputed = regression_metrics(group["actual_kwh"], group["predicted_kwh"])
        for metric, value in recomputed.items():
            if not np.isclose(value, scores.loc[name, metric], atol=1e-10):
                raise SystemExit(f"Saved {name} {metric} does not match the prediction evidence.")
    print("PASS: every saved test score matches recomputed metrics on identical 24-h target blocks.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["uci_household", "xky_2025", "all"], default="uci_household")
    args = parser.parse_args()
    for dataset_id in (["uci_household", "xky_2025"] if args.dataset == "all" else [args.dataset]):
        validate_dataset(dataset_id)


if __name__ == "__main__":
    main()
