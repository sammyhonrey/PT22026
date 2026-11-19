#!/usr/bin/env python3
"""Validate real experiment evidence and verify/publish local release manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.artifacts import ArtifactReleaseError, capture_digests, publish_manifest, read_release
from src.datasets import DATASETS
from src.features import FEATURE_COLUMNS, build_supervised_dataset
from src.forecasting import load_model_bundle
from scripts.validate_artifacts import validate_dataset


def verify_fitted_predictions(dataset_id: str) -> None:
    """Reproduce every saved test prediction from the actual fitted bundle."""
    spec = DATASETS[dataset_id]
    if not spec.processed_path.exists():
        raise ArtifactReleaseError("Processed data is required to publish a release. Run: " + spec.prepare_command)
    hourly = pd.read_csv(spec.processed_path, parse_dates=["timestamp"]).set_index("timestamp")
    saved = pd.read_csv(spec.artifact_dir / "test_predictions.csv", parse_dates=["forecast_origin", "target_timestamp"])
    expected_models = {"Seasonal Naive", "Ridge", "HistGradientBoosting"}
    if set(saved["model"]) != expected_models:
        raise ArtifactReleaseError("Release evidence must contain all three evaluated methods.")
    origins = pd.DatetimeIndex(saved["forecast_origin"].drop_duplicates().sort_values())
    features = build_supervised_dataset(hourly, origins).sort_values(["forecast_origin", "horizon"]).reset_index(drop=True)
    bundle = load_model_bundle(spec.artifact_dir / "model_bundle.joblib", dataset_id)
    if not {"Ridge", "HistGradientBoosting"}.issubset(bundle["models"]):
        raise ArtifactReleaseError("The release bundle is missing an evaluated fitted model.")
    for name in sorted(expected_models):
        group = saved.loc[saved.model.eq(name)].sort_values(["forecast_origin", "horizon"]).reset_index(drop=True)
        for column in ["forecast_origin", "target_timestamp", "horizon"]:
            if not group[column].equals(features[column]):
                raise ArtifactReleaseError("Saved test keys do not match reconstructed test targets.")
        if not np.allclose(group.actual_kwh, features.target_kwh, rtol=1e-12, atol=1e-12):
            raise ArtifactReleaseError("Saved observed targets do not match the processed source.")
        prediction = (features.target_lag_24.to_numpy() if name == "Seasonal Naive"
                      else np.clip(bundle["models"][name].predict(features[FEATURE_COLUMNS]), 0, None))
        if not np.isfinite(prediction).all() or not np.allclose(prediction, group.predicted_kwh, rtol=1e-10, atol=1e-10):
            raise ArtifactReleaseError("Saved predictions do not match the fitted " + name + " model. Recreate the whole experiment.")
    print(f"PASS [{dataset_id}]: all {len(saved):,} saved test predictions and their source targets reproduced.")


def verify_experiments(dataset_ids, *, write_manifests: bool = False) -> None:
    for dataset_id in dataset_ids:
        spec = DATASETS[dataset_id]
        before = capture_digests(spec.artifact_dir)
        try:
            if not write_manifests:
                read_release(spec.artifact_dir, dataset_id, require_manifest=True)
            validate_dataset(dataset_id)
            if write_manifests:
                verify_fitted_predictions(dataset_id)
            if capture_digests(spec.artifact_dir) != before:
                raise ArtifactReleaseError("Runtime artifacts changed during validation. Retry on a stable experiment directory.")
            release = (publish_manifest(spec.artifact_dir, dataset_id, before) if write_manifests
                       else read_release(spec.artifact_dir, dataset_id, require_manifest=True))
        except (ArtifactReleaseError, OSError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        print(f"PASS [{dataset_id}]: verified experiment release {release.release_id}")
    print("Local experiment preflight passed. This does not verify hosting, TLS, dependency security or visual accessibility.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--write-manifests", action="store_true", help="Publish only after validating the saved experiment evidence.")
    args = parser.parse_args()
    verify_experiments(DATASETS if args.dataset == "all" else [args.dataset], write_manifests=args.write_manifests)


if __name__ == "__main__":
    main()
