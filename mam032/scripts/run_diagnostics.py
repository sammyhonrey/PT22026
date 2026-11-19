#!/usr/bin/env python3
"""Compute fixed test diagnostics and development-only temporal folds separately."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.artifacts import capture_digests, read_release, verified_bytes, file_digest
from src.datasets import DATASETS
from src.diagnostics import prediction_diagnostics, write_diagnostics
from src.forecasting import load_model_bundle
from src.robustness import run_development_diagnostics


def run(dataset_id: str, protocol: dict, protocol_sha: str) -> dict:
    spec = DATASETS[dataset_id]
    release = read_release(spec.artifact_dir, dataset_id, require_manifest=True)
    before = capture_digests(spec.artifact_dir)
    metadata = json.loads(verified_bytes(spec.artifact_dir / "training_metadata.json", before["training_metadata.json"]))
    bundle = load_model_bundle(spec.artifact_dir / "model_bundle.joblib", dataset_id, before["model_bundle.joblib"])
    evidence_path = spec.artifact_dir / "test_predictions.csv"
    evidence_sha = file_digest(evidence_path)
    saved = pd.read_csv(io.BytesIO(verified_bytes(evidence_path, evidence_sha)))
    scores = pd.read_csv(io.BytesIO(verified_bytes(spec.artifact_dir / "test_metrics.csv", before["test_metrics.csv"])))
    descriptive = prediction_diagnostics(saved, float(bundle["peak_threshold_kwh"]), scores)
    hourly = pd.read_csv(spec.processed_path, parse_dates=["timestamp"]).set_index("timestamp")
    with threadpool_limits(limits=1):
        rolling = run_development_diagnostics(hourly, metadata, protocol)
    reserved_targets = set(pd.to_datetime(saved.target_timestamp))
    if reserved_targets.intersection(rolling["predictions"].target_timestamp):
        raise ValueError("Rolling diagnostics overlap reserved test targets.")
    if capture_digests(spec.artifact_dir) != before or file_digest(evidence_path) != evidence_sha:
        raise ValueError("Published experiment files changed during diagnostics; no output was published.")
    folder = spec.artifact_dir / "diagnostics"
    folder.mkdir(parents=True, exist_ok=True)
    raw_path = folder / "rolling_predictions.csv"
    rolling["predictions"].to_csv(raw_path, index=False)
    fold_manifest = {"dataset_id": dataset_id, "purpose": protocol["purpose"], "protocol": protocol,
                     "development_cutoff": rolling["development_cutoff"], "development_origins": rolling["development_origins"],
                     "development_fingerprint": rolling["development_fingerprint"],
                     "reserved_test_first_target": str(pd.to_datetime(saved.target_timestamp).min()),
                     "folds": rolling["folds"], "published_model_replaced": False, "parameter_search_performed": False}
    frames = {"test_event_metrics.csv": descriptive["summary"], "test_daily_metrics.csv": descriptive["daily"],
              "test_horizon_metrics.csv": descriptive["horizon"], "rolling_fold_metrics.csv": rolling["fold_metrics"],
              "rolling_horizon_metrics.csv": rolling["horizon_metrics"]}
    manifest = write_diagnostics(folder, dataset_id, release.release_id, frames, fold_manifest,
                                 {"protocol_sha256": protocol_sha, "test_prediction_sha256": evidence_sha,
                                  "rolling_prediction_sha256": file_digest(raw_path), "rolling_prediction_rows": len(rolling["predictions"]),
                                  "rolling_predictions_git_policy": "generated locally; regenerate from processed data",
                                  "scope": "Fixed-threshold test descriptions; refitted folds use development data only. No benchmark or production-model update."})
    print(f"PASS [{dataset_id}]: {rolling['development_origins']} development origins, {len(rolling['folds'])} expanding folds; reserved test targets excluded.")
    print(descriptive["summary"][["model", "true_positives", "false_positives", "false_negatives", "precision", "recall"]].to_string(index=False))
    print(rolling["fold_metrics"].to_string(index=False))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    args = parser.parse_args()
    content = (ROOT / "config/robustness_protocol.json").read_bytes()
    protocol = json.loads(content)
    for dataset_id in DATASETS if args.dataset == "all" else [args.dataset]:
        run(dataset_id, protocol, hashlib.sha256(content).hexdigest())


if __name__ == "__main__":
    main()
