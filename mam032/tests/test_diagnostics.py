"""Reliability evidence must stay aligned, reproducible and release-bound."""

import json

import numpy as np
import pandas as pd
import pytest

from src.diagnostics import DISPLAY_FILES, load_diagnostics, prediction_diagnostics, write_diagnostics
from src.evaluation import regression_metrics


def recorded_predictions():
    origin = pd.Timestamp("2025-01-01 23:00")
    actual = np.array([3, 3] + [1] * 22, dtype=float)
    predicted = np.array([3, 1, 3] + [1] * 21, dtype=float)
    first = pd.DataFrame({"model": "HistGradientBoosting", "forecast_origin": origin,
                         "target_timestamp": pd.date_range(origin + pd.Timedelta(hours=1), periods=24, freq="h"),
                         "horizon": range(1, 25), "actual_kwh": actual, "predicted_kwh": predicted})
    second = first.copy()
    for column in ["forecast_origin", "target_timestamp"]:
        second[column] += pd.Timedelta(days=1)
    return pd.concat([first, second], ignore_index=True)


def test_event_counts_and_peak_ties_use_recorded_values():
    result = prediction_diagnostics(recorded_predictions(), 2)
    row = result["summary"].iloc[0]
    assert (row.true_positives, row.false_positives, row.false_negatives, row.true_negatives) == (2, 2, 2, 42)
    assert row.precision == row.recall == row.f1 == .5
    assert row.mean_peak_time_error_hours == 0  # Earliest tied maximum.
    assert row.daily_total_mae_kwh == 0
    assert len(result["horizon"]) == 24


def test_undefined_event_rates_remain_missing():
    row = prediction_diagnostics(recorded_predictions(), 9)["summary"].iloc[0]
    assert np.isnan(row.precision) and np.isnan(row.recall) and np.isnan(row.f1)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "clock", "nan", "negative", "identity", "misaligned_models"])
def test_invalid_prediction_evidence_is_rejected(fault):
    data = recorded_predictions()
    if fault == "missing":
        data = data.iloc[:-1]
    elif fault == "duplicate":
        data = pd.concat([data, data.iloc[:1]])
    elif fault == "clock":
        data.loc[0, "target_timestamp"] += pd.Timedelta(hours=1)
    elif fault == "nan":
        data.loc[0, "predicted_kwh"] = np.nan
    elif fault == "negative":
        data.loc[0, "actual_kwh"] = -1
    elif fault == "identity":
        data["model"] = "Unknown"
    else:
        other = data.assign(model="Ridge")
        other.loc[0, "actual_kwh"] += .1
        data = pd.concat([data, other])
    with pytest.raises(ValueError):
        prediction_diagnostics(data, 2)


def test_saved_benchmark_scores_must_reproduce():
    data = recorded_predictions()
    scores = pd.DataFrame([{"model": "HistGradientBoosting", **regression_metrics(data.actual_kwh, data.predicted_kwh)}])
    prediction_diagnostics(data, 2, scores)
    scores.loc[0, "mae_kwh"] += .1
    with pytest.raises(ValueError, match="benchmark scores"):
        prediction_diagnostics(data, 2, scores)


def test_display_artifacts_are_hashed_and_bound_to_dataset_and_release(tmp_path, diagnostic_payload):
    frames, folds = diagnostic_payload
    write_diagnostics(tmp_path, "test-house", "release-A", frames, folds, {})
    result = load_diagnostics(tmp_path, "test-house", "release-A")
    assert result["test_event_metrics.csv"].source_release_id.eq("release-A").all()
    for dataset, release in [("other-house", "release-A"), ("test-house", "release-B")]:
        with pytest.raises(ValueError):
            load_diagnostics(tmp_path, dataset, release)
    path = tmp_path / "test_event_metrics.csv"
    path.write_text(path.read_text().replace(",1,", ",2,") + "\n")
    with pytest.raises(ValueError):
        load_diagnostics(tmp_path, "test-house", "release-A")


@pytest.mark.parametrize("fault", ["column", "missing_model", "duplicate_model", "negative_count", "fractional_count",
                                 "bad_ratio", "bad_total", "missing_horizon", "extra_fold", "wrong_day_count",
                                 "fold_type", "empty_origins", "future_training", "reserved_overlap", "identity"])
def test_hash_valid_but_malformed_diagnostics_are_rejected(tmp_path, diagnostic_payload, fault):
    frames, folds = diagnostic_payload
    summary = frames["test_event_metrics.csv"]
    if fault == "column":
        frames["test_event_metrics.csv"] = summary.drop(columns="precision")
    elif fault == "missing_model":
        frames["test_event_metrics.csv"] = summary.iloc[:-1]
    elif fault == "duplicate_model":
        frames["test_event_metrics.csv"] = pd.concat([summary, summary.iloc[:1]])
    elif fault in ["negative_count", "fractional_count"]:
        summary["false_positives"] = -1 if fault == "negative_count" else .5
    elif fault == "bad_ratio":
        summary["precision"] = .99
    elif fault == "bad_total":
        summary["actual_events"] += 1
    elif fault == "missing_horizon":
        frames["test_horizon_metrics.csv"] = frames["test_horizon_metrics.csv"].iloc[:-1]
    elif fault == "extra_fold":
        frames["rolling_fold_metrics.csv"].loc[0, "fold"] = 9
    elif fault == "wrong_day_count":
        frames["rolling_fold_metrics.csv"]["training_days"] = 999
    elif fault == "fold_type":
        folds = []
    elif fault == "empty_origins":
        folds["folds"][0]["train_origins"] = []
    elif fault == "future_training":
        folds["folds"][0]["last_training_target"] = "2026-01-01"
    elif fault == "reserved_overlap":
        folds["folds"][0]["last_evaluation_target"] = folds["reserved_test_first_target"]
    else:
        folds["dataset_id"] = "other-house"
    # Rebuild every checksum, simulating a bad regeneration rather than byte corruption.
    write_diagnostics(tmp_path, "test-house", "release-A", frames, folds, {})
    with pytest.raises(ValueError):
        load_diagnostics(tmp_path, "test-house", "release-A")


@pytest.mark.parametrize("manifest", [[], None, {"schema_version": 1, "dataset_id": "x", "source_release_id": "r", "files": []}])
def test_malformed_diagnostics_manifest_is_recoverable(tmp_path, manifest):
    (tmp_path / "diagnostics_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_diagnostics(tmp_path, "x", "r")
