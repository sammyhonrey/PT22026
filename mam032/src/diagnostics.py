"""Descriptive reliability of saved forecasts, without test-set tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import hashlib
import io
import json
from pathlib import Path

from .evaluation import metrics_by_horizon, regression_metrics
from .artifacts import verified_bytes

# The reliability files the Model Lab screen reads. They are made by
# scripts/run_diagnostics.py and must be made again after every retraining.
DISPLAY_FILES = ("test_event_metrics.csv", "test_daily_metrics.csv", "test_horizon_metrics.csv",
                 "rolling_fold_metrics.csv", "rolling_horizon_metrics.csv", "fold_manifest.json")
MODEL_NAMES = {"Seasonal Naive", "Ridge", "HistGradientBoosting"}


def _validate_display_payload(data: dict, dataset_id: str, release_id: str) -> None:
    """Checksums identify bytes; validate the structure consumed by the views too.

    The hash check only proves a file did not change. This function also
    checks that the tables have the right columns, the right models, sensible
    numbers, and that the event counts and fold dates agree with each other.
    Raises ValueError with a clear message at the first problem.
    """
    # For each table: the columns it must have.
    required = {
        "test_event_metrics.csv": {"test_days", "test_hours", "threshold_kwh", "actual_events", "predicted_events",
                                   "true_positives", "false_positives", "false_negatives", "true_negatives",
                                   "precision", "recall", "f1", "event_prevalence", "daily_total_mae_kwh",
                                   "mean_peak_magnitude_error_kwh", "mean_peak_time_error_hours"},
        "test_daily_metrics.csv": {"forecast_origin", "mae_kwh", "rmse_kwh", "actual_total_kwh", "predicted_total_kwh",
                                   "absolute_total_error_kwh", "peak_time_error_hours", "peak_magnitude_error_kwh"},
        "test_horizon_metrics.csv": {"horizon", "mae_kwh", "rmse_kwh"},
        "rolling_fold_metrics.csv": {"fold", "evaluation_days", "training_days", "mae_kwh", "rmse_kwh"},
        "rolling_horizon_metrics.csv": {"horizon", "mae_kwh", "rmse_kwh"},
    }
    for name, columns in required.items():
        frame = data[name]
        if (frame.empty or not (columns | {"model", "dataset_id", "source_release_id"}).issubset(frame.columns)
                or frame.model.isna().any() or set(frame.model) != MODEL_NAMES
                or not frame.dataset_id.eq(dataset_id).all() or not frame.source_release_id.eq(release_id).all()):
            raise ValueError("Invalid reliability table schema or experiment identity: " + name)
        numeric = columns - {"forecast_origin", "precision", "recall", "f1"}
        values = frame[sorted(numeric)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Reliability evidence contains invalid numeric values: " + name)
        for key in columns & {"horizon", "fold", "test_days", "test_hours", "actual_events", "predicted_events",
                              "true_positives", "false_positives", "false_negatives", "true_negatives",
                              "training_days", "evaluation_days", "peak_time_error_hours"}:
            if not frame[key].mod(1).eq(0).all():
                raise ValueError("Reliability counts and positions must be integers: " + name)
        keys = ["model"] + (["horizon"] if "horizon" in columns else ["fold"] if "fold" in columns
                            else ["forecast_origin"] if "forecast_origin" in columns else [])
        if frame.duplicated(keys).any():
            raise ValueError("Reliability evidence contains duplicate rows: " + name)
        if "horizon" in columns and any(sorted(group.horizon) != list(range(1, 25)) for _, group in frame.groupby("model")):
            raise ValueError("Reliability evidence must cover all 24 horizons for every model.")
    # Check the high-demand event counts. TP = predicted and real high demand,
    # FP = predicted but not real, FN = real but missed, TN = both normal.
    summary = data["test_event_metrics.csv"]
    for row in summary.itertuples():
        tp, fp, fn, tn = row.true_positives, row.false_positives, row.false_negatives, row.true_negatives
        if (row.test_days < 1 or row.test_hours != row.test_days * 24 or tp + fp + fn + tn != row.test_hours
                or row.actual_events != tp + fn or row.predicted_events != tp + fp):
            raise ValueError("Reliability event totals do not agree.")
        for key, numerator, denominator in [("precision", tp, tp + fp), ("recall", tp, tp + fn),
                                             ("f1", 2 * tp, 2 * tp + fp + fn),
                                             ("event_prevalence", tp + fn, row.test_hours)]:
            expected = numerator / denominator if denominator else np.nan
            if not np.isclose(getattr(row, key), expected, equal_nan=True):
                raise ValueError("Reliability ratios do not match the recorded counts.")
    if any(summary[column].nunique() != 1 for column in ["test_days", "threshold_kwh", "actual_events"]):
        raise ValueError("Reliability models must describe identical source test targets and thresholds.")
    daily = data["test_daily_metrics.csv"]
    daily["forecast_origin"] = pd.to_datetime(daily.forecast_origin, errors="raise")
    reference = None
    for model, group in daily.groupby("model"):
        origins = group.forecast_origin.sort_values().reset_index(drop=True)
        if origins.isna().any() or origins.duplicated().any() or len(origins) != int(summary.loc[summary.model.eq(model), "test_days"].iloc[0]):
            raise ValueError("Reliability daily evidence does not match its source test-day count.")
        if reference is not None and not origins.equals(reference):
            raise ValueError("Reliability daily targets differ between models.")
        reference = origins
    # Check the fold dates: training must end before testing starts, and
    # testing must end before the reserved test period begins.
    fold = data["fold_manifest.json"]
    fields = {"dataset_id", "development_origins", "reserved_test_first_target", "folds"}
    if (not isinstance(fold, dict) or not fields.issubset(fold) or fold["dataset_id"] != dataset_id
            or type(fold["development_origins"]) is not int or fold["development_origins"] < 1
            or not isinstance(fold["folds"], list) or len(fold["folds"]) < 2):
        raise ValueError("Reliability fold metadata is invalid.")
    reserved = pd.Timestamp(fold["reserved_test_first_target"])
    if pd.isna(reserved):
        raise ValueError("Reliability reserved-test boundary is invalid.")
    rolling = data["rolling_fold_metrics.csv"]
    fold_fields = {"fold", "train_origins", "evaluation_origins", "last_training_target", "first_evaluation_target", "last_evaluation_target"}
    for number, item in enumerate(fold["folds"], 1):
        if not isinstance(item, dict) or not fold_fields.issubset(item) or item["fold"] != number:
            raise ValueError("Reliability fold metadata is incomplete or out of order.")
        for key in ["train_origins", "evaluation_origins"]:
            if not isinstance(item[key], list) or not item[key]:
                raise ValueError("Reliability fold origins must be nonempty lists.")
            times = pd.DatetimeIndex(pd.to_datetime(item[key], errors="raise"))
            if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
                raise ValueError("Reliability fold origins must be valid, unique and ascending.")
        last_train, first_target, last_target = [pd.Timestamp(item[key]) for key in
                                                ["last_training_target", "first_evaluation_target", "last_evaluation_target"]]
        if (any(pd.isna(t) for t in [last_train, first_target, last_target])
                or not last_train < first_target <= last_target < reserved):
            raise ValueError("Reliability folds overlap future evaluation or reserved-test targets.")
        if (last_train != pd.Timestamp(item["train_origins"][-1]) + pd.Timedelta(hours=24)
                or first_target != pd.Timestamp(item["evaluation_origins"][0]) + pd.Timedelta(hours=1)
                or last_target != pd.Timestamp(item["evaluation_origins"][-1]) + pd.Timedelta(hours=24)):
            raise ValueError("Reliability target boundaries do not match their 24-hour forecast origins.")
        rows = rolling.loc[rolling.fold.eq(number)]
        if (len(rows) != 3 or set(rows.model) != MODEL_NAMES
                or not rows.training_days.eq(len(item["train_origins"])).all()
                or not rows.evaluation_days.eq(len(item["evaluation_origins"])).all()):
            raise ValueError("Reliability fold table and metadata do not agree.")
    if set(rolling.fold) != set(range(1, len(fold["folds"]) + 1)):
        raise ValueError("Reliability fold table has missing or unexpected folds.")


def write_diagnostics(folder: Path, dataset_id: str, release_id: str, frames: dict[str, pd.DataFrame],
                      fold_manifest: dict, provenance: dict) -> dict:
    """Save the reliability tables and a manifest with their SHA-256 hashes.

    Returns the manifest. The app later checks the hashes, so it can tell if a
    file was changed or belongs to a different model release.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in DISPLAY_FILES:
        content = ((json.dumps(fold_manifest, indent=2) + "\n").encode() if name.endswith(".json")
                   else frames[name].assign(dataset_id=dataset_id, source_release_id=release_id).to_csv(index=False).encode())
        (folder / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    manifest = {"schema_version": 1, "dataset_id": dataset_id, "source_release_id": release_id,
                "files": hashes, "provenance": provenance}
    # Publish this last. Readers never accept a partially replaced file set.
    temporary = folder / "diagnostics_manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(folder / "diagnostics_manifest.json")
    return manifest


def load_diagnostics(folder: Path, dataset_id: str, release_id: str) -> dict:
    """Read the reliability files, but only if they match this dataset and release.

    Raises ValueError if the manifest does not match, and an error from
    verified_bytes if a file hash is wrong. Returns a dict of the tables.
    """
    folder = Path(folder)
    manifest = json.loads((folder / "diagnostics_manifest.json").read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("dataset_id") != dataset_id
            or manifest.get("source_release_id") != release_id or not isinstance(manifest.get("files"), dict)
            or set(manifest["files"]) != set(DISPLAY_FILES)):
        raise ValueError("Reliability diagnostics do not match this experiment release. Regenerate them.")
    result = {"manifest": manifest}
    for name in DISPLAY_FILES:
        content = verified_bytes(folder / name, manifest["files"][name])
        result[name] = json.loads(content) if name.endswith(".json") else pd.read_csv(io.BytesIO(content))
    _validate_display_payload(result, dataset_id, release_id)
    return result


def prediction_diagnostics(predictions: pd.DataFrame, threshold: float,
                           expected_metrics: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Evaluate a fixed training-derived threshold on aligned 24-hour blocks.

    This describes the supplied recorded-source evidence. It does not choose a
    threshold, fit an alert classifier or validate an uploaded household.

    A "high-demand hour" is an hour with at least `threshold` kWh. For each
    model we count how many of these hours were found or missed, and how well
    the daily total and the daily peak were predicted. Returns three tables:
    summary (one row per model), daily and horizon.
    """
    required = {"model", "forecast_origin", "target_timestamp", "horizon", "actual_kwh", "predicted_kwh"}
    if not required.issubset(predictions) or predictions.empty:
        raise ValueError("Prediction evidence is empty or missing required columns.")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Demand threshold must be finite and non-negative.")
    data = predictions.copy()
    for name in ["forecast_origin", "target_timestamp"]:
        data[name] = pd.to_datetime(data[name], errors="raise")
        if data[name].isna().any():
            raise ValueError("Prediction timestamps must be valid.")
    values = data[["actual_kwh", "predicted_kwh"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Prediction evidence must contain finite non-negative energy.")
    if data["model"].isna().any() or not set(data.model).issubset({"Seasonal Naive", "Ridge", "HistGradientBoosting"}):
        raise ValueError("Prediction evidence has an unknown model identity.")
    if data.duplicated(["model", "forecast_origin", "horizon"]).any():
        raise ValueError("Prediction evidence contains duplicate horizons.")
    if expected_metrics is not None and set(data.model) != set(expected_metrics.model):
        raise ValueError("Saved predictions and benchmark metrics contain different models.")
    summaries, days = [], []
    reference = None
    for name, group in data.groupby("model", sort=True):
        group = group.sort_values(["forecast_origin", "horizon"]).reset_index(drop=True)
        keys = group[["forecast_origin", "target_timestamp", "horizon", "actual_kwh"]]
        if reference is None:
            reference = keys
        elif not keys.equals(reference):
            raise ValueError("Models must use identical recorded target blocks.")
        score = regression_metrics(group.actual_kwh, group.predicted_kwh)
        if expected_metrics is not None:
            expected = expected_metrics.loc[expected_metrics.model.eq(name)]
            if len(expected) != 1 or any(not np.isclose(score[key], expected.iloc[0][key], rtol=1e-8, atol=1e-10) for key in score):
                raise ValueError("Prediction evidence does not match the saved benchmark scores.")
        # True where the real (actual) or predicted energy reaches high demand.
        actual = group.actual_kwh.to_numpy() >= threshold
        predicted = group.predicted_kwh.to_numpy() >= threshold
        tp = int((actual & predicted).sum())
        fp = int((~actual & predicted).sum())
        fn = int((actual & ~predicted).sum())
        tn = int((~actual & ~predicted).sum())
        model_days = []
        for origin, day in group.groupby("forecast_origin", sort=True):
            if day.horizon.tolist() != list(range(1, 25)):
                raise ValueError("Every forecast origin must contain exactly horizons 1–24.")
            expected_times = pd.DatetimeIndex([origin + pd.Timedelta(hours=h) for h in range(1, 25)])
            if not pd.DatetimeIndex(day.target_timestamp).equals(expected_times):
                raise ValueError("Forecast targets must align to their origin and horizon.")
            # The peak is the hour with the highest energy in the day.
            observed_peak, predicted_peak = int(np.argmax(day.actual_kwh)), int(np.argmax(day.predicted_kwh))
            row = {"model": name, "forecast_origin": origin, **regression_metrics(day.actual_kwh, day.predicted_kwh),
                   "actual_total_kwh": float(day.actual_kwh.sum()), "predicted_total_kwh": float(day.predicted_kwh.sum()),
                   "absolute_total_error_kwh": float(abs(day.predicted_kwh.sum() - day.actual_kwh.sum())),
                   "peak_time_error_hours": abs(predicted_peak - observed_peak),
                   "peak_magnitude_error_kwh": float(abs(day.predicted_kwh.max() - day.actual_kwh.max()))}
            days.append(row)
            model_days.append(row)
        summaries.append({"model": name, "test_days": len(model_days), "test_hours": len(group),
                          "threshold_kwh": threshold, "actual_events": tp + fn, "predicted_events": tp + fp,
                          "true_positives": tp, "false_positives": fp, "false_negatives": fn, "true_negatives": tn,
                          "precision": tp / (tp + fp) if tp + fp else np.nan,
                          "recall": tp / (tp + fn) if tp + fn else np.nan,
                          "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
                          "event_prevalence": (tp + fn) / len(group),
                          "mean_peak_time_error_hours": float(np.mean([r["peak_time_error_hours"] for r in model_days])),
                          "daily_total_mae_kwh": float(np.mean([r["absolute_total_error_kwh"] for r in model_days])),
                          "mean_peak_magnitude_error_kwh": float(np.mean([r["peak_magnitude_error_kwh"] for r in model_days]))})
    return {"summary": pd.DataFrame(summaries), "daily": pd.DataFrame(days), "horizon": metrics_by_horizon(data)}
