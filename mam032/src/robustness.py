"""Fixed-parameter expanding-origin diagnostics that exclude the reserved test."""

from __future__ import annotations

import hashlib

import pandas as pd
import numpy as np

from .evaluation import regression_metrics, metrics_by_horizon
from .features import build_supervised_dataset, FEATURE_COLUMNS
from .training import _models, _predict_frame


def development_data(hourly: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Exclude reserved targets before building features, fitting or summaries."""
    validation_end = pd.Timestamp(metadata["boundaries"]["validation_end"])
    test_origin = pd.Timestamp(metadata["boundaries"]["test_start"])
    cutoff = validation_end + pd.Timedelta(hours=24)
    if cutoff > test_origin:
        raise ValueError("Development target cutoff overlaps the reserved test period.")
    if not isinstance(hourly.index, pd.DatetimeIndex) or hourly.index.has_duplicates or not hourly.index.is_monotonic_increasing:
        raise ValueError("Processed source must have a unique ascending DatetimeIndex.")
    development = hourly.loc[hourly.index <= cutoff, ["energy_kwh"]].copy()
    if development.empty:
        raise ValueError("No development observations precede the test boundary.")
    return development, cutoff


def expanding_folds(dataset: pd.DataFrame, fold_count: int = 3, initial_fraction: float = 0.5) -> list[dict]:
    """Make time-ordered folds where the training period keeps growing.

    The first part of the days (initial_fraction) is only used for training.
    The remaining days are cut into fold_count blocks, in time order.
    Each fold tests on one block and trains on the days before it.
    Returns a list of dicts with the fold number, training days and test days.
    """
    if isinstance(fold_count, bool) or not isinstance(fold_count, int) or fold_count < 2:
        raise ValueError("Use at least two expanding-window folds.")
    if not 0 < initial_fraction < 1:
        raise ValueError("Initial training fraction must lie between zero and one.")
    origins = pd.DatetimeIndex(dataset.forecast_origin.drop_duplicates().sort_values())
    initial = max(10, int(len(origins) * initial_fraction))
    if len(origins) - initial < fold_count:
        raise ValueError("Not enough development origins for the configured folds.")
    last_targets = dataset.groupby("forecast_origin").target_timestamp.max()
    folds = []
    for number, block in enumerate(np.array_split(np.arange(initial, len(origins)), fold_count), 1):
        evaluation_origins = origins[block]
        # Train only on earlier days whose 24 target hours also end before
        # the test block starts. This stops the training targets from
        # overlapping the test period (called "purging").
        training_origins = origins[(origins < evaluation_origins.min()) & last_targets.reindex(origins).le(evaluation_origins.min()).to_numpy()]
        if training_origins.empty:
            raise ValueError("Purging leaves an empty training fold.")
        folds.append({"fold": number, "train_origins": training_origins, "evaluation_origins": evaluation_origins})
    return folds


def run_development_diagnostics(hourly: pd.DataFrame, metadata: dict, protocol: dict, model_factory=None) -> dict:
    """Train and score the models on each expanding fold, without using the test set.

    Inputs: the hourly data, the saved training metadata (split dates), the
    protocol settings (fold count and so on) and an optional model factory.
    Output: a dict with fold scores, all predictions, scores per hour ahead,
    the fold dates, and a fingerprint of the data that was used.
    """
    development, cutoff = development_data(hourly, metadata)
    dataset = build_supervised_dataset(development)
    validation_end = pd.Timestamp(metadata["boundaries"]["validation_end"])
    # Keep only origins up to the end of validation. The test days stay hidden.
    dataset = dataset.loc[dataset.forecast_origin <= validation_end].copy()
    if not dataset.forecast_origin.dt.hour.eq(protocol["origin_hour"]).all():
        raise ValueError("Development origins do not match the recorded daily schedule.")
    folds = expanding_folds(dataset, protocol["fold_count"], protocol["initial_training_fraction"])
    factory = model_factory or _models
    frames, metrics, manifest = [], [], []
    for fold in folds:
        train = dataset.loc[dataset.forecast_origin.isin(fold["train_origins"])].copy()
        evaluation = dataset.loc[dataset.forecast_origin.isin(fold["evaluation_origins"])].copy()
        if train.target_timestamp.max() > evaluation.forecast_origin.min():
            raise ValueError("Training targets extend beyond the evaluation origin.")
        # The 90th percentile of training energy values, saved with the results.
        threshold = float(train.target_kwh.quantile(0.9))
        # Create fresh models for every fold. Seasonal Naive needs no training.
        fitted = factory()
        predictions = [_predict_frame(evaluation, "Seasonal Naive", None)]
        for name, estimator in fitted.items():
            estimator.fit(train[FEATURE_COLUMNS], train.target_kwh)
            predictions.append(_predict_frame(evaluation, name, estimator))
        prediction = pd.concat(predictions, ignore_index=True)
        prediction["fold"] = fold["fold"]
        prediction["fold_training_threshold_kwh"] = threshold
        frames.append(prediction)
        for name, group in prediction.groupby("model"):
            metrics.append({"fold": fold["fold"], "model": name, "evaluation_days": len(fold["evaluation_origins"]),
                            "training_days": len(fold["train_origins"]), **regression_metrics(group.actual_kwh, group.predicted_kwh)})
        manifest.append({"fold": fold["fold"], "train_origins": [str(t) for t in fold["train_origins"]],
                         "evaluation_origins": [str(t) for t in fold["evaluation_origins"]],
                         "last_training_target": str(train.target_timestamp.max()),
                         "first_evaluation_target": str(evaluation.target_timestamp.min()),
                         "last_evaluation_target": str(evaluation.target_timestamp.max()),
                         "training_threshold_kwh": threshold})
    predictions = pd.concat(frames, ignore_index=True)
    return {"fold_metrics": pd.DataFrame(metrics), "predictions": predictions,
            "horizon_metrics": metrics_by_horizon(predictions), "folds": manifest,
            "development_origins": int(dataset.forecast_origin.nunique()), "development_cutoff": str(cutoff),
            "development_fingerprint": hashlib.sha256(pd.util.hash_pandas_object(development, index=True).values.tobytes()).hexdigest()}
