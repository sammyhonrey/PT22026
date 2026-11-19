"""Training and artifact generation for EcoWatt AI models."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import chronological_origin_split, metrics_by_horizon, regression_metrics
from .features import FEATURE_COLUMNS, build_supervised_dataset


def _models() -> dict[str, object]:
    """The two models we train. (Seasonal Naive needs no training.)"""
    return {
        # Ridge: a straight-line model. The scaler puts all features on the same
        # scale first. alpha=10 keeps the weights small to avoid overfitting.
        "Ridge": Pipeline(
            [("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))]
        ),
        # HistGradientBoosting: many small decision trees, each one fixing the
        # mistakes of the trees before it. It can learn curved patterns.
        "HistGradientBoosting": HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=220,
            max_leaf_nodes=24,
            min_samples_leaf=30,
            l2_regularization=0.2,
            random_state=42,
            # A random internal holdout would mix horizon rows and nearby
            # origins. Use the pre-specified iteration budget instead, and
            # reserve chronological validation for model selection.
            early_stopping=False,
        ),
    }


def _predict_frame(split: pd.DataFrame, model_name: str, model: object | None) -> pd.DataFrame:
    """Predict one data split and return a tidy table of actual vs predicted.

    Input: a split (train, validation or test), the model name, and the model.
    For "Seasonal Naive" we pass None as the model, because it needs no training.
    Output: one row per forecast hour, with the real value and the prediction.
    """
    # Seasonal Naive copies the same hour from yesterday. The other models
    # predict from the features. Energy cannot be negative, so we cut at 0.
    predicted = (
        split["target_lag_24"].to_numpy()
        if model_name == "Seasonal Naive"
        else np.clip(model.predict(split[FEATURE_COLUMNS]), 0.0, None)
    )
    return pd.DataFrame(
        {
            "forecast_origin": split["forecast_origin"].to_numpy(),
            "target_timestamp": split["target_timestamp"].to_numpy(),
            "horizon": split["horizon"].to_numpy(),
            "actual_kwh": split["target_kwh"].to_numpy(),
            "predicted_kwh": predicted,
            "model": model_name,
        }
    )


def train_and_evaluate(hourly: pd.DataFrame, artifact_dir: Path, *, dataset_id: str = "uci_household") -> dict:
    """Fit models, evaluate an untouched chronological test set, and save evidence.

    Inputs: the hourly energy table, the folder to save files in, and a
    dataset id (for example "xky_2025").
    Output: a dict of training metadata (also saved as training_metadata.json).
    Files saved in the folder: the model bundle (model_bundle.joblib), the
    predictions and scores on validation and test days, the scores per hour
    ahead, and the feature importance table.
    """
    # Overview: 1) build data, 2) split by time, 3) train, 4) score, 5) save.
    start = time.perf_counter()
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # Step 1-2: make the feature table, then split it in time order.
    dataset = build_supervised_dataset(hourly)
    train, validation, test = chronological_origin_split(dataset)

    # Step 3: train both models on the training days only.
    fitted = _models()
    for model in fitted.values():
        model.fit(train[FEATURE_COLUMNS], train["target_kwh"])

    # Step 4: predict with all three models, including the baseline.
    validation_predictions = [_predict_frame(validation, "Seasonal Naive", None)]
    test_predictions = [_predict_frame(test, "Seasonal Naive", None)]
    for name, model in fitted.items():
        validation_predictions.append(_predict_frame(validation, name, model))
        test_predictions.append(_predict_frame(test, name, model))
    validation_predictions = pd.concat(validation_predictions, ignore_index=True)
    test_predictions = pd.concat(test_predictions, ignore_index=True)

    # Score every model on validation and on test.
    validation_metrics = []
    test_metrics = []
    for name, group in validation_predictions.groupby("model"):
        validation_metrics.append({"model": name, **regression_metrics(group.actual_kwh, group.predicted_kwh)})
    for name, group in test_predictions.groupby("model"):
        test_metrics.append({"model": name, **regression_metrics(group.actual_kwh, group.predicted_kwh)})
    validation_metrics = pd.DataFrame(validation_metrics).sort_values("mae_kwh")
    test_metrics = pd.DataFrame(test_metrics).sort_values("mae_kwh")
    # The best model is chosen on validation MAE only. The test set stays
    # untouched so the final test score is an honest estimate.
    selected_model = str(validation_metrics.iloc[0]["model"])

    # The "high-demand" level: the value that 90% of training hours stay below.
    # It comes only from training targets, never validation or test, so the
    # later high-demand checks stay fair.
    peak_threshold = float(train["target_kwh"].quantile(0.90))
    # Which inputs matter most? Shuffle one feature at a time and see how much
    # the error grows (permutation importance).
    importance_model = fitted["HistGradientBoosting"]
    # Use at most 4000 validation rows (a random sample, fixed seed 42), so the
    # calculation stays fast and gives the same answer every time.
    importance_sample =validation.sample(min(4000, len(validation)), random_state=42)
    importance = permutation_importance(
        importance_model,
        importance_sample[FEATURE_COLUMNS],
        importance_sample["target_kwh"],
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )
    importance_frame = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    # Step 5: save the models and all evidence files.
    # The bundle is the one file the app loads. It holds the trained models and
    # the facts the app needs to use them correctly.
    bundle = {
        "dataset_id": dataset_id,
        "models": fitted,
        "feature_columns": FEATURE_COLUMNS,
        "selected_model": selected_model,
        "peak_threshold_kwh": peak_threshold,
        "trained_through": str(pd.Timestamp(train["target_timestamp"].max())),
        "data_first_timestamp": str(hourly.index.min()),
        "data_last_timestamp": str(hourly.index.max()),
    }
    joblib.dump(bundle, artifact_dir / "model_bundle.joblib", compress=3)
    validation_predictions.to_csv(artifact_dir / "validation_predictions.csv", index=False)
    test_predictions.to_csv(artifact_dir / "test_predictions.csv", index=False)
    validation_metrics.to_csv(artifact_dir / "validation_metrics.csv", index=False)
    test_metrics.to_csv(artifact_dir / "test_metrics.csv", index=False)
    metrics_by_horizon(test_predictions).to_csv(artifact_dir / "per_horizon_metrics.csv", index=False)
    importance_frame.to_csv(artifact_dir / "permutation_importance.csv", index=False)

    # The metadata records exactly how this training run was done (settings,
    # split dates, library versions), so the result can be repeated and checked.
    metadata = {
        "dataset_id": dataset_id,
        "selected_model": selected_model,
        "selection_rule": "lowest validation MAE; test set was not used for selection",
        "training_configuration": {
            "Ridge": {"alpha": 10.0, "standardized": True},
            "HistGradientBoosting": {
                "max_iter": 220,
                "learning_rate": 0.06,
                "max_leaf_nodes": 24,
                "min_samples_leaf": 30,
                "l2_regularization": 0.2,
                "early_stopping": False,
                "random_state": 42,
            },
        },
        "peak_threshold_kwh": peak_threshold,
        "origins": {
            "train": int(train["forecast_origin"].nunique()),
            "validation": int(validation["forecast_origin"].nunique()),
            "test": int(test["forecast_origin"].nunique()),
        },
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "boundaries": {
            "train_start": str(train["forecast_origin"].min()),
            "train_end": str(train["forecast_origin"].max()),
            "validation_start": str(validation["forecast_origin"].min()),
            "validation_end": str(validation["forecast_origin"].max()),
            "test_start": str(test["forecast_origin"].min()),
            "test_end": str(test["forecast_origin"].max()),
        },
        "runtime_seconds": round(time.perf_counter() - start, 3),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }
    (artifact_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata
