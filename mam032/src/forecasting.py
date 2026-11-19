"""Inference functions shared by the Streamlit application and tests."""

from __future__ import annotations

from pathlib import Path

import joblib
import io
import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, features_for_origin
from .artifacts import verified_bytes


class ModelArtifactError(RuntimeError):
    """Raised when a saved model artifact is missing or invalid."""


def _validate_bundle(bundle: object) -> dict:
    """Check that a loaded model file has the structure the app expects.

    The bundle must hold the models, the selected model name, the high-demand
    threshold and the feature list. Raises ModelArtifactError if anything is
    wrong, and returns the same bundle if it is fine.
    """
    if not isinstance(bundle, dict):
        raise ModelArtifactError("Model artifact must be a dictionary.")
    required = {"models", "selected_model", "peak_threshold_kwh", "feature_columns"}
    if not required.issubset(bundle):
        raise ModelArtifactError("Model artifact is missing required keys.")
    # The model must have been trained on exactly the same features as the app uses.
    if bundle["feature_columns"] != FEATURE_COLUMNS:
        raise ModelArtifactError("Model feature schema does not match this application. Retrain the models.")
    if not isinstance(bundle["models"], dict):
        raise ModelArtifactError("Model artifact contains an invalid models mapping.")
    try:
        threshold = float(bundle["peak_threshold_kwh"])
    except (TypeError, ValueError) as exc:
        raise ModelArtifactError("Peak threshold must be finite and non-negative.") from exc
    if not np.isfinite(threshold) or threshold < 0:
        raise ModelArtifactError("Peak threshold must be finite and non-negative.")
    selected = bundle["selected_model"]
    if selected != "Seasonal Naive" and selected not in bundle["models"]:
        raise ModelArtifactError("The selected model is unavailable in the model artifact.")
    for name, model in bundle["models"].items():
        if not callable(getattr(model, "predict", None)):
            raise ModelArtifactError(f"Model '{name}' does not provide a prediction method.")
    return bundle


def load_model_bundle(path: Path, expected_dataset_id: str | None = None, expected_sha256: str | None = None) -> dict:
    """Load a locally generated, trusted joblib bundle and validate its schema.

    Joblib uses pickle internally. Never use this function with uploaded or
    untrusted files; user uploads are limited to consumption CSV data.
    """
    path = Path(path)
    if not path.exists():
        raise ModelArtifactError(
            f"Model artifact not found at {path}. Run: python scripts/train_models.py"
        )
    try:
        bundle = joblib.load(io.BytesIO(verified_bytes(path, expected_sha256)))
    except Exception as exc:
        raise ModelArtifactError(f"Could not load model artifact: {exc}") from exc
    bundle = _validate_bundle(bundle)
    # Old bundles have no dataset_id. They were made for the UCI dataset.
    actual_id = bundle.get("dataset_id", "uci_household")
    if expected_dataset_id is not None and actual_id != expected_dataset_id:
        raise ModelArtifactError("The saved model belongs to a different dataset. Train the selected dataset.")
    return bundle


def forecast_24h(
    hourly: pd.DataFrame | pd.Series,
    origin: pd.Timestamp,
    bundle: dict,
    model_name: str | None = None,
) -> pd.DataFrame:
    """Predict the next 24 hourly values (kWh) after the forecast origin."""
    # Step 1: build the inputs from the history up to the origin.
    features = features_for_origin(hourly, pd.Timestamp(origin), horizons=24)
    model_name = model_name or bundle["selected_model"]
    # Step 2: predict.
    if model_name == "Seasonal Naive":
        # The baseline simply repeats the same hour from yesterday.
        prediction = features["target_lag_24"].to_numpy()
    else:
        model = bundle.get("models", {}).get(model_name)
        if model is None:
            raise ModelArtifactError(f"Model '{model_name}' is unavailable.")
        try:
            prediction = np.asarray(model.predict(features[FEATURE_COLUMNS]), dtype=float)
        except Exception as exc:
            raise ModelArtifactError(f"Model '{model_name}' could not produce predictions: {exc}") from exc
        if prediction.shape != (24,) or not np.isfinite(prediction).all():
            raise ModelArtifactError("Model must produce exactly 24 finite hourly predictions.")
        # Energy use cannot be negative, so we cut any negative prediction to 0.
        prediction = np.clip(prediction, 0.0, None)
    # Step 3: put the answer in a clear table.
    result = features.loc[:, ["forecast_origin", "target_timestamp", "horizon"]].copy()
    result["forecast_issued_at"] = pd.Timestamp(origin) + pd.Timedelta(hours=1)
    result["predicted_kwh"] = prediction
    result["model"] = model_name
    # Mark hours above the "high demand" level found on the training data.
    result["is_high_demand"] = result["predicted_kwh"] >= float(bundle["peak_threshold_kwh"])
    # In a historical test, the real values are known, so we add them for comparison.
    series = hourly["energy_kwh"] if isinstance(hourly, pd.DataFrame) else hourly
    result["actual_kwh"] = series.reindex(pd.DatetimeIndex(result["target_timestamp"])).to_numpy()
    return result
