import numpy as np
import pandas as pd
import joblib
import pytest

from src.forecasting import ModelArtifactError, forecast_24h, load_model_bundle
from src.features import FEATURE_COLUMNS


def test_seasonal_naive_returns_24_aligned_horizons(hourly_series: pd.DataFrame) -> None:
    origin = hourly_series.index[24 * 20 + 23]
    bundle = {"selected_model": "Seasonal Naive", "peak_threshold_kwh": 1.0, "models": {}}
    result = forecast_24h(hourly_series, origin, bundle)
    assert len(result) == 24
    assert result["horizon"].tolist() == list(range(1, 25))
    assert result.iloc[0]["target_timestamp"] == origin + pd.Timedelta(hours=1)
    assert result.iloc[-1]["target_timestamp"] == origin + pd.Timedelta(hours=24)
    assert (result["forecast_issued_at"] == origin + pd.Timedelta(hours=1)).all()
    assert np.isfinite(result["predicted_kwh"]).all()


def test_missing_model_artifact_has_actionable_error(tmp_path) -> None:
    try:
        load_model_bundle(tmp_path / "missing.joblib")
    except ModelArtifactError as exc:
        assert "train_models.py" in str(exc)
    else:
        raise AssertionError("Expected ModelArtifactError")


def test_model_from_other_household_is_rejected(tmp_path):
    path = tmp_path / "other.joblib"
    joblib.dump({"dataset_id": "uci_household", "models": {},
                 "selected_model": "Seasonal Naive", "peak_threshold_kwh": 1.0,
                 "feature_columns": FEATURE_COLUMNS}, path)
    with pytest.raises(ModelArtifactError, match="different dataset"):
        load_model_bundle(path, expected_dataset_id="xky_2025")


def test_legacy_model_is_recognized_only_as_uci(tmp_path):
    path = tmp_path / "legacy.joblib"
    joblib.dump({"models": {}, "selected_model": "Seasonal Naive", "peak_threshold_kwh": 1.0,
                 "feature_columns": FEATURE_COLUMNS}, path)
    assert load_model_bundle(path, expected_dataset_id="uci_household")
    with pytest.raises(ModelArtifactError, match="different dataset"):
        load_model_bundle(path, expected_dataset_id="xky_2025")
