"""Regression tests for scientific invariants rather than implementation details."""

import joblib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.linear_model import Ridge

from src.evaluation import chronological_origin_split
from src.data import DataValidationError, minute_power_to_hourly
from src.features import FEATURE_COLUMNS, build_supervised_dataset, features_for_origin
from src.forecasting import ModelArtifactError, forecast_24h, load_model_bundle
from src.training import _models


@pytest.mark.parametrize("horizons", [0, 25, 168, 1.5, True])
def test_unsafe_horizon_is_rejected(hourly_series, horizons):
    with pytest.raises(ValueError, match="integer from 1 to 24"):
        features_for_origin(hourly_series, hourly_series.index[500], horizons)


def test_missing_hour_is_not_replaced_by_older_observation(hourly_series):
    origin = hourly_series.index[500]
    irregular = hourly_series.drop(hourly_series.index[450])
    with pytest.raises(ValueError, match="169 consecutive"):
        features_for_origin(irregular, origin)


def test_duplicate_timestamp_is_rejected(hourly_series):
    duplicated = pd.concat([hourly_series, hourly_series.iloc[[100]]])
    with pytest.raises(ValueError, match="valid and unique"):
        features_for_origin(duplicated, hourly_series.index[500])


@pytest.mark.parametrize("value", [-1.0, np.inf, np.nan])
def test_invalid_required_history_is_rejected(hourly_series, value):
    origin = hourly_series.index[500]
    broken = hourly_series.copy()
    broken.loc[origin, "energy_kwh"] = value
    with pytest.raises(ValueError, match="missing, non-finite, or negative"):
        features_for_origin(broken, origin)


def test_custom_origin_requires_complete_target_block(hourly_series):
    with pytest.raises(ValueError, match="24 finite"):
        build_supervised_dataset(hourly_series, pd.DatetimeIndex([hourly_series.index[-10]]))


def test_fitted_prediction_is_invariant_to_future_mutation(hourly_series):
    dataset = build_supervised_dataset(hourly_series)
    train = dataset[dataset["forecast_origin"] < hourly_series.index[24 * 30]]
    ridge = Ridge(alpha=10).fit(train[FEATURE_COLUMNS], train["target_kwh"])
    bundle = {"models": {"Ridge": ridge}, "selected_model": "Ridge", "peak_threshold_kwh": 1.0}
    origin = hourly_series.index[24 * 35 + 23]
    original = forecast_24h(hourly_series, origin, bundle)
    altered = hourly_series.copy()
    altered.loc[altered.index > origin, "energy_kwh"] = 9999.0
    mutated = forecast_24h(altered, origin, bundle)
    assert_frame_equal(
        original[["target_timestamp", "horizon", "predicted_kwh"]],
        mutated[["target_timestamp", "horizon", "predicted_kwh"]],
    )


def test_overlapping_origins_are_purged_at_split_boundaries(hourly_series):
    origins = hourly_series.index[200:300]
    dataset = build_supervised_dataset(hourly_series, origins)
    train, validation, test = chronological_origin_split(dataset, 0.5, 0.25)
    assert train["target_timestamp"].max() <= validation["forecast_origin"].min()
    assert validation["target_timestamp"].max() <= test["forecast_origin"].min()
    assert (train.groupby("forecast_origin").size() == 24).all()
    assert (validation.groupby("forecast_origin").size() == 24).all()


@pytest.mark.parametrize("fractions", [(0, 0.2), (0.8, 0.3), (0.7, -0.1)])
def test_invalid_split_fraction_is_rejected(hourly_series, fractions):
    dataset = build_supervised_dataset(hourly_series)
    with pytest.raises(ValueError, match="fractions"):
        chronological_origin_split(dataset, *fractions)


def test_boosting_does_not_create_random_internal_validation_split():
    assert _models()["HistGradientBoosting"].early_stopping is False


@pytest.mark.parametrize("bundle,message", [
    ([1, 2], "dictionary"),
    ({"models": {}}, "missing required"),
    ({"models": {}, "selected_model": "Seasonal Naive", "feature_columns": ["wrong"], "peak_threshold_kwh": 1.0}, "feature schema"),
    ({"models": {}, "selected_model": "Seasonal Naive", "feature_columns": FEATURE_COLUMNS, "peak_threshold_kwh": np.inf}, "threshold"),
    ({"models": {}, "selected_model": "Ridge", "feature_columns": FEATURE_COLUMNS, "peak_threshold_kwh": 1.0}, "selected model"),
])
def test_invalid_model_bundle_is_rejected(tmp_path, bundle, message):
    path = tmp_path / "invalid.joblib"
    joblib.dump(bundle, path)
    with pytest.raises(ModelArtifactError, match=message):
        load_model_bundle(path)


@pytest.mark.parametrize("case", ["duplicate", "infinite", "subminute"])
def test_minute_quality_cannot_overstate_coverage(case):
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=60, freq="min"),
        "global_active_power_kw": np.ones(60),
    })
    if case == "duplicate":
        frame.loc[59, "timestamp"] = frame.loc[58, "timestamp"]
    elif case == "infinite":
        frame.loc[59, "global_active_power_kw"] = np.inf
    else:
        frame.loc[59, "timestamp"] += pd.Timedelta(seconds=30)
    with pytest.raises(DataValidationError):
        minute_power_to_hourly(frame)
