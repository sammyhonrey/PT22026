"""Evaluation metrics and chronological split helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Score predictions with MAE, RMSE and R2 (all except R2 are in kWh)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    # MAE: average size of the error. RMSE: like MAE but punishes big errors more.
    # R2: how much of the up-and-down pattern the model explains (1.0 is perfect).
    return {
        "mae_kwh": float(mean_absolute_error(actual, predicted)),
        "rmse_kwh": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }


def chronological_origin_split(
    dataset: pd.DataFrame, train_fraction: float = 0.70, validation_fraction: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split days in time order: oldest days train, then validation, newest test.

    We never shuffle. Shuffling would let the model learn from days that come
    after the days it is tested on, which is cheating for time series.

    Input: a table with forecast_origin and target_timestamp columns, and the
    fractions for training (70%) and validation (15%). The rest (15%) is test.
    Output: three tables: train, validation and test.
    """
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1 and train_fraction + validation_fraction < 1):
        raise ValueError("Split fractions must be positive and leave a non-empty test fraction.")
    required = {"forecast_origin", "target_timestamp"}
    if not required.issubset(dataset.columns):
        raise ValueError("Dataset must contain forecast_origin and target_timestamp columns.")
    # Sort the forecast days from oldest to newest, then cut the list in three.
    origins = pd.DatetimeIndex(dataset["forecast_origin"].drop_duplicates().sort_values())
    n = len(origins)
    if n < 30:
        raise ValueError("At least 30 valid daily origins are required for chronological evaluation.")
    # Cut points in the list of days. Each part keeps at least one day, so the
    # max() and min() calls protect against empty parts.
    train_end = max(1, int(n * train_fraction))
    validation_end = max(train_end + 1, int(n * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, n - 1)
    train_origins = origins[:train_end]
    validation_origins = origins[train_end:validation_end]
    test_origins = origins[validation_end:]
    train = dataset[dataset["forecast_origin"].isin(train_origins)].copy()
    validation = dataset[dataset["forecast_origin"].isin(validation_origins)].copy()
    test = dataset[dataset["forecast_origin"].isin(test_origins)].copy()
    # Purge whole origins, rather than individual horizons, if a target block
    # would extend past the next partition's first available forecast origin.
    # This also protects callers that use hourly/overlapping origins.
    train_max_target = train.groupby("forecast_origin")["target_timestamp"].max()
    validation_max_target = validation.groupby("forecast_origin")["target_timestamp"].max()
    safe_train = train_max_target.index[train_max_target <= validation_origins.min()]
    safe_validation = validation_max_target.index[validation_max_target <= test_origins.min()]
    train = train[train["forecast_origin"].isin(safe_train)].copy()
    validation = validation[validation["forecast_origin"].isin(safe_validation)].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Chronological partitions are empty after purging overlapping target blocks.")
    return train, validation, test


def metrics_by_horizon(predictions: pd.DataFrame) -> pd.DataFrame:
    """Score each model separately for every hour ahead (1 to 24)."""
    records = []
    for (model, horizon), group in predictions.groupby(["model", "horizon"], sort=True):
        records.append(
            {
                "model": model,
                "horizon": int(horizon),
                **regression_metrics(group["actual_kwh"], group["predicted_kwh"]),
            }
        )
    return pd.DataFrame(records)
