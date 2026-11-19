from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def hourly_series() -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=24 * 50, freq="h")
    values = 0.8 + 0.25 * np.sin(2 * np.pi * index.hour / 24) + 0.05 * np.cos(2 * np.pi * index.dayofweek / 7)
    return pd.DataFrame({"energy_kwh": values}, index=index)


@pytest.fixture
def diagnostic_payload():
    """Small, clearly synthetic three-model evidence for schema/recovery tests."""
    from src.diagnostics import prediction_diagnostics

    origins = pd.date_range("2025-01-31 23:00", periods=2, freq="D")
    predictions = pd.concat([
        pd.DataFrame({"model": model, "forecast_origin": origin,
                      "target_timestamp": pd.date_range(origin + pd.Timedelta(hours=1), periods=24, freq="h"),
                      "horizon": range(1, 25), "actual_kwh": [3, 3] + [1] * 22,
                      "predicted_kwh": [3, 1, 3] + [1] * 21})
        for model in ["HistGradientBoosting", "Ridge", "Seasonal Naive"] for origin in origins
    ], ignore_index=True)
    descriptive = prediction_diagnostics(predictions, 2)
    fold_rows, folds = [], []
    for number in range(1, 4):
        train = pd.date_range("2025-01-01 23:00", periods=number, freq="D")
        evaluation = pd.Timestamp("2025-01-10 23:00") + pd.Timedelta(days=number)
        folds.append({"fold": number, "train_origins": [str(t) for t in train], "evaluation_origins": [str(evaluation)],
                      "last_training_target": str(train[-1] + pd.Timedelta(hours=24)),
                      "first_evaluation_target": str(evaluation + pd.Timedelta(hours=1)),
                      "last_evaluation_target": str(evaluation + pd.Timedelta(hours=24))})
        fold_rows.extend({"fold": number, "model": model, "training_days": number, "evaluation_days": 1,
                          "mae_kwh": .2, "rmse_kwh": .4} for model in predictions.model.unique())
    frames = {"test_event_metrics.csv": descriptive["summary"], "test_daily_metrics.csv": descriptive["daily"],
              "test_horizon_metrics.csv": descriptive["horizon"], "rolling_horizon_metrics.csv": descriptive["horizon"].copy(),
              "rolling_fold_metrics.csv": pd.DataFrame(fold_rows)}
    manifest = {"dataset_id": "test-house", "development_origins": 6,
                "reserved_test_first_target": "2025-02-01 00:00", "folds": folds}
    return frames, manifest
