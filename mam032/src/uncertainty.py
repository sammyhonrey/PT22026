"""Simple extras for the forecast: a strong baseline and an 80% range.

Everything here uses only data that is known at the forecast origin, or the
errors made on the validation days. The test days are never used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BLOCK_HOURS = 6          # we group the day into four blocks of 6 hours
LOWER_QUANTILE = 0.10    # 10% to 90% gives an 80% range
UPPER_QUANTILE = 0.90
MIN_ROWS_PER_BLOCK = 30  # too few validation rows means no reliable range


def seasonal_median_forecast(hourly: pd.DataFrame | pd.Series, origin: pd.Timestamp, days: int = 7) -> pd.DataFrame:
    """Predict each of the next 24 hours as the median of the same hour on the last 7 days.

    Example: to predict tomorrow 08:00, take 08:00 from each of the last 7 days
    and use the middle value. The median ignores one strange day.
    """
    series = hourly["energy_kwh"] if isinstance(hourly, pd.DataFrame) else hourly
    origin = pd.Timestamp(origin)
    rows = []
    for horizon in range(1, 25):
        target_time = origin + pd.Timedelta(hours=horizon)
        # Same hour on earlier days. Every one of these hours is at or before the origin.
        earlier_times = [target_time - pd.Timedelta(hours=24 * k) for k in range(1, days + 1)]
        values = series.reindex(pd.DatetimeIndex(earlier_times)).dropna()
        rows.append({
            "target_timestamp": target_time,
            "horizon": horizon,
            "baseline_kwh": float(values.median()) if len(values) else np.nan,
        })
    return pd.DataFrame(rows)


def interval_offsets(validation: pd.DataFrame, model_name: str) -> dict[int, tuple[float, float]] | None:
    """Learn how far real values were below and above the prediction, per 6-hour block.

    Uses validation days only. Returns None when there is not enough data.
    """
    needed = {"target_timestamp", "actual_kwh", "predicted_kwh", "model"}
    if not needed.issubset(validation.columns):
        return None
    rows = validation[validation["model"] == model_name]
    if rows.empty:
        return None
    errors = rows["actual_kwh"] - rows["predicted_kwh"]
    blocks = pd.DatetimeIndex(rows["target_timestamp"]).hour // BLOCK_HOURS
    offsets = {}
    for block in range(24 // BLOCK_HOURS):
        block_errors = errors[np.asarray(blocks == block)]
        if len(block_errors) < MIN_ROWS_PER_BLOCK:
            return None
        offsets[block] = (float(block_errors.quantile(LOWER_QUANTILE)), float(block_errors.quantile(UPPER_QUANTILE)))
    return offsets


def load_interval_offsets(folder: Path, model_name: str) -> dict[int, tuple[float, float]] | None:
    """Read the saved validation predictions, if they exist, and build the offsets."""
    path = Path(folder) / "validation_predictions.csv"
    try:
        validation = pd.read_csv(path, parse_dates=["target_timestamp"])
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    return interval_offsets(validation, model_name)


def add_interval(forecast: pd.DataFrame, offsets: dict[int, tuple[float, float]]) -> pd.DataFrame:
    """Add lower_kwh and upper_kwh columns (energy cannot be below zero)."""
    result = forecast.copy()
    blocks = pd.DatetimeIndex(result["target_timestamp"]).hour // BLOCK_HOURS
    low = np.array([offsets[int(b)][0] for b in blocks])
    high = np.array([offsets[int(b)][1] for b in blocks])
    result["lower_kwh"] = np.clip(result["predicted_kwh"].to_numpy() + low, 0.0, None)
    result["upper_kwh"] = np.clip(result["predicted_kwh"].to_numpy() + high, 0.0, None)
    return result


def baseline_test_scores(hourly: pd.DataFrame | pd.Series, test_origins: list[pd.Timestamp]) -> dict[str, float]:
    """Score the 7-day same-hour median over the same test origins as the models."""
    series = hourly["energy_kwh"] if isinstance(hourly, pd.DataFrame) else hourly
    errors = []
    for origin in test_origins:
        forecast = seasonal_median_forecast(series, origin)
        actual = series.reindex(pd.DatetimeIndex(forecast["target_timestamp"])).to_numpy()
        errors.append(np.abs(actual - forecast["baseline_kwh"].to_numpy()))
    all_errors = np.concatenate(errors) if errors else np.array([])
    all_errors = all_errors[np.isfinite(all_errors)]
    if all_errors.size == 0:
        return {"mae_kwh": float("nan"), "n_hours": 0}
    return {"mae_kwh": float(all_errors.mean()), "n_hours": int(all_errors.size)}
