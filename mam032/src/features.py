"""Leakage-free feature engineering for direct 24-hour forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

# The 16 inputs (features) the models use. All of them are known at the
# forecast origin, so the model never "sees the future".
FEATURE_COLUMNS = [
    # Which hour ahead we predict (1 to 24).
    "horizon",
    # Recent usage: the last value, and values 1, 2 and 7 days ago.
    "last_value",
    "origin_lag_24",
    "origin_lag_48",
    "origin_lag_168",
    # Usage at the target hour yesterday and one week ago.
    "target_lag_24",
    "target_lag_168",
    # Average and spread of usage over the last 24 hours and last 7 days.
    "rolling_mean_24",
    "rolling_std_24",
    "rolling_mean_168",
    "rolling_std_168",
    # Time of the target hour. Sine/cosine let the model know that
    # 23:00 and 00:00 are close to each other.
    "target_hour_sin",
    "target_hour_cos",
    "target_dow_sin",
    "target_dow_cos",
    "target_is_weekend",
]


def _series(hourly: pd.DataFrame | pd.Series) -> pd.Series:
    """Return the hourly energy as a clean, time-sorted Series of floats.

    Accepts a table with an energy_kwh column, or a Series. Raises ValueError
    if the column is missing, the index is not time, or timestamps repeat.
    """
    if isinstance(hourly, pd.DataFrame) and "energy_kwh" not in hourly.columns:
        raise ValueError("Hourly data must contain an energy_kwh column.")
    series = hourly["energy_kwh"] if isinstance(hourly, pd.DataFrame) else hourly
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Hourly energy must use a DatetimeIndex.")
    if series.index.hasnans or series.index.has_duplicates:
        raise ValueError("Hourly timestamps must be valid and unique.")
    return series.astype(float).sort_index()


def features_for_origin(
    hourly: pd.DataFrame | pd.Series, origin: pd.Timestamp, horizons: int = 24
) -> pd.DataFrame:
    """Create features using consumption observed no later than ``origin``.

    The origin is the last hour we already know. One row is made for each
    of the next ``horizons`` hours (24 by default).
    """
    if isinstance(horizons, bool) or not isinstance(horizons, (int, np.integer)) or not 1 <= horizons <= 24:
        raise ValueError("The supported forecast horizon must be an integer from 1 to 24.")
    y = _series(hourly)
    origin = pd.Timestamp(origin)
    if origin not in y.index:
        raise ValueError("Forecast origin is not present in the hourly series.")
    # We need 169 hours of history: 168 hours (7 days) plus the origin hour.
    required_index = pd.date_range(end=origin, periods=169, freq="h")
    history = y.reindex(required_index)
    if not required_index.isin(y.index).all():
        raise ValueError("At least 169 consecutive hourly observations are required through the origin.")
    if not np.isfinite(history).all() or (history < 0).any():
        raise ValueError("The required 169-hour history contains missing, non-finite, or negative energy values.")

    # Build one row of features for every hour we want to predict.
    rows = []
    for horizon in range(1, horizons + 1):
        target_time = origin + pd.Timedelta(hours=horizon)
        row = {
            "forecast_origin": origin,
            "target_timestamp": target_time,
            "horizon": horizon,
            "last_value": history.iloc[-1],
            "origin_lag_24": history.loc[origin - pd.Timedelta(hours=24)],
            "origin_lag_48": history.loc[origin - pd.Timedelta(hours=48)],
            "origin_lag_168": history.loc[origin - pd.Timedelta(hours=168)],
            "target_lag_24": history.loc[target_time - pd.Timedelta(hours=24)],
            "target_lag_168": history.loc[target_time - pd.Timedelta(hours=168)],
            "rolling_mean_24": history.iloc[-24:].mean(),
            "rolling_std_24": history.iloc[-24:].std(ddof=0),
            "rolling_mean_168": history.iloc[-168:].mean(),
            "rolling_std_168": history.iloc[-168:].std(ddof=0),
            "target_hour_sin": np.sin(2 * np.pi * target_time.hour / 24),
            "target_hour_cos": np.cos(2 * np.pi * target_time.hour / 24),
            "target_dow_sin": np.sin(2 * np.pi * target_time.dayofweek / 7),
            "target_dow_cos": np.cos(2 * np.pi * target_time.dayofweek / 7),
            "target_is_weekend": int(target_time.dayofweek >= 5),
        }
        rows.append(row)
    features = pd.DataFrame(rows)
    return features


def valid_daily_origins(hourly: pd.DataFrame | pd.Series) -> pd.DatetimeIndex:
    """Return local-clock 23:00 origins with complete past features and 24-hour targets."""
    y = _series(hourly)
    # One forecast per day, issued after the 23:00 hour is complete.
    candidates = y.index[y.index.hour == 23]
    valid = []
    for origin in candidates:
        past = y.reindex(pd.date_range(end=origin, periods=169, freq="h"))
        future = y.reindex(pd.date_range(start=origin + pd.Timedelta(hours=1), periods=24, freq="h"))
        # Keep the day only if the past 169 hours and next 24 hours are all valid.
        if np.isfinite(past).all() and np.isfinite(future).all() and (past >= 0).all() and (future >= 0).all():
            valid.append(origin)
    return pd.DatetimeIndex(valid)


def build_supervised_dataset(
    hourly: pd.DataFrame | pd.Series, origins: pd.DatetimeIndex | None = None
) -> pd.DataFrame:
    """Build one row per origin and horizon with only origin-safe predictors."""
    y = _series(hourly)
    origins = valid_daily_origins(y) if origins is None else pd.DatetimeIndex(origins)
    if origins.hasnans or origins.has_duplicates:
        raise ValueError("Forecast origins must be valid and unique.")
    # For each origin, make the features and attach the real answer (target).
    parts = []
    for origin in origins:
        part = features_for_origin(y, origin)
        part["target_kwh"] = y.reindex(pd.DatetimeIndex(part["target_timestamp"])).to_numpy()
        if not np.isfinite(part["target_kwh"]).all() or (part["target_kwh"] < 0).any():
            raise ValueError(f"Origin {origin} requires 24 finite, non-negative hourly targets.")
        parts.append(part)
    if not parts:
        raise ValueError("No valid forecast origins were found.")
    dataset = pd.concat(parts, ignore_index=True)
    needed = FEATURE_COLUMNS + ["target_kwh"]
    return dataset.dropna(subset=needed).reset_index(drop=True)
