"""Bounded recorded-data analysis and explicit user-supplied energy scenarios.

These functions never fit a model, change a benchmark, identify an appliance,
fetch tariffs or issue switching commands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import regression_metrics
from .features import FEATURE_COLUMNS, features_for_origin
from .forecasting import ModelArtifactError

MODELS = ("Seasonal Naive", "Ridge", "HistGradientBoosting")
# The most recorded days the user can replay in one comparison.
MAX_REPLAY_DAYS = 14


def _hourly(hourly: pd.DataFrame) -> pd.Series:
    """Check the hourly table and return its energy_kwh column as a Series.

    The data must have one row per hour, in time order, with no duplicates,
    and every value must be a normal number that is not negative.
    Raises ValueError if any rule is broken.
    """
    if not isinstance(hourly, pd.DataFrame) or "energy_kwh" not in hourly:
        raise ValueError("Provide an hourly energy_kwh table.")
    index = hourly.index
    if (not isinstance(index, pd.DatetimeIndex) or index.hasnans or index.has_duplicates
            or not index.is_monotonic_increasing or hourly.empty
            or not ((index.minute == 0) & (index.second == 0) & (index.microsecond == 0) & (index.nanosecond == 0)).all()
            or not index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all()):
        raise ValueError("Analysis requires unique, ascending, consecutive hourly timestamps.")
    values = hourly.energy_kwh.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Analysis requires finite, non-negative hourly kWh.")
    return hourly.energy_kwh.astype(float)


def replay_origins(hourly: pd.DataFrame) -> pd.DatetimeIndex:
    """23:00 origins with 169 history hours and 24 recorded targets, no overlap."""
    y = _hourly(hourly)
    candidates = y.index[168:-24]
    return candidates[candidates.hour == 23]


def replay_models(hourly: pd.DataFrame, bundle: dict, days: int) -> dict:
    """Run all three models on the last `days` recorded days and score them.

    For each day we forecast from 23:00 and compare with the real values.
    Returns a dict with a summary table, daily scores, all predictions and
    the origins used. Raises ModelArtifactError if a model is missing or
    gives bad output.
    """
    if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= MAX_REPLAY_DAYS:
        raise ValueError(f"Choose between 1 and {MAX_REPLAY_DAYS} recorded days.")
    candidates = replay_origins(hourly)
    if len(candidates) < days:
        raise ValueError("Not enough complete 23:00 forecast windows for the selected day count.")
    origins = candidates[-days:]
    features = pd.concat([features_for_origin(hourly, origin) for origin in origins], ignore_index=True)
    actual = hourly.energy_kwh.reindex(pd.DatetimeIndex(features.target_timestamp)).to_numpy(dtype=float)
    frames, scores, daily = [], [], []
    for name in MODELS:
        if name == "Seasonal Naive":
            # The baseline predicts the same hour of the previous day.
            prediction = features.target_lag_24.to_numpy(dtype=float)
        else:
            estimator = bundle.get("models", {}).get(name)
            if estimator is None:
                raise ModelArtifactError(f"Cannot compare all methods: {name} is unavailable.")
            try:
                prediction = np.asarray(estimator.predict(features[FEATURE_COLUMNS]), dtype=float)
            except Exception as exc:
                raise ModelArtifactError(f"Recorded-window prediction failed for {name}: {exc}") from exc
            if prediction.shape != (len(features),) or not np.isfinite(prediction).all():
                raise ModelArtifactError(f"{name} returned invalid recorded-window predictions.")
            # Energy cannot be negative, so we cut negative predictions to zero.
            prediction = np.clip(prediction, 0, None)
        frame = features[["forecast_origin", "target_timestamp", "horizon"]].copy()
        frame = frame.assign(model=name, actual_kwh=actual, predicted_kwh=prediction)
        frames.append(frame)
        scores.append({"model": name, "days": days, "hours": len(frame), **regression_metrics(actual, prediction),
                       "bias_kwh": float(np.mean(prediction - actual))})
        for origin, group in frame.groupby("forecast_origin", sort=True):
            daily.append({"model": name, "forecast_origin": origin, **regression_metrics(group.actual_kwh, group.predicted_kwh),
                          "actual_total_kwh": float(group.actual_kwh.sum()),
                          "predicted_total_kwh": float(group.predicted_kwh.sum())})
    # Show how much each model's MAE is lower than the Seasonal Naive MAE.
    summary = pd.DataFrame(scores)
    baseline = float(summary.loc[summary.model.eq("Seasonal Naive"), "mae_kwh"].iloc[0])
    summary["mae_reduction_vs_naive_pct"] = (1 - summary.mae_kwh / baseline) * 100 if baseline > 0 else np.nan
    return {"summary": summary.sort_values("mae_kwh", kind="stable").reset_index(drop=True),
            "daily": pd.DataFrame(daily), "predictions": pd.concat(frames, ignore_index=True),
            "origins": origins, "scope": "Fixed pretrained-model replay; not a newly held-out benchmark or model selection."}


def consumption_profile(hourly: pd.DataFrame) -> dict:
    """Summarize at most 28 days of actual input; partial days stay partial."""
    # Use the last 28 days (28 * 24 hours) of data at most.
    y = _hourly(hourly).tail(28 * 24)
    rows = pd.DataFrame({"energy_kwh": y, "date": y.index.strftime("%Y-%m-%d"), "hour": y.index.hour})
    hourly_profile = rows.groupby("hour").energy_kwh.agg(mean_kwh="mean", observations="count").reset_index()
    daily = rows.groupby("date").energy_kwh.agg(total_kwh="sum", recorded_hours="count").reset_index()
    daily["complete_day"] = daily.recorded_hours.eq(24)
    heatmap = rows.pivot(index="date", columns="hour", values="energy_kwh").reindex(columns=range(24))
    # Compare the last 7 days (168 hours) with the 7 days before them.
    # This needs at least 14 days (336 hours) of data.
    change = None
    if len(y) >= 336:
        previous, recent = float(y.iloc[-336:-168].sum()), float(y.iloc[-168:].sum())
        change = {"previous_168h_kwh": previous, "recent_168h_kwh": recent, "difference_kwh": recent - previous,
                  "change_percent": (recent / previous - 1) * 100 if previous > 0 else None,
                  "previous_start": y.index[-336], "previous_end": y.index[-169],
                  "recent_start": y.index[-168], "recent_end": y.index[-1]}
    return {"hourly": hourly_profile, "daily": daily, "heatmap": heatmap, "change": change,
            "first_timestamp": y.index[0], "last_timestamp": y.index[-1], "rows": len(y)}


def _forecast_frame(forecast: pd.DataFrame) -> pd.DataFrame:
    """Check that a forecast has 24 valid hourly rows, then return a copy.

    Raises ValueError for the wrong row count, bad timestamps or bad values.
    """
    if not {"target_timestamp", "predicted_kwh"}.issubset(forecast) or len(forecast) != 24:
        raise ValueError("Planning requires exactly 24 computed hourly predictions.")
    frame = forecast.copy()
    times = pd.DatetimeIndex(pd.to_datetime(frame.target_timestamp, errors="raise"))
    if times.hasnans or times.has_duplicates or not times.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Planning timestamps must be valid, ordered and consecutive hours.")
    if not ((times.minute == 0) & (times.second == 0) & (times.microsecond == 0) & (times.nanosecond == 0)).all():
        raise ValueError("Planning timestamps must label whole-hour intervals.")
    values = frame.predicted_kwh.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Planning energy must be finite and non-negative.")
    frame["target_timestamp"] = times
    return frame


def energy_costs(forecast: pd.DataFrame, off_peak_rate: float, *, peak_rate: float | None = None,
                 peak_start: int = 17, peak_end: int = 22) -> pd.DataFrame:
    """Energy-only scenario cost in the explicit forecast-label clock.

    Peak start is inclusive and peak end exclusive; a start after end wraps
    midnight. Equal boundaries are rejected rather than guessed to mean 24h.
    """
    # The user types the prices. We only check they are sensible numbers.
    frame = _forecast_frame(forecast)
    rates = [off_peak_rate] + ([] if peak_rate is None else [peak_rate])
    if any(isinstance(rate, bool) or not isinstance(rate, (int, float, np.number))
           or not np.isfinite(rate) or rate < 0 for rate in rates):
        raise ValueError("Supply finite, non-negative rates in currency per kWh.")
    peak = np.zeros(24, dtype=bool)
    if peak_rate is not None:
        if any(isinstance(hour, bool) or not isinstance(hour, (int, np.integer)) or not 0 <= hour <= 23
               for hour in [peak_start, peak_end]) or peak_start == peak_end:
            raise ValueError("Peak start/end must be distinct whole hours from 0 to 23.")
        hours = frame.target_timestamp.dt.hour.to_numpy()
        peak = ((hours >= peak_start) & (hours < peak_end) if peak_start < peak_end
                else (hours >= peak_start) | (hours < peak_end))
    # Cost of each hour = predicted kWh * price per kWh for that hour.
    frame["peak_tariff_period"] = peak
    frame["tariff_per_kwh"] = np.where(peak, peak_rate if peak_rate is not None else off_peak_rate, off_peak_rate)
    frame["estimated_energy_cost"] = frame.predicted_kwh * frame.tariff_per_kwh
    if not np.isfinite(frame.estimated_energy_cost).all():
        raise ValueError("Scenario cost exceeds the supported numeric range.")
    return frame


def additional_load_windows(costs: pd.DataFrame, energy_kwh: float, duration_hours: int) -> pd.DataFrame:
    """Compare contiguous placements of a hypothetical *additional* even load.

    Example: a 2 kWh job that runs for 2 hours. We try every possible start
    hour in the 24-hour forecast and sort the options from cheapest to most
    expensive. `costs` is the table made by energy_costs().
    """
    frame = _forecast_frame(costs)
    if "tariff_per_kwh" not in frame or not np.isfinite(frame.tariff_per_kwh).all() or frame.tariff_per_kwh.lt(0).any():
        raise ValueError("Calculate valid hourly tariff rates before comparing load windows.")
    if isinstance(energy_kwh, bool) or not np.isfinite(energy_kwh) or energy_kwh < 0:
        raise ValueError("Additional energy must be finite and non-negative kWh.")
    if isinstance(duration_hours, bool) or not isinstance(duration_hours, int) or not 1 <= duration_hours <= 24:
        raise ValueError("Additional load duration must be a whole number from 1 to 24 hours.")
    values = frame.predicted_kwh.to_numpy(dtype=float)
    records = []
    # Every start hour where the whole job still fits inside the 24 hours.
    for start in range(25 - duration_hours):
        # The extra energy is spread evenly over the job's hours.
        adjusted = values.copy()
        adjusted[start:start + duration_hours] += energy_kwh / duration_hours
        incremental_cost = float(frame.tariff_per_kwh.iloc[start:start + duration_hours].sum() * energy_kwh / duration_hours)
        records.append({"start_index": start, "start": frame.target_timestamp.iloc[start],
                        "end_exclusive": frame.target_timestamp.iloc[start + duration_hours - 1] + pd.Timedelta(hours=1),
                        "additional_energy_kwh": float(energy_kwh), "duration_hours": duration_hours,
                        "incremental_energy_cost": incremental_cost, "peak_with_additional_load_kwh": float(adjusted.max())})
    result = pd.DataFrame(records).sort_values(["incremental_energy_cost", "start_index"], kind="stable").reset_index(drop=True)
    if not np.isfinite(result[["incremental_energy_cost", "peak_with_additional_load_kwh"]]).all().all():
        raise ValueError("Additional-load scenario exceeds the supported numeric range.")
    return result
