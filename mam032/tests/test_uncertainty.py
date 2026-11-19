"""The simple baseline and the 80% range must be honest and leak-free."""

import numpy as np
import pandas as pd

from src.uncertainty import add_interval, interval_offsets, seasonal_median_forecast
from src.visualization import forecast_figure


def _hourly(days: int = 20) -> pd.Series:
    index = pd.date_range("2025-01-01", periods=days * 24, freq="h")
    return pd.Series(np.tile(np.arange(24) / 10.0, days), index=index, name="energy_kwh")


def test_baseline_is_median_of_same_hour_on_last_seven_days():
    hourly = _hourly()
    origin = hourly.index[-25]  # leaves 24 future hours after the origin
    result = seasonal_median_forecast(hourly, origin)
    assert len(result) == 24
    # Every day has the same pattern, so the median equals the hour-of-day value.
    expected = (pd.DatetimeIndex(result["target_timestamp"]).hour / 10.0).to_numpy()
    np.testing.assert_allclose(result["baseline_kwh"], expected)


def test_baseline_never_uses_values_after_the_origin():
    hourly = _hourly()
    origin = hourly.index[-25]
    before = seasonal_median_forecast(hourly, origin)
    changed = hourly.copy()
    changed[changed.index > origin] = 999.0  # change the future
    after = seasonal_median_forecast(changed, origin)
    pd.testing.assert_frame_equal(before, after)


def _validation(rows_per_block: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    times = pd.date_range("2025-01-01", periods=24 * rows_per_block // 6 * 4, freq="h")
    predicted = np.full(len(times), 1.0)
    return pd.DataFrame({"target_timestamp": times, "predicted_kwh": predicted,
                         "actual_kwh": predicted + rng.normal(0, 0.2, len(times)), "model": "Ridge"})


def test_interval_offsets_are_ordered_and_cover_four_blocks():
    offsets = interval_offsets(_validation(), "Ridge")
    assert offsets is not None and set(offsets) == {0, 1, 2, 3}
    assert all(low < high for low, high in offsets.values())


def test_interval_is_none_without_enough_rows_or_model():
    assert interval_offsets(_validation().head(10), "Ridge") is None
    assert interval_offsets(_validation(), "Missing model") is None


def test_add_interval_keeps_order_and_never_goes_below_zero():
    forecast = pd.DataFrame({"target_timestamp": pd.date_range("2025-03-01", periods=24, freq="h"),
                             "horizon": range(1, 25), "predicted_kwh": [0.05] * 24})
    result = add_interval(forecast, interval_offsets(_validation(), "Ridge"))
    assert (result["lower_kwh"] >= 0).all()
    assert (result["lower_kwh"] <= result["upper_kwh"]).all()


def test_chart_shows_range_and_rule_only_when_columns_exist():
    forecast = pd.DataFrame({"target_timestamp": pd.date_range("2025-03-01", periods=24, freq="h"),
                             "horizon": range(1, 25), "predicted_kwh": [0.5] * 24})
    assert "Likely range (80%)" not in [t.name for t in forecast_figure(forecast, False).data]
    forecast["lower_kwh"], forecast["upper_kwh"], forecast["baseline_kwh"] = 0.3, 0.8, 0.4
    names = [t.name for t in forecast_figure(forecast, False).data]
    assert "Likely range (80%)" in names and "Simple rule (7-day median)" in names
