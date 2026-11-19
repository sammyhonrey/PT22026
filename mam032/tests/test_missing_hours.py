import numpy as np
import pandas as pd

from src.data import minute_power_to_hourly


def test_55_to_59_minutes_are_estimated() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=55, freq="min"),
            "global_active_power_kw": [1.5] * 55,
        }
    )
    row = minute_power_to_hourly(frame).iloc[0]
    assert row["energy_kwh"] == 1.5
    assert bool(row["estimated"])


def test_below_55_minutes_is_insufficient() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=54, freq="min"),
            "global_active_power_kw": [1.5] * 54,
        }
    )
    row = minute_power_to_hourly(frame).iloc[0]
    assert np.isnan(row["energy_kwh"])
    assert row["quality"] == "insufficient"
