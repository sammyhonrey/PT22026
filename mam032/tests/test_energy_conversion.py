import pandas as pd

from src.data import minute_power_to_hourly


def test_complete_hour_converts_kw_to_kwh() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=60, freq="min"),
            "global_active_power_kw": [2.0] * 60,
        }
    )
    hourly = minute_power_to_hourly(frame)
    assert hourly.iloc[0]["energy_kwh"] == 2.0
    assert hourly.iloc[0]["quality"] == "observed"


def test_variable_minute_power_uses_sum_divided_by_sixty() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=60, freq="min"),
            "global_active_power_kw": list(range(60)),
        }
    )
    assert minute_power_to_hourly(frame).iloc[0]["energy_kwh"] == sum(range(60)) / 60
