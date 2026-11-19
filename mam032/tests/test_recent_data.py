"""Check cumulative Wh conversion and its hourly availability boundary."""

import numpy as np
import pandas as pd
import pytest
import hashlib
from pandas.testing import assert_frame_equal

from src.data import DataValidationError
from src.recent_data import cumulative_wh_to_hourly, prepare_xky_hourly


def meter():
    return pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=181, freq="min"),
                         "cumulative_wh": np.arange(181) * 10.0})


def test_cumulative_wh_is_differenced_then_divided_by_1000():
    result = cumulative_wh_to_hourly(meter())
    assert result.energy_kwh.tolist() == [0.6, 0.6, 0.6]
    assert result.estimated.all()
    assert (result.measured_span_seconds == 3600).all()


def test_hourly_estimate_uses_no_reading_after_its_end_boundary():
    original = meter()
    mutated = original.copy()
    mutated.loc[mutated.timestamp > pd.Timestamp("2025-01-01 02:00"), "cumulative_wh"] += 100000
    assert_frame_equal(cumulative_wh_to_hourly(original).iloc[:2],
                       cumulative_wh_to_hourly(mutated).iloc[:2])


def test_recent_data_unsorted_rows_do_not_change_result():
    assert_frame_equal(cumulative_wh_to_hourly(meter()),
                       cumulative_wh_to_hourly(meter().sample(frac=1, random_state=42)))


def test_missing_boundary_is_rejected_instead_of_interpolated():
    data = meter().loc[lambda x: ~x.timestamp.between("2025-01-01 00:56", "2025-01-01 01:01")]
    result = cumulative_wh_to_hourly(data)
    assert result.energy_kwh.iloc[:2].isna().all()
    assert result.energy_kwh.iloc[2] == 0.6


def test_early_checkpoints_are_flagged_with_their_age():
    data = meter()
    data.timestamp -= pd.Timedelta(seconds=30)
    result = cumulative_wh_to_hourly(data)
    retained = result.dropna(subset=["energy_kwh"])
    assert (retained.end_checkpoint_age_seconds == 30).all()
    assert (retained.energy_kwh == 0.6).all()


def test_counter_reset_inside_hour_is_rejected_even_with_positive_net_delta():
    data = meter()
    data.loc[30, "cumulative_wh"] = 0
    result = cumulative_wh_to_hourly(data)
    assert pd.isna(result.energy_kwh.iloc[0])
    assert result.energy_kwh.iloc[1] == 0.6


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1])
def test_invalid_cumulative_values_raise_clear_error(bad):
    data = meter()
    data.loc[50, "cumulative_wh"] = bad
    with pytest.raises(DataValidationError):
        cumulative_wh_to_hourly(data)


def test_duplicate_meter_timestamp_rejected():
    data = pd.concat([meter(), meter().iloc[[0]]])
    with pytest.raises(DataValidationError, match="unique"):
        cumulative_wh_to_hourly(data)


def test_conflicting_tariff_counter_cannot_become_hour_boundary_target(tmp_path, monkeypatch):
    """A corrupt large BASE spike must not create artificial boundary demand."""
    import json
    import src.recent_data as recent

    raw = meter().rename(columns={"timestamp": "Time(UTC)", "cumulative_wh": "BASE"})
    raw["HCHC"] = raw.BASE
    raw["HCHP"] = 0.0
    raw.loc[60, "BASE"] = 9999.0  # Independent tariff total remains 600 Wh.
    raw = pd.concat([raw, raw.iloc[[50]]], ignore_index=True)
    source = tmp_path / "test-only-meter.csv"
    raw.to_csv(source, index=False)
    monkeypatch.setattr(recent, "XKY_MD5", hashlib.md5(source.read_bytes()).hexdigest())
    audit_path = tmp_path / "audit.json"
    result = prepare_xky_hourly(source, tmp_path / "hourly.csv", audit_path)
    audit = json.loads(audit_path.read_text())
    assert audit["conflicting_counter_rows_excluded"] == 1
    assert audit["identical_duplicate_timestamps_removed"] == 1
    assert result.energy_kwh.iloc[0] == 0.59  # Safe checkpoint at 00:59.
    assert result.energy_kwh.iloc[1] == 0.61
    assert result.energy_kwh.sum() == pytest.approx(1.8)
