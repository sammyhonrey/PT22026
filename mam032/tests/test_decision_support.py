"""New features must calculate real values, preserve clocks and avoid future inputs."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.decision_support import additional_load_windows, consumption_profile, energy_costs, replay_models, replay_origins
from src.forecasting import ModelArtifactError


class PastOnlyModel:
    def __init__(self, offset=0):
        self.offset = offset

    def predict(self, features):
        return features.target_lag_24.to_numpy() + self.offset

    def fit(self, *args, **kwargs):
        raise AssertionError("Replay must never fit or replace a reference model.")


@pytest.fixture
def replay_bundle():
    return {"models": {"Ridge": PastOnlyModel(.1), "HistGradientBoosting": PastOnlyModel(.2)}}


def forecast():
    return pd.DataFrame({"target_timestamp": pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC"),
                         "predicted_kwh": np.arange(1, 25) / 10})


def test_replay_compares_identical_complete_nonoverlapping_targets(hourly_series, replay_bundle):
    result = replay_models(hourly_series, replay_bundle, 3)
    data = result["predictions"]
    assert len(data) == 3 * 3 * 24
    assert data.groupby(["model", "forecast_origin"]).size().eq(24).all()
    assert result["origins"].hour.tolist() == [23] * 3
    reference = None
    for _, rows in data.groupby("model"):
        target_keys = rows[["forecast_origin", "target_timestamp", "horizon", "actual_kwh"]].reset_index(drop=True)
        if reference is not None:
            assert_frame_equal(reference, target_keys)
        reference = target_keys
        assert not rows.target_timestamp.duplicated().any()
        np.testing.assert_allclose(rows.actual_kwh, hourly_series.loc[rows.target_timestamp, "energy_kwh"])


def test_replay_predictions_at_an_origin_ignore_all_later_observations(hourly_series, replay_bundle):
    before = replay_models(hourly_series, replay_bundle, 2)
    origin = before["origins"][0]
    changed = hourly_series.copy()
    changed.loc[changed.index > origin, "energy_kwh"] = 999
    after = replay_models(changed, replay_bundle, 2)
    first = before["predictions"].loc[lambda frame: frame.forecast_origin.eq(origin)]
    second = after["predictions"].loc[lambda frame: frame.forecast_origin.eq(origin)]
    np.testing.assert_array_equal(first.predicted_kwh, second.predicted_kwh)
    assert not np.array_equal(first.actual_kwh, second.actual_kwh)


def test_zero_baseline_has_undefined_percentage_skill(hourly_series, replay_bundle):
    result = replay_models(hourly_series.assign(energy_kwh=0), replay_bundle, 2)
    assert result["summary"].mae_reduction_vs_naive_pct.isna().all()
    assert result["summary"].loc[lambda frame: frame.model.eq("Ridge"), "bias_kwh"].iloc[0] == pytest.approx(.1)


@pytest.mark.parametrize("days", [0, 15, True, 1.5])
def test_replay_budget_is_bounded(hourly_series, replay_bundle, days):
    with pytest.raises(ValueError):
        replay_models(hourly_series, replay_bundle, days)


def test_future_only_and_missing_models_are_not_fake_comparisons(hourly_series):
    short = hourly_series.iloc[:169]
    assert replay_origins(short).empty
    with pytest.raises(ValueError, match="Not enough"):
        replay_models(short, {"models": {}}, 1)
    with pytest.raises(ModelArtifactError, match="unavailable"):
        replay_models(hourly_series, {"models": {}}, 1)


@pytest.mark.parametrize("fault", ["gap", "duplicate", "negative", "nan", "reverse", "partial_hour"])
def test_invalid_profile_input_is_rejected(hourly_series, fault):
    changed = hourly_series.copy()
    if fault == "gap":
        changed = changed.drop(changed.index[6])
    elif fault == "duplicate":
        changed = pd.concat([changed.iloc[:1], changed])
    elif fault == "reverse":
        changed = changed.iloc[::-1]
    elif fault == "partial_hour":
        changed.index += pd.Timedelta(minutes=30)
    else:
        changed.iloc[0, 0] = -1 if fault == "negative" else np.nan
    with pytest.raises(ValueError):
        consumption_profile(changed)


def test_partial_days_and_blank_heatmap_cells_are_not_zero_filled():
    index = pd.date_range("2026-01-01 06:00", periods=48, freq="h", tz="Etc/GMT-6")
    source = pd.DataFrame({"energy_kwh": 2.0}, index=index)
    result = consumption_profile(source)
    assert result["daily"].recorded_hours.tolist() == [18, 24, 6]
    assert result["daily"].complete_day.tolist() == [False, True, False]
    assert result["daily"].total_kwh.sum() == 96
    assert result["heatmap"].isna().sum().sum() == 24
    assert result["hourly"].observations.eq(2).all()
    assert str(result["first_timestamp"].tz) == "Etc/GMT-6"
    assert result["change"] is None


def test_two_equal_length_recent_blocks_and_zero_denominator():
    index = pd.date_range("2026-01-01", periods=336, freq="h")
    result = consumption_profile(pd.DataFrame({"energy_kwh": [0] * 168 + [2] * 168}, index=index))
    assert result["change"]["recent_168h_kwh"] == 336
    assert result["change"]["previous_168h_kwh"] == 0
    assert result["change"]["change_percent"] is None
    assert result["change"]["previous_end"] < result["change"]["recent_start"]


def test_profile_caps_large_inputs_to_672_actual_rows(hourly_series):
    data = consumption_profile(hourly_series)
    assert data["rows"] == 672
    assert data["daily"].total_kwh.sum() == pytest.approx(hourly_series.energy_kwh.tail(672).sum())


def test_flat_cost_is_energy_times_rate_and_does_not_mutate_predictions():
    original = forecast()
    before = original.copy()
    costs = energy_costs(original, 3.5)
    assert costs.estimated_energy_cost.sum() == pytest.approx(original.predicted_kwh.sum() * 3.5)
    assert not costs.peak_tariff_period.any()
    assert_frame_equal(original, before)
    assert str(costs.target_timestamp.dt.tz) == "UTC"


def test_midnight_wrapping_tariff_has_inclusive_start_and_exclusive_end():
    costs = energy_costs(forecast(), 1, peak_rate=4, peak_start=22, peak_end=6)
    assert costs.loc[costs.peak_tariff_period, "target_timestamp"].dt.hour.tolist() == [0, 1, 2, 3, 4, 5, 22, 23]
    assert costs.tariff_per_kwh.iloc[6] == 1
    assert costs.tariff_per_kwh.iloc[22] == 4


@pytest.mark.parametrize("rate,peak,start,end", [(float("nan"), None, 17, 22), (-1, None, 17, 22),
                                               (1, float("inf"), 17, 22), (1, 2, 7, 7), (1, 2, 24, 6),
                                               (True, None, 17, 22)])
def test_invalid_tariffs_are_rejected(rate, peak, start, end):
    with pytest.raises(ValueError):
        energy_costs(forecast(), rate, peak_rate=peak, peak_start=start, peak_end=end)


def test_cheapest_additional_load_window_uses_its_full_duration():
    costs = energy_costs(forecast(), 1, peak_rate=5, peak_start=0, peak_end=8)
    windows = additional_load_windows(costs, 4, 2)
    best = windows.iloc[0]
    assert len(windows) == 23 and best.start_index == 8
    assert best.incremental_energy_cost == 4
    assert best.end_exclusive - best.start == pd.Timedelta(hours=2)
    adjusted = costs.predicted_kwh.to_numpy().copy()
    adjusted[8:10] += 2
    assert best.peak_with_additional_load_kwh == adjusted.max()
    assert windows.loc[windows.start_index.eq(0), "incremental_energy_cost"].iloc[0] == 20


def test_full_day_additional_load_has_one_window_and_zero_rate_is_valid():
    costs = energy_costs(forecast(), 0)
    windows = additional_load_windows(costs, 12, 24)
    assert len(windows) == 1 and windows.incremental_energy_cost.iloc[0] == 0
    assert windows.peak_with_additional_load_kwh.iloc[0] == pytest.approx(2.9)


@pytest.mark.parametrize("energy,duration", [(-1, 2), (float("nan"), 2), (1, 0), (1, 25), (1, True)])
def test_invalid_added_loads_are_rejected(energy, duration):
    with pytest.raises(ValueError):
        additional_load_windows(energy_costs(forecast(), 1), energy, duration)


def test_nonconsecutive_forecast_cannot_be_costed():
    source = forecast()
    source.loc[5, "target_timestamp"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError):
        energy_costs(source, 1)
