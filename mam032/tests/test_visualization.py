"""Chart semantics: plotted quantities remain faithful to computed evidence."""

import numpy as np
import pandas as pd

from src.visualization import forecast_figure, importance_figure, model_comparison_figure


def _forecast() -> pd.DataFrame:
    return pd.DataFrame({
        "target_timestamp": pd.date_range("2025-03-01", periods=24, freq="h"),
        "horizon": range(1, 25),
        "predicted_kwh": [0.4] * 7 + [1.8] + [0.6] * 16,
        "actual_kwh": np.linspace(0.2, 1.5, 24),
        "is_high_demand": [False] * 24,
    })


def test_future_chart_never_displays_observations_even_if_present():
    source = _forecast()
    figure = forecast_figure(source, historical_backtest=False)
    assert "Observed" not in [trace.name for trace in figure.data]
    plotted = next(trace for trace in figure.data if trace.name == "Forecast")
    np.testing.assert_array_equal(plotted.y, source["predicted_kwh"])
    np.testing.assert_array_equal(plotted.x, source["target_timestamp"])


def test_historical_chart_uses_real_observed_values_and_labels_the_mode():
    source = _forecast()
    figure = forecast_figure(source, historical_backtest=True)
    observed = next(trace for trace in figure.data if trace.name == "Observed")
    np.testing.assert_array_equal(observed.y, source["actual_kwh"])
    assert "Historical backtest" in figure.layout.title.text
    assert "kWh" in figure.layout.yaxis.title.text


def test_missing_actual_values_are_not_filled_or_displayed_as_complete():
    source = _forecast()
    source.loc[3, "actual_kwh"] = np.nan
    figure = forecast_figure(source, historical_backtest=True)
    assert "Observed" not in [trace.name for trace in figure.data]


def test_peak_marker_is_an_existing_forecast_point():
    source = _forecast()
    figure = forecast_figure(source, historical_backtest=False)
    peak = next(trace for trace in figure.data if trace.name == "Forecast peak")
    assert list(peak.y) == [1.8]
    assert list(peak.x) == [source.loc[7, "target_timestamp"]]
    assert len(figure.frames) == 0


def test_comparison_orders_lowest_mae_first_and_preserves_full_model_name():
    metrics = pd.DataFrame({
        "model": ["Seasonal Naive", "HistGradientBoosting", "Ridge"],
        "mae_kwh": [0.51, 0.42, 0.46],
    })
    figure = model_comparison_figure(metrics)
    trace = figure.data[0]
    assert list(trace.y) == ["HGB", "Ridge", "Seasonal Naive"]
    np.testing.assert_array_equal(trace.x, [0.42, 0.46, 0.51])
    assert list(trace.customdata[:, 0]) == ["HistGradientBoosting", "Ridge", "Seasonal Naive"]
    assert figure.layout.yaxis.autorange == "reversed"
    assert "kWh" in figure.layout.xaxis.title.text
    assert "lower is better" in figure.layout.title.text


def test_importance_preserves_ranked_means_error_bars_and_raw_feature_hover():
    evidence = pd.DataFrame({
        "feature": ["rolling_mean_24", "target_lag_168", "horizon"],
        "importance_mean": [0.003, 0.034, 0.173],
        "importance_std": [0.001, 0.002, 0.005],
    })
    figure = importance_figure(evidence)
    trace = figure.data[0]
    np.testing.assert_array_equal(trace.x, [0.173, 0.034, 0.003])
    np.testing.assert_array_equal(trace.error_x.array, [0.005, 0.002, 0.001])
    assert list(trace.customdata[:, 0]) == ["horizon", "target_lag_168", "rolling_mean_24"]
    assert list(trace.y) == ["Forecast horizon", "Same hour · previous week", "Daily average"]
    assert "kWh" in figure.layout.xaxis.title.text
