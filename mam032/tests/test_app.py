"""Exercise the production UI using a tiny synthetic, test-only model bundle.

No production model file, full UCI download, or network connection is required.
Real UCI experiment results remain in the separately tracked evaluation CSVs.
"""

from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_series_equal
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from streamlit.testing.v1 import AppTest

from src.features import FEATURE_COLUMNS, build_supervised_dataset
from src.forecasting import ModelArtifactError
from src.datasets import DATASETS

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def synthetic_test_bundles() -> dict[str, dict]:
    """Fit separate tiny test-only households; never use them for report scores."""
    index = pd.date_range("2000-01-01", periods=24 * 12, freq="h")
    bundles = {}
    for dataset_id, baseline in [("xky_2025", 1.4), ("uci_household", 0.8)]:
        values = baseline + 0.2 * np.sin(2 * np.pi * index.hour / 24)
        dataset = build_supervised_dataset(pd.DataFrame({"energy_kwh": values}, index=index))
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        hgb = HistGradientBoostingRegressor(
            max_iter=4, max_leaf_nodes=4, min_samples_leaf=5,
            early_stopping=False, random_state=42,
        )
        for model in (ridge, hgb):
            model.fit(dataset[FEATURE_COLUMNS], dataset["target_kwh"])
        bundles[dataset_id] = {
            "dataset_id": dataset_id,
            "models": {"Ridge": ridge, "HistGradientBoosting": hgb},
            "selected_model": "HistGradientBoosting" if dataset_id == "xky_2025" else "Ridge",
            "feature_columns": FEATURE_COLUMNS,
            "peak_threshold_kwh": float(np.quantile(values, 0.9)),
            "trained_through": str(dataset["target_timestamp"].max()),
            "test_only": True,
        }
    return bundles


@pytest.fixture(scope="module")
def synthetic_app_context(synthetic_test_bundles, tmp_path_factory):
    """Write only synthetic test input/score fixtures into pytest's temporary area."""
    fixture_root = tmp_path_factory.mktemp("synthetic_app_datasets")
    registry = {}
    for dataset_id, first_date, offset in [
        ("xky_2025", "2025-01-01", 0.1),
        ("uci_household", "2010-01-01", 0.3),
    ]:
        base = fixture_root / dataset_id
        base.mkdir()
        (base / "model_bundle.joblib").write_bytes(b"synthetic-test-only-model-A")
        (base / "training_metadata.json").write_text(json.dumps({"dataset_id": dataset_id}))
        demo_path = base / "synthetic_test_only_demo.csv"
        index = pd.date_range(first_date, periods=24 * 12, freq="h")
        pd.DataFrame({"timestamp": index, "energy_kwh": 0.8 + offset}).to_csv(demo_path, index=False)
        pd.DataFrame({
            "model": ["Seasonal Naive", "Ridge", "HistGradientBoosting"],
            "mae_kwh": [offset + 0.2, offset + 0.1, offset],
            "rmse_kwh": [offset + 0.4, offset + 0.3, offset + 0.2],
            "r2": [0.1, 0.2, 0.3],
        }).to_csv(base / "test_metrics.csv", index=False)
        pd.DataFrame({"feature": ["horizon"], "importance_mean": [offset], "importance_std": [0.01]}).to_csv(
            base / "permutation_importance.csv", index=False
        )
        registry[dataset_id] = replace(DATASETS[dataset_id], artifact_dir=base, demo_path=demo_path)
    return registry, synthetic_test_bundles


@pytest.fixture(autouse=True)
def mock_local_model_artifact(synthetic_app_context):
    """Keep AppTest runnable from a clean checkout with no ignored joblib file."""
    import streamlit as st

    registry, bundles = synthetic_app_context

    def load_synthetic_bundle(path, expected_dataset_id=None, expected_sha256=None):
        assert expected_dataset_id in bundles
        assert path == registry[expected_dataset_id].artifact_dir / "model_bundle.joblib"
        return bundles[expected_dataset_id]

    st.cache_resource.clear()
    st.cache_data.clear()
    with patch("src.datasets.DATASETS", registry), patch(
        "src.forecasting.load_model_bundle", side_effect=load_synthetic_bundle
    ):
        yield
    st.cache_resource.clear()
    st.cache_data.clear()


def _app(page: str = "Forecast Studio") -> AppTest:
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    return _widget(app.radio, "Workspace").set_value(page).run()


def _widget(widgets, label: str):
    return next(widget for widget in widgets if widget.label == label)


def _upload_csv(rows: int, *, negative: bool = False) -> bytes:
    index = pd.date_range("2024-01-01", periods=rows, freq="h")
    values = [-1.0 if negative else 0.8] * rows
    return pd.DataFrame({"timestamp": index, "energy_kwh": values}).to_csv(index=False).encode()


@pytest.mark.parametrize("page", ["Forecast Studio", "Model Lab", "Data Hub"])
def test_each_rendered_chart_has_a_semantic_data_alternative(page):
    app = _app(page)
    if page == "Forecast Studio":
        _widget(app.button, "Generate 24-hour forecast").click().run()
    assert not app.exception
    chart_count = len(app.get("plotly_chart"))
    alternatives = [item.value for item in app.markdown if 'class="ew-chart-data"' in item.value]
    assert chart_count > 0 and len(alternatives) == chart_count
    assert all('<caption>' in markup and '<th scope="col">' in markup for markup in alternatives)


def test_recorded_replay_compares_models_exports_and_survives_navigation():
    import streamlit as st
    app = _app("Model Lab")
    _widget(app.number_input, "Recorded days to evaluate").set_value(2).run()
    assert next(panel for panel in app.expander if panel.label.startswith("Evaluate your recorded days")).proto.expanded
    with patch("streamlit.download_button", wraps=st.download_button) as downloads:
        _widget(app.button, "Evaluate recorded days").click().run()
        assert not app.exception
        replay = app.session_state["personal_replay"]["data"]
        assert next(panel for panel in app.expander if panel.label.startswith("Evaluate your recorded days")).proto.expanded
        assert len(replay["predictions"]) == 2 * 24 * 3
        assert set(replay["summary"].model) == {"Seasonal Naive", "Ridge", "HistGradientBoosting"}
        payload = next(call.args[1] for call in downloads.call_args_list if call.args[0] == "Download replay predictions CSV")
        exported = pd.read_csv(io.BytesIO(payload))
        assert len(exported) == 144 and exported.dataset_id.eq("xky_2025").all()
        assert exported.evaluation_scope.str.contains("not a newly held-out").all()
        _widget(app.radio, "Workspace").set_value("Data Hub").run()
        _widget(app.radio, "Workspace").set_value("Model Lab").run()
        assert not app.exception
        assert _widget(app.number_input, "Recorded days to evaluate").value == 2
        pd.testing.assert_frame_equal(replay["predictions"], app.session_state["personal_replay"]["data"]["predictions"])


def test_recorded_replay_hides_stale_results_and_does_not_fit_missing_targets():
    app = _app("Model Lab")
    _widget(app.number_input, "Recorded days to evaluate").set_value(2).run()
    _widget(app.button, "Evaluate recorded days").click().run()
    _widget(app.number_input, "Recorded days to evaluate").set_value(1).run()
    assert not app.exception
    assert not any(button.label == "Download replay metrics CSV" for button in app.get("download_button"))
    with patch("streamlit.file_uploader", side_effect=lambda *args, **kwargs: io.BytesIO(_upload_csv(169))):
        _widget(app.radio, "Data source").set_value("Upload hourly CSV").run()
        assert not app.exception
        assert any("No complete daily replay" in item.value for item in app.info)
        assert not any(button.label == "Evaluate recorded days" for button in app.button)


def test_cost_planner_uses_entered_rates_and_persists_its_controls():
    app = _app()
    _widget(app.button, "Generate 24-hour forecast").click().run()
    assert _widget(app.button, "Calculate planning scenario").disabled
    _widget(app.selectbox, "Scenario currency").set_value("EUR").run()
    _widget(app.number_input, "Base / off-peak rate per kWh").set_value(2.0).run()
    assert next(panel for panel in app.expander if panel.label == "Energy cost and additional-load planner").proto.expanded
    _widget(app.button, "Calculate planning scenario").click().run()
    assert not app.exception
    result = app.session_state["planning_result"]
    forecast = app.session_state["forecast_result"]["data"]
    assert result["costs"].estimated_energy_cost.sum() == pytest.approx(forecast.predicted_kwh.sum() * 2)
    assert result["costs"].scenario_currency.eq("EUR").all()
    _widget(app.radio, "Workspace").set_value("Data Hub").run()
    _widget(app.radio, "Workspace").set_value("Forecast Studio").run()
    assert not app.exception
    assert _widget(app.number_input, "Base / off-peak rate per kWh").value == 2.0
    assert _widget(app.selectbox, "Scenario currency").value == "EUR"
    _widget(app.number_input, "Base / off-peak rate per kWh").set_value(3.0).run()
    assert not any(button.label == "Download hourly cost scenario CSV" for button in app.get("download_button"))


def test_peak_tariff_planning_handles_midnight_and_invalid_equal_hours():
    app = _app()
    _widget(app.button, "Generate 24-hour forecast").click().run()
    _widget(app.radio, "Tariff scenario").set_value("Peak / off-peak").run()
    _widget(app.number_input, "Base / off-peak rate per kWh").set_value(1.0).run()
    assert _widget(app.button, "Calculate planning scenario").disabled
    _widget(app.number_input, "Peak rate per kWh").set_value(5.0).run()
    _widget(app.number_input, "Peak period starts (inclusive hour)").set_value(22).run()
    _widget(app.number_input, "Peak period ends (exclusive hour)").set_value(6).run()
    _widget(app.button, "Calculate planning scenario").click().run()
    assert not app.exception
    costs = app.session_state["planning_result"]["costs"]
    assert costs.peak_tariff_period.sum() == 8
    _widget(app.number_input, "Peak period ends (exclusive hour)").set_value(22).run()
    _widget(app.button, "Calculate planning scenario").click().run()
    assert not app.exception and any("distinct whole hours" in item.value for item in app.error)
    assert "planning_result" not in app.session_state


def test_consumption_profile_uses_input_and_keeps_original_row_inspection_first():
    app = _app("Data Hub")
    assert not app.exception
    assert len(app.dataframe[0].value) == 288
    assert len(app.dataframe[1].value) == 12
    assert app.dataframe[1].value.total_kwh.sum() == pytest.approx(.9 * 288)
    assert any("At least 336" in item.value for item in app.info)
    heatmap = json.loads(app.get("plotly_chart")[2].proto.spec)["data"][0]
    assert heatmap["type"] == "heatmap"
    assert heatmap["hoverongaps"] is False


def test_guided_demo_computes_once_and_preserves_real_result():
    from src.forecasting import forecast_24h
    app = _app("Overview")
    assert "forecast_result" not in app.session_state
    with patch("src.views.forecast_24h", wraps=forecast_24h) as calculate:
        _widget(app.button, "Run the 24-hour forecast").click().run()
        assert not app.exception
        assert calculate.call_count == 1
        assert _widget(app.radio, "Workspace").value == "Forecast Studio"
        assert _widget(app.radio, "Forecast view").value == "Historical backtest"
        result = app.session_state["forecast_result"]["data"].copy()
        assert len(result) == 24 and result.actual_kwh.notna().all()
        assert result.dataset_id.eq("xky_2025").all()
        assert result.model.eq("HistGradientBoosting").all()
        assert result.forecast_origin.dt.hour.eq(23).all()
        assert any("Guided demo computed" in item.value for item in app.success)
        assert any(item.label == "Understand these results" and item.proto.expanded for item in app.expander)
        app.run()
        assert not app.exception and calculate.call_count == 1
        pd.testing.assert_frame_equal(result, app.session_state["forecast_result"]["data"])


def _has_first_time_hint(app) -> bool:
    return any("First time here" in item.value for item in app.caption)


def test_first_time_hint_is_shown_before_a_forecast_and_removed_after_it():
    app = _app()
    assert "forecast_result" not in app.session_state
    # Before any forecast, the beginner hint must be visible.
    assert _has_first_time_hint(app)
    _widget(app.button, "Generate 24-hour forecast").click().run()
    assert not app.exception
    assert len(app.session_state["forecast_result"]["data"]) == 24
    # In the same run that shows the result, the old hint must be gone.
    assert not _has_first_time_hint(app)
    assert len(app.metric) == 4
    # The hint stays hidden on later reruns, because a forecast already exists.
    app.run()
    assert not _has_first_time_hint(app)


def test_guided_demo_does_not_show_the_first_time_hint():
    app = _app("Overview")
    _widget(app.button, "Run the 24-hour forecast").click().run()
    assert not app.exception
    assert any("Guided demo computed" in item.value for item in app.success)
    assert not _has_first_time_hint(app)


def test_first_time_hint_is_not_hidden_by_visiting_other_pages():
    # Visiting other pages without a forecast must not hide the hint.
    app = _app("Data Hub")
    _widget(app.radio, "Workspace").set_value("Forecast Studio").run()
    assert not app.exception
    assert _has_first_time_hint(app)


def test_guided_demo_has_no_session_state_default_warning():
    import streamlit.elements.lib.policies as policies

    with patch.object(policies, "_shown_default_value_warning", False):
        app = _app("Overview")
        _widget(app.button, "Run the 24-hour forecast").click().run()
        assert not app.exception
        assert _widget(app.selectbox, "Forecast model").value == "HistGradientBoosting"
        assert not any("created with a default value" in item.value for item in app.warning)


def test_guided_demo_resets_other_benchmark_upload_and_future_view():
    app = _app()
    _widget(app.selectbox, "Benchmark dataset").set_value("uci_household").run()
    _widget(app.radio, "Forecast view").set_value("Future forecast").run()
    _widget(app.radio, "Data source").set_value("Upload hourly CSV").run()
    _widget(app.radio, "Workspace").set_value("Overview").run()
    _widget(app.button, "Run the 24-hour forecast").click().run()
    assert not app.exception
    assert _widget(app.selectbox, "Benchmark dataset").value == "xky_2025"
    assert _widget(app.radio, "Data source").value == "Built-in dataset demo"
    assert _widget(app.radio, "Forecast view").value == "Historical backtest"
    assert app.session_state["forecast_result"]["data"].actual_kwh.notna().all()


def test_corrupt_optional_diagnostics_does_not_break_forecasting(synthetic_app_context):
    registry, _ = synthetic_app_context
    path = registry["xky_2025"].artifact_dir / "diagnostics"
    path.mkdir(exist_ok=True)
    manifest = path / "diagnostics_manifest.json"
    manifest.write_text("[]")
    try:
        app = _app("Model Lab")
        assert not app.exception
        assert any("reliability evidence is not available" in item.value for item in app.info)
        _widget(app.radio, "Workspace").set_value("Overview").run()
        _widget(app.button, "Run the 24-hour forecast").click().run()
        assert not app.exception and len(app.session_state["forecast_result"]["data"]) == 24
    finally:
        manifest.unlink()


@pytest.mark.parametrize("fault", ["missing_column", "missing_model", "invalid_fold"])
def test_hash_valid_bad_diagnostics_recovers_in_model_lab_and_forecast(synthetic_app_context, diagnostic_payload, fault):
    from src.artifacts import read_release
    from src.diagnostics import DISPLAY_FILES, write_diagnostics

    registry, _ = synthetic_app_context
    spec = registry["xky_2025"]
    release = read_release(spec.artifact_dir, spec.id)
    folder = spec.artifact_dir / "diagnostics"
    frames, folds = diagnostic_payload
    folds["dataset_id"] = spec.id
    if fault == "missing_column":
        frames["test_event_metrics.csv"] = frames["test_event_metrics.csv"].drop(columns="precision")
    elif fault == "missing_model":
        frames["test_event_metrics.csv"] = frames["test_event_metrics.csv"].iloc[:-1]
    else:
        folds["folds"][0]["train_origins"] = []
    write_diagnostics(folder, spec.id, release.release_id, frames, folds, {})
    try:
        app = _app("Model Lab")
        assert not app.exception
        assert any("missing, outdated or invalid" in item.value for item in app.info)
        assert len(app.dataframe[0].value) == 3  # Original benchmark is still available.
        _widget(app.radio, "Workspace").set_value("Overview").run()
        _widget(app.button, "Run the 24-hour forecast").click().run()
        assert not app.exception
        assert len(app.session_state["forecast_result"]["data"]) == 24
    finally:
        for name in [*DISPLAY_FILES, "diagnostics_manifest.json"]:
            (folder / name).unlink(missing_ok=True)


def test_builtin_historical_forecast_survives_rerun() -> None:
    app = _app()
    assert not app.exception
    _widget(app.button, "Generate 24-hour forecast").click().run()
    assert not app.exception
    assert len(app.metric) == 4
    result = app.session_state["forecast_result"]["data"]
    assert len(result) == 24
    assert result["actual_kwh"].notna().all()
    assert "Observed kWh" in app.dataframe[0].value.columns
    app.run()  # Downloading results and other widget actions also rerun Streamlit.
    assert not app.exception
    assert len(app.metric) == 4
    assert len(app.dataframe[0].value) == 24


def test_future_forecast_has_no_observed_targets() -> None:
    app = _app()
    _widget(app.radio, "Forecast view").set_value("Future forecast").run()
    _widget(app.button, "Generate 24-hour forecast").click().run()
    assert not app.exception
    result = app.session_state["forecast_result"]["data"]
    assert len(result) == 24
    assert result["actual_kwh"].isna().all()
    assert "Observed kWh" not in app.dataframe[0].value.columns
    assert result["target_timestamp"].min() > result["forecast_origin"].max()


def test_upload_with_exact_minimum_history_can_forecast() -> None:
    csv_bytes = _upload_csv(169)
    with patch("streamlit.file_uploader", side_effect=lambda *args, **kwargs: io.BytesIO(csv_bytes)):
        app = _app()
        _widget(app.radio, "Data source").set_value("Upload hourly CSV").run()
        assert not app.exception
        assert _widget(app.radio, "Forecast view").options == ["Future forecast"]
        _widget(app.button, "Generate 24-hour forecast").click().run()
        assert not app.exception
        assert len(app.session_state["forecast_result"]["data"]) == 24


def test_invalid_upload_shows_actionable_error() -> None:
    csv_bytes = _upload_csv(169, negative=True)
    with patch("streamlit.file_uploader", side_effect=lambda *args, **kwargs: io.BytesIO(csv_bytes)):
        app = _app()
        _widget(app.radio, "Data source").set_value("Upload hourly CSV").run()
        assert not app.exception
        assert "non-negative" in app.error[0].value
        assert not app.metric


def test_model_change_does_not_display_stale_forecast() -> None:
    app = _app()
    _widget(app.button, "Generate 24-hour forecast").click().run()
    assert len(app.metric) == 4
    _widget(app.selectbox, "Forecast model").set_value("Seasonal Naive").run()
    assert not app.exception
    assert not app.metric
    assert any("Inputs have changed" in caption.value for caption in app.caption)


def test_forecast_download_contains_the_computed_24_horizons() -> None:
    import streamlit as st

    with patch("streamlit.download_button", wraps=st.download_button) as download:
        app = _app()
        _widget(app.button, "Generate 24-hour forecast").click().run()
        assert not app.exception
        forecast_download = next(
            call for call in download.call_args_list
            if call.args and call.args[0] == "Download forecast CSV"
        )
        downloaded = pd.read_csv(
            io.BytesIO(forecast_download.args[1]),
            parse_dates=["forecast_origin", "target_timestamp"],
        )
        computed = app.session_state["forecast_result"]["data"]
        assert downloaded["horizon"].tolist() == list(range(1, 25))
        assert_series_equal(downloaded["predicted_kwh"], computed["predicted_kwh"], check_exact=False)
        assert_series_equal(downloaded["target_timestamp"], computed["target_timestamp"])
        assert downloaded["dataset_id"].eq("xky_2025").all()
        assert downloaded["timestamp_clock"].eq("UTC").all()
        assert downloaded["timestamp_input_clock"].notna().all()
        assert downloaded["timestamp_normalization"].notna().all()
        assert downloaded["artifact_release_id"].eq(computed["artifact_release_id"].iloc[0]).all()
        assert forecast_download.args[2:] == ("ecowatt_xky_2025_24h_forecast.csv", "text/csv")


def test_dataset_selector_isolates_models_metrics_and_demo_inputs() -> None:
    app = _app()
    assert _widget(app.selectbox, "Benchmark dataset").value == "xky_2025"
    assert _widget(app.selectbox, "Forecast model").value == "HistGradientBoosting"
    _widget(app.button, "Generate 24-hour forecast").click().run()
    recent = app.session_state["forecast_result"]["data"]
    assert recent["dataset_id"].eq("xky_2025").all()
    assert recent["target_timestamp"].dt.year.eq(2025).all()
    assert _widget(app.metric, "XKY 2025 test MAE").value == "0.100 kWh"

    _widget(app.selectbox, "Benchmark dataset").set_value("uci_household").run()
    assert not app.exception
    assert not app.metric  # Never show the recent household's saved forecast.
    assert _widget(app.selectbox, "Forecast model").value == "Ridge"
    _widget(app.button, "Generate 24-hour forecast").click().run()
    classic = app.session_state["forecast_result"]["data"]
    assert classic["dataset_id"].eq("uci_household").all()
    assert classic["target_timestamp"].dt.year.eq(2010).all()
    assert _widget(app.metric, "UCI 2006–2010 test MAE").value == "0.400 kWh"

    _widget(app.selectbox, "Benchmark dataset").set_value("xky_2025").run()
    _widget(app.button, "Generate 24-hour forecast").click().run()
    assert not app.exception
    assert app.session_state["forecast_result"]["data"]["dataset_id"].eq("xky_2025").all()
    assert _widget(app.metric, "XKY 2025 test MAE").value == "0.100 kWh"


def test_upload_warns_that_benchmark_score_is_not_upload_accuracy() -> None:
    csv_bytes = _upload_csv(169)
    with patch("streamlit.file_uploader", side_effect=lambda *args, **kwargs: io.BytesIO(csv_bytes)):
        app = _app()
        _widget(app.radio, "Data source").set_value("Upload hourly CSV").run()
        assert not app.exception
        assert any("accuracy on this upload is unverified" in warning.value for warning in app.warning)


def test_missing_model_dependency_produces_training_instructions() -> None:
    import streamlit as st

    st.cache_resource.clear()
    try:
        with patch("src.forecasting.load_model_bundle", side_effect=ModelArtifactError("Model artifact missing")):
            app = _app()
            assert not app.exception
            assert "Model artifact missing" in app.error[0].value
            assert "train_models.py" in app.code[0].value
            assert "prepare_xky_data.py" in app.code[0].value
            assert "--dataset xky_2025" in app.code[0].value
    finally:
        st.cache_resource.clear()


@pytest.mark.parametrize("dataset_id", ["xky_2025", "uci_household"])
@pytest.mark.parametrize("timezone", ["UTC", "Etc/GMT-6"])
def test_timezone_aware_upload_preserves_instant_and_exports_clock_metadata(
    dataset_id: str, timezone: str
) -> None:
    """Accepted offsets must survive the complete UI-to-download forecast path."""
    import streamlit as st

    index = pd.date_range("2024-01-01", periods=169, freq="h", tz=timezone)
    csv_bytes = pd.DataFrame({"timestamp": index, "energy_kwh": 0.8}).to_csv(index=False).encode()
    with patch("streamlit.file_uploader", side_effect=lambda *args, **kwargs: io.BytesIO(csv_bytes)), patch(
        "streamlit.download_button", wraps=st.download_button
    ) as download:
        app = _app()
        if dataset_id != "xky_2025":
            _widget(app.selectbox, "Benchmark dataset").set_value(dataset_id).run()
        _widget(app.radio, "Data source").set_value("Upload hourly CSV").run()
        assert not app.exception
        _widget(app.button, "Generate 24-hour forecast").click().run()
        assert not app.exception

        result = app.session_state["forecast_result"]["data"]
        expected = index[-1] + pd.Timedelta(hours=1)
        if dataset_id == "xky_2025":
            expected = expected.tz_convert("UTC").tz_localize(None)
            assert result["target_timestamp"].dt.tz is None
            assert result["timestamp_clock"].eq("UTC").all()
        else:
            assert result["target_timestamp"].dt.tz is not None
            assert pd.Timestamp(result.iloc[0]["target_timestamp"]).utcoffset() == index[-1].utcoffset()
        assert result.iloc[0]["target_timestamp"] == expected
        assert result["horizon"].tolist() == list(range(1, 25))
        assert result["actual_kwh"].isna().all()
        assert result["dataset_id"].eq(dataset_id).all()

        forecast_download = next(
            call for call in download.call_args_list
            if call.args and call.args[0] == "Download forecast CSV"
        )
        downloaded = pd.read_csv(io.BytesIO(forecast_download.args[1]))
        for field in ["dataset_id", "timestamp_clock", "timestamp_input_clock", "timestamp_normalization"]:
            assert downloaded[field].notna().all()
            assert downloaded[field].tolist() == result[field].tolist()
        visible_text = " ".join(element.value for element in [*app.markdown, *app.caption])
        assert result.iloc[0]["timestamp_clock"] in visible_text


@pytest.mark.parametrize("dataset_id", ["xky_2025", "uci_household"])
def test_uploaded_household_before_benchmark_cutoff_is_not_labeled_in_sample(dataset_id: str) -> None:
    """Benchmark dates cannot determine membership for an unrelated upload."""
    index = pd.date_range("1999-01-01", periods=193, freq="h")
    csv_bytes = pd.DataFrame({"timestamp": index, "energy_kwh": 0.8}).to_csv(index=False).encode()
    with patch("streamlit.file_uploader", side_effect=lambda *args, **kwargs: io.BytesIO(csv_bytes)):
        app = _app()
        if dataset_id != "xky_2025":
            _widget(app.selectbox, "Benchmark dataset").set_value(dataset_id).run()
        _widget(app.radio, "Data source").set_value("Upload hourly CSV").run()
        assert not app.exception
        assert _widget(app.radio, "Forecast view").value == "Historical backtest"
        _widget(app.button, "Generate 24-hour forecast").click().run()
        assert not app.exception
        assert not any("in-sample replay" in message.value for message in app.warning)
        assert any("accuracy on this upload is unverified" in message.value for message in app.warning)
        assert app.session_state["forecast_result"]["data"]["actual_kwh"].notna().all()


@pytest.mark.parametrize("forecast_model", ["Ridge", "Seasonal Naive"])
def test_feature_sensitivity_names_its_analysis_model_when_forecast_model_differs(forecast_model: str) -> None:
    """Saved HGB importance must not be presented as Ridge/Naive explanation."""
    app = _app("Model Lab")
    _widget(app.selectbox, "Forecast model").set_value(forecast_model).run()
    assert not app.exception
    assert any("HistGradientBoosting · validation feature sensitivity" in item.value for item in app.markdown)
    assert any(
        "Analysis model: HistGradientBoosting" in item.value
        and f"selected forecast model is {forecast_model}" in item.value
        for item in app.caption
    )
    assert any(
        "chart below describes HistGradientBoosting" in item.value and forecast_model in item.value
        for item in app.info
    )


def test_offset_label_change_clears_stale_forecast_and_refreshes_export_provenance() -> None:
    """Equal UTC histories may have different upload clock provenance."""
    utc_index = pd.date_range("2024-01-01", periods=169, freq="h", tz="UTC")
    current = {
        "csv": pd.DataFrame({"timestamp": utc_index, "energy_kwh": 0.8}).to_csv(index=False).encode()
    }
    with patch("streamlit.file_uploader", side_effect=lambda *args, **kwargs: io.BytesIO(current["csv"])):
        app = _app()
        _widget(app.radio, "Data source").set_value("Upload hourly CSV").run()
        _widget(app.button, "Generate 24-hour forecast").click().run()
        assert not app.exception
        first = app.session_state["forecast_result"]["data"].copy()
        assert len(app.metric) == 4

        # These labels express exactly the same instants and consumption values.
        current["csv"] = pd.DataFrame({
            "timestamp": utc_index.tz_convert("Etc/GMT-6"), "energy_kwh": 0.8,
        }).to_csv(index=False).encode()
        app.run()
        assert not app.exception
        assert not app.metric
        assert any("Inputs have changed" in item.value for item in app.caption)

        _widget(app.button, "Generate 24-hour forecast").click().run()
        assert not app.exception
        second = app.session_state["forecast_result"]["data"]
        assert len(app.metric) == 4
        assert_series_equal(first["target_timestamp"], second["target_timestamp"])
        assert_series_equal(first["predicted_kwh"], second["predicted_kwh"])
        assert first["timestamp_clock"].eq("UTC").all()
        assert second["timestamp_clock"].eq("UTC").all()
        assert first.iloc[0]["timestamp_input_clock"] != second.iloc[0]["timestamp_input_clock"]
        assert "+06:00" in second.iloc[0]["timestamp_input_clock"]


def test_overview_is_the_entry_screen_and_opens_working_forecast_studio() -> None:
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert _widget(app.radio, "Workspace").value == "Overview"
    _widget(app.button, "Open Forecast Studio").click().run()
    assert _widget(app.radio, "Workspace").value == "Forecast Studio"
    _widget(app.button, "Generate 24-hour forecast").click().run()
    assert not app.exception
    assert len(app.session_state["forecast_result"]["data"]) == 24


def test_overview_shows_measured_history_without_running_forecast(synthetic_app_context):
    """The cinematic page must never present measured input as a prediction."""
    from src.forecasting import forecast_24h
    registry, _ = synthetic_app_context
    with patch("src.views.forecast_24h", wraps=forecast_24h) as calculate:
        app = _app("Overview")
        expected = pd.read_csv(registry["xky_2025"].demo_path).tail(24)
        # The opening draws the measured hours as the hour ribbon, never a forecast.
        assert any("These are measurements, not predictions" in item.value for item in app.caption)
        peak = _widget(app.metric, "Busiest recorded hour")
        assert peak.value == f"{expected.energy_kwh.max():.2f} kWh"
        assert _widget(app.metric, "Energy in those 24 hours").value == f"{expected.energy_kwh.sum():.2f} kWh"
        assert calculate.call_count == 0
        assert "forecast_result" not in app.session_state


def test_masthead_routes_and_overview_evidence_follows_benchmark():
    app = _app("Overview")
    _widget(app.selectbox, "Benchmark dataset").set_value("uci_household").run()
    assert not app.exception
    assert any("Classic benchmark" in item.value for item in list(app.markdown) + list(app.caption))
    _widget(app.button, "Menu").click().run()
    # The menu button now uses the full screen name, the same as the sidebar.
    _widget(app.button, "Model Lab").click().run()
    assert _widget(app.radio, "Workspace").value == "Model Lab"
    _widget(app.button, "Menu").click().run()
    _widget(app.button, "Overview").click().run()
    assert not app.exception
    assert _widget(app.radio, "Workspace").value == "Overview"
    assert _widget(app.selectbox, "Benchmark dataset").value == "uci_household"


@pytest.mark.parametrize("view", ["Historical backtest", "Future forecast"])
def test_navigation_preserves_origin_view_and_computed_result(view: str) -> None:
    app = _app()
    _widget(app.radio, "Forecast view").set_value(view).run()
    if view == "Historical backtest":
        _widget(app.selectbox, "Historical forecast origin (last known hour)").select_index(0).run()
    _widget(app.button, "Generate 24-hour forecast").click().run()
    before = app.session_state["forecast_result"]["data"].copy()
    for page in ["Overview", "Data Hub", "Model Lab", "Help", "Forecast Studio"]:
        _widget(app.radio, "Workspace").set_value(page).run()
        assert not app.exception, page
        pd.testing.assert_frame_equal(before, app.session_state["forecast_result"]["data"])
    assert _widget(app.radio, "Forecast view").value == view
    assert len(app.metric) == 4
    if view == "Historical backtest":
        assert _widget(app.selectbox, "Historical forecast origin (last known hour)").value == before.forecast_origin.iloc[0]
    assert app.dataframe[0].value["Forecast kWh"].tolist() == before.predicted_kwh.tolist()


def test_model_change_on_another_screen_cannot_restore_stale_output() -> None:
    app = _app()
    _widget(app.button, "Generate 24-hour forecast").click().run()
    _widget(app.radio, "Workspace").set_value("Model Lab").run()
    _widget(app.selectbox, "Forecast model").set_value("Seasonal Naive").run()
    _widget(app.radio, "Workspace").set_value("Overview").run()
    assert any("Inputs have changed" in caption.value for caption in app.caption)
    _widget(app.radio, "Workspace").set_value("Forecast Studio").run()
    assert not app.exception
    assert not app.metric
    _widget(app.button, "Generate 24-hour forecast").click().run()
    assert app.session_state["forecast_result"]["data"].model.eq("Seasonal Naive").all()


def test_invalid_upload_does_not_block_help_or_model_lab_and_can_recover() -> None:
    with patch("streamlit.file_uploader", side_effect=lambda *args, **kwargs: io.BytesIO(_upload_csv(169, negative=True))):
        app = _app("Data Hub")
        _widget(app.radio, "Data source").set_value("Upload hourly CSV").run()
        assert "non-negative" in app.error[0].value
        for page in ["Help", "Model Lab"]:
            _widget(app.radio, "Workspace").set_value(page).run()
            assert not app.exception
            assert not app.error
        _widget(app.radio, "Workspace").set_value("Help").run()
        assert any("Why are the dates older than today?" == item.label for item in app.expander)
        _widget(app.radio, "Data source").set_value("Built-in dataset demo").run()
        _widget(app.button, "Try Forecast Studio").click().run()
        _widget(app.button, "Generate 24-hour forecast").click().run()
        assert not app.exception
        assert not app.error
        assert len(app.metric) == 4


def test_missing_upload_does_not_block_other_screens() -> None:
    with patch("streamlit.file_uploader", return_value=None):
        app = _app("Data Hub")
        _widget(app.radio, "Data source").set_value("Upload hourly CSV").run()
        assert not app.exception
        assert any("Choose a CSV" in item.value for item in app.info)
        _widget(app.radio, "Workspace").set_value("Help").run()
        assert any("Start here" in item.label for item in app.expander)
        _widget(app.radio, "Workspace").set_value("Model Lab").run()
        assert len(app.dataframe[0].value) == 3


def test_appearance_and_replay_preserve_computed_predictions() -> None:
    app = _app()
    _widget(app.radio, "Forecast view").set_value("Future forecast").run()
    _widget(app.button, "Generate 24-hour forecast").click().run()
    before = app.session_state["forecast_result"]["data"].copy()
    for palette in ["Ember", "Daylight", "Control Room"]:
        _widget(app.selectbox, "Colour palette").set_value(palette).run()
        assert not app.exception
        assert len(app.metric) == 4
        pd.testing.assert_frame_equal(before, app.session_state["forecast_result"]["data"])
    _widget(app.toggle, "Reduce motion").set_value(True).run()
    _widget(app.button, "Replay page entrance").click().run()
    assert not app.exception
    assert len(app.metric) == 4
    assert _widget(app.radio, "Forecast view").value == "Future forecast"
    pd.testing.assert_frame_equal(before, app.session_state["forecast_result"]["data"])


def test_layout_width_changes_recreate_chart_without_recomputing_or_losing_forecast():
    from src.forecasting import forecast_24h
    width = {"value": 1280}
    with patch("src.ui.layout_width", side_effect=lambda: width["value"]), patch(
        "src.views.forecast_24h", wraps=forecast_24h
    ) as calculate:
        app = _app()
        _widget(app.button, "Generate 24-hour forecast").click().run()
        before = app.session_state["forecast_result"]["data"].copy()
        ids = []
        for value in [1280, 320, 390, 768]:
            width["value"] = value
            app.run()
            assert not app.exception
            ids.append(app.get("plotly_chart")[0].proto.id)
            pd.testing.assert_frame_equal(before, app.session_state["forecast_result"]["data"])
        assert len(set(ids)) == 4
        assert calculate.call_count == 1


def test_data_hub_download_contains_the_validated_rows(synthetic_app_context) -> None:
    import streamlit as st

    with patch("streamlit.download_button", wraps=st.download_button) as download:
        app = _app("Data Hub")
        assert not app.exception
        call = next(call for call in download.call_args_list if call.args[0] == "Download validated input CSV")
        exported = pd.read_csv(io.BytesIO(call.args[1]))
        registry, _ = synthetic_app_context
        expected = pd.read_csv(registry["xky_2025"].demo_path)
        pd.testing.assert_series_equal(exported.energy_kwh, expected.energy_kwh)
        assert exported.timestamp.tolist() == expected.timestamp.tolist()
        assert exported.timestamp_clock.eq("UTC").all()


def test_uploaded_energy_changes_real_forecast_after_visiting_data_hub() -> None:
    current = {"csv": _upload_csv(169)}
    with patch("streamlit.file_uploader", side_effect=lambda *args, **kwargs: io.BytesIO(current["csv"])):
        app = _app("Data Hub")
        _widget(app.radio, "Data source").set_value("Upload hourly CSV").run()
        _widget(app.selectbox, "Forecast model").set_value("Seasonal Naive").run()
        _widget(app.button, "Forecast with this data").click().run()
        _widget(app.button, "Generate 24-hour forecast").click().run()
        assert app.session_state["forecast_result"]["data"].predicted_kwh.eq(0.8).all()
        _widget(app.radio, "Workspace").set_value("Data Hub").run()
        changed = pd.read_csv(io.BytesIO(current["csv"]))
        changed["energy_kwh"] = 2.3
        current["csv"] = changed.to_csv(index=False).encode()
        app.run()
        _widget(app.button, "Forecast with this data").click().run()
        assert not app.metric
        _widget(app.button, "Generate 24-hour forecast").click().run()
        assert not app.exception
        assert app.session_state["forecast_result"]["data"].predicted_kwh.eq(2.3).all()


def test_help_remains_reachable_when_model_is_missing() -> None:
    import streamlit as st

    st.cache_resource.clear()
    with patch("src.forecasting.load_model_bundle", side_effect=ModelArtifactError("Model artifact missing")):
        app = _app("Help")
        assert not app.exception
        assert any("Missing data or model files" == item.label for item in app.expander)
        assert "train_models.py" in app.code[0].value


def test_missing_metrics_do_not_prevent_inference_or_navigation(synthetic_app_context) -> None:
    registry, _ = synthetic_app_context
    metrics_path = registry["xky_2025"].artifact_dir / "test_metrics.csv"
    original = metrics_path.read_bytes()
    metrics_path.unlink()
    try:
        app = _app()
        assert not app.exception
        _widget(app.button, "Generate 24-hour forecast").click().run()
        assert not app.exception
        assert len(app.session_state["forecast_result"]["data"]) == 24
        assert _widget(app.metric, "XKY 2025 test MAE").value == "Unavailable"
        _widget(app.radio, "Workspace").set_value("Model Lab").run()
        assert not app.exception
        assert any("test_metrics.csv" in item.value for item in app.warning)
    finally:
        metrics_path.write_bytes(original)


def test_model_replacement_under_same_filename_refreshes_cache_and_forecast(synthetic_app_context) -> None:
    registry, bundles = synthetic_app_context
    model_path = registry["xky_2025"].artifact_dir / "model_bundle.joblib"
    original = model_path.read_bytes()
    active = {"bundle": bundles["xky_2025"]}

    def load_current(path, expected_dataset_id=None, expected_sha256=None):
        return active["bundle"]

    try:
        with patch("src.forecasting.load_model_bundle", side_effect=load_current) as loader:
            app = _app()
            _widget(app.button, "Generate 24-hour forecast").click().run()
            before = app.session_state["forecast_result"]["data"].copy()
            active["bundle"] = {**bundles["xky_2025"], "peak_threshold_kwh": 0.0}
            model_path.write_bytes(original.replace(b"model-A", b"model-B"))
            app.run()
            assert not app.exception
            assert not app.metric
            assert loader.call_count == 2
            _widget(app.button, "Generate 24-hour forecast").click().run()
            after = app.session_state["forecast_result"]["data"]
            assert after.is_high_demand.all()
            assert not before.is_high_demand.all()
            assert before.artifact_release_id.iloc[0] != after.artifact_release_id.iloc[0]
    finally:
        model_path.write_bytes(original)


def test_metrics_replacement_refreshes_display_and_invalidates_old_result(synthetic_app_context) -> None:
    registry, _ = synthetic_app_context
    path = registry["xky_2025"].artifact_dir / "test_metrics.csv"
    original = path.read_bytes()
    try:
        app = _app()
        _widget(app.button, "Generate 24-hour forecast").click().run()
        changed = pd.read_csv(path)
        changed.loc[changed.model.eq("HistGradientBoosting"), "mae_kwh"] = 0.123
        changed.to_csv(path, index=False)
        app.run()
        assert not app.metric
        _widget(app.button, "Generate 24-hour forecast").click().run()
        assert not app.exception
        assert _widget(app.metric, "XKY 2025 test MAE").value == "0.123 kWh"
        _widget(app.radio, "Workspace").set_value("Model Lab").run()
        assert app.dataframe[0].value.loc[lambda frame: frame.model.eq("HistGradientBoosting"), "mae_kwh"].iloc[0] == 0.123
    finally:
        path.write_bytes(original)


def test_manifest_mismatch_blocks_mixed_release_but_preserves_help(synthetic_app_context) -> None:
    from src.artifacts import capture_digests, publish_manifest

    registry, _ = synthetic_app_context
    folder = registry["xky_2025"].artifact_dir
    path = folder / "test_metrics.csv"
    original = path.read_bytes()
    try:
        publish_manifest(folder, "xky_2025", capture_digests(folder))
        app = _app()
        _widget(app.button, "Generate 24-hour forecast").click().run()
        path.write_bytes(original + b"\n")
        app.run()
        assert not app.exception
        assert not app.metric
        assert any("do not match their manifest" in item.value for item in app.error)
        assert not any(item.label == "Generate 24-hour forecast" for item in app.button)
        _widget(app.radio, "Workspace").set_value("Help").run()
        assert not app.exception
        assert any("Start here" in item.label for item in app.expander)
    finally:
        path.write_bytes(original)
        (folder / "release_manifest.json").unlink(missing_ok=True)


def test_importance_replacement_refreshes_the_rendered_evidence(synthetic_app_context) -> None:
    import base64

    registry, _ = synthetic_app_context
    path = registry["xky_2025"].artifact_dir / "permutation_importance.csv"
    original = path.read_bytes()

    def rendered_mean(app):
        values = json.loads(app.get("plotly_chart")[1].proto.spec)["data"][0]["x"]
        return np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"])[0]

    try:
        app = _app("Model Lab")
        assert rendered_mean(app) == pytest.approx(0.1)
        changed = pd.read_csv(path)
        changed["importance_mean"] = 0.8
        changed.to_csv(path, index=False)
        app.run()
        assert not app.exception
        assert rendered_mean(app) == pytest.approx(0.8)
    finally:
        path.write_bytes(original)
