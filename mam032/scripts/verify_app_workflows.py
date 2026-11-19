#!/usr/bin/env python3
"""Exercise actual Streamlit flows with both locally trained production bundles.

This is an optional, offline integration check after preparing the two datasets.
It uses Streamlit AppTest (not a visual browser) and captures the actual download
payload. Offset-labeled uploads are test scenarios, not new accuracy experiments.
No models, datasets, benchmark scores, or assignment reports are modified.
"""

from __future__ import annotations

import argparse
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import DATASETS
from src.input_contract import INPUT_CONTRACT


def widget(widgets, label: str):
    return next(item for item in widgets if item.label == label)


def healthy(app: AppTest) -> None:
    if app.exception:
        raise AssertionError("Streamlit exception: " + str(app.exception[0].message))
    if app.error:
        raise AssertionError("Streamlit error: " + app.error[0].value)


def inspect_forecast(app: AppTest, download, dataset_id: str, model: str, scenario: str, *, generate: bool = True) -> dict:
    if generate:
        widget(app.button, "Generate 24-hour forecast").click().run()
    healthy(app)
    assert len(app.metric) == 4, "Forecast cards were not rendered."
    result = app.session_state["forecast_result"]["data"]
    assert len(result) == 24
    assert result["horizon"].tolist() == list(range(1, 25))
    assert result["dataset_id"].eq(dataset_id).all()
    assert result["model"].eq(model).all()
    assert np.isfinite(result["predicted_kwh"]).all()
    assert (result["predicted_kwh"] >= 0).all()
    assert result["target_timestamp"].iloc[0] == result["forecast_origin"].iloc[0] + pd.Timedelta(hours=1)
    assert result["forecast_issued_at"].eq(result["target_timestamp"].iloc[0]).all()
    assert result["target_timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all()
    payload = next(
        call.args[1] for call in reversed(download.call_args_list)
        if call.args and call.args[0] == "Download forecast CSV"
    )
    exported = pd.read_csv(io.BytesIO(payload))
    assert len(exported) == 24
    assert exported["horizon"].tolist() == list(range(1, 25))
    np.testing.assert_allclose(exported["predicted_kwh"], result["predicted_kwh"], rtol=1e-12, atol=1e-12)
    for column in ["dataset_id", "model", "timestamp_clock", "timestamp_input_clock", "timestamp_normalization", "artifact_release_id"]:
        assert exported[column].tolist() == result[column].tolist(), f"Export lost {column}."
    for column in ["forecast_origin", "target_timestamp", "forecast_issued_at"]:
        parsed = pd.to_datetime(exported[column])
        pd.testing.assert_series_equal(parsed, result[column], check_names=False)
    before = result.copy()
    app.run()
    healthy(app)
    pd.testing.assert_frame_equal(before, app.session_state["forecast_result"]["data"])
    assert len(app.metric) == 4, "A rerun lost the saved forecast."
    navigation = None
    if scenario == "built-in / Historical backtest" and model == "HistGradientBoosting":
        for page in ["Overview", "Data Hub", "Model Lab", "Help", "Forecast Studio"]:
            widget(app.radio, "Workspace").set_value(page).run()
            healthy(app)
            pd.testing.assert_frame_equal(before, app.session_state["forecast_result"]["data"])
            if page == "Model Lab":
                diagnostic_path = DATASETS[dataset_id].artifact_dir / "diagnostics/test_event_metrics.csv"
                if diagnostic_path.exists():
                    expected = pd.read_csv(diagnostic_path)
                    payload = next(call.args[1] for call in reversed(download.call_args_list)
                                   if call.args and call.args[0] == "Download high-demand diagnostics CSV")
                    exported_diagnostics = pd.read_csv(io.BytesIO(payload))
                    pd.testing.assert_frame_equal(expected, exported_diagnostics)
                    assert any("Reserved test targets" in item.value for item in app.caption)
                    assert any("not evidence of uniform performance" in item.value for item in app.info)
        assert widget(app.radio, "Forecast view").value == "Historical backtest"
        assert len(app.metric) == 4
        navigation = "All five screens passed; computed result and historical controls preserved."
    return {
        "dataset_id": dataset_id,
        "scenario": scenario,
        "model": model,
        "forecast_origin": str(result["forecast_origin"].iloc[0]),
        "first_target": str(result["target_timestamp"].iloc[0]),
        "last_target": str(result["target_timestamp"].iloc[-1]),
        "timestamp_clock": result["timestamp_clock"].iloc[0],
        "artifact_release_id": result["artifact_release_id"].iloc[0],
        "rows": len(result),
        "total_predicted_kwh": float(result["predicted_kwh"].sum()),
        "observed_targets": int(result["actual_kwh"].notna().sum()),
        "download_and_rerun": "passed",
        "five_screen_journey": navigation,
    }


def verify() -> list[dict]:
    records = []
    models = ["Seasonal Naive", "Ridge", "HistGradientBoosting"]
    st.cache_data.clear()
    st.cache_resource.clear()
    for dataset_id, spec in DATASETS.items():
        for view in ["Historical backtest", "Future forecast"]:
            with patch("streamlit.download_button", wraps=st.download_button) as download:
                app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
                healthy(app)
                widget(app.radio, "Workspace").set_value("Forecast Studio").run()
                widget(app.selectbox, "Benchmark dataset").set_value(dataset_id).run()
                widget(app.radio, "Forecast view").set_value(view).run()
                for model in models:
                    widget(app.selectbox, "Forecast model").set_value(model).run()
                    healthy(app)
                    record = inspect_forecast(app, download, dataset_id, model, f"built-in / {view}")
                    assert record["observed_targets"] == (24 if view == "Historical backtest" else 0)
                    records.append(record)
        # Use authentic energy values with explicit, test-only offset labels.
        # No claim about transfer accuracy is made for these clock exercises.
        excerpt = pd.read_csv(spec.demo_path).head(INPUT_CONTRACT["minimum_history_rows"])
        for offset in ["UTC", "Etc/GMT-6"]:
            upload = excerpt.copy()
            upload["timestamp"] = pd.to_datetime(upload["timestamp"]).dt.tz_localize(offset)
            csv_bytes = upload.to_csv(index=False).encode()
            with patch("streamlit.file_uploader", side_effect=lambda *args, **kwargs: io.BytesIO(csv_bytes)), patch(
                "streamlit.download_button", wraps=st.download_button
            ) as download:
                app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
                widget(app.radio, "Workspace").set_value("Forecast Studio").run()
                widget(app.selectbox, "Benchmark dataset").set_value(dataset_id).run()
                widget(app.radio, "Data source").set_value("Upload hourly CSV").run()
                healthy(app)
                assert any("accuracy on this upload is unverified" in warning.value for warning in app.warning)
                expected_origin = upload["timestamp"].iloc[-1]
                if spec.input_clock_policy == "utc":
                    expected_origin = expected_origin.tz_convert("UTC").tz_localize(None)
                for model in models:
                    widget(app.selectbox, "Forecast model").set_value(model).run()
                    record = inspect_forecast(app, download, dataset_id, model, f"upload clock exercise / {offset}")
                    assert pd.Timestamp(record["forecast_origin"]) == expected_origin
                    assert record["observed_targets"] == 0
                    records.append(record)
        print(f"PASS [{dataset_id}]: 12 real-bundle app flows, downloads and reruns.")
    for starting_dataset in DATASETS:
        with patch("streamlit.download_button", wraps=st.download_button) as download:
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            widget(app.selectbox, "Benchmark dataset").set_value(starting_dataset).run()
            widget(app.radio, "Data source").set_value("Upload hourly CSV").run()
            widget(app.button, "Run the 24-hour forecast").click().run()
            healthy(app)
            assert widget(app.radio, "Forecast view").value == "Historical backtest"
            assert any("Guided demo computed" in item.value for item in app.success)
            record = inspect_forecast(app, download, "xky_2025", "HistGradientBoosting",
                                      f"guided demo from {starting_dataset} / empty upload", generate=False)
            assert record["observed_targets"] == 24
            records.append(record)
    print("PASS: guided demos reset both source selections and compute real predictions.")
    st.cache_data.clear()
    st.cache_resource.clear()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON evidence path.")
    args = parser.parse_args()
    records = verify()
    evidence = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Streamlit AppTest with actual local production bundles; not a visual browser test.",
        "scope": "Clock-labeled uploads are integration exercises, not new accuracy evaluations.",
        "input_contract_version": INPUT_CONTRACT["version"],
        "flow_count": len(records),
        "flows": records,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"PASS: {len(records)} real-bundle forecast / CSV-download / rerun flows.")


if __name__ == "__main__":
    main()
