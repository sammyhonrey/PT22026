#!/usr/bin/env python3
"""Verify new decision-support journeys with both actual production bundles.

Uses Streamlit AppTest, not browser visual testing. Offset labels and tariff
rates are explicit integration-test inputs, not new measurements or real prices.
"""

import argparse
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.verify_app_workflows import healthy, widget
from src.data import validate_hourly_csv
from src.datasets import DATASETS
from src.decision_support import consumption_profile
from src.forecasting import forecast_24h, load_model_bundle


def download_data(downloads, label):
    payload = next(call.args[1] for call in reversed(downloads.call_args_list) if call.args and call.args[0] == label)
    return pd.read_csv(io.BytesIO(payload))


def verify() -> list[dict]:
    records = []
    for dataset_id, spec in DATASETS.items():
        for mode in ["built-in", "offset-labeled upload exercise"]:
            excerpt = pd.read_csv(spec.demo_path)
            uploaded = excerpt.copy()
            uploaded["timestamp"] = pd.to_datetime(uploaded.timestamp).dt.tz_localize("Etc/GMT-6")
            content = uploaded.to_csv(index=False).encode()
            with patch("streamlit.file_uploader", side_effect=lambda *args, **kwargs: io.BytesIO(content)), patch(
                "streamlit.download_button", wraps=st.download_button
            ) as downloads:
                started = time.perf_counter()
                app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
                widget(app.selectbox, "Benchmark dataset").set_value(dataset_id).run()
                if mode != "built-in":
                    widget(app.radio, "Data source").set_value("Upload hourly CSV").run()
                source = validate_hourly_csv(spec.demo_path if mode == "built-in" else io.BytesIO(content), clock_policy=spec.input_clock_policy)
                widget(app.radio, "Workspace").set_value("Data Hub").run()
                healthy(app)
                profile = download_data(downloads, "Download consumption profile CSV")
                expected_profile = consumption_profile(source)
                np.testing.assert_allclose(profile.mean_kwh, expected_profile["hourly"].mean_kwh)
                assert profile.observations.sum() == expected_profile["rows"]
                widget(app.radio, "Workspace").set_value("Model Lab").run()
                widget(app.number_input, "Recorded days to evaluate").set_value(3).run()
                widget(app.button, "Evaluate recorded days").click().run()
                healthy(app)
                replay = app.session_state["personal_replay"]["data"]
                predictions = download_data(downloads, "Download replay predictions CSV")
                metrics = download_data(downloads, "Download replay metrics CSV")
                assert len(predictions) == 216 and len(metrics) == 3
                assert predictions.dataset_id.eq(dataset_id).all()
                assert predictions.timestamp_clock.eq(source.attrs["timestamp_clock"]).all()
                assert predictions.timestamp_input_clock.eq(source.attrs["timestamp_input_clock"]).all()
                bundle = load_model_bundle(spec.artifact_dir / "model_bundle.joblib", dataset_id)
                for (model, origin), rows in replay["predictions"].groupby(["model", "forecast_origin"]):
                    expected = forecast_24h(source, origin, bundle, model)
                    np.testing.assert_allclose(rows.predicted_kwh, expected.predicted_kwh, rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(rows.actual_kwh, expected.actual_kwh, rtol=1e-12, atol=1e-12)
                widget(app.radio, "Workspace").set_value("Forecast Studio").run()
                widget(app.radio, "Forecast view").set_value("Future forecast").run()
                widget(app.button, "Generate 24-hour forecast").click().run()
                healthy(app)
                assert widget(app.button, "Calculate planning scenario").disabled
                widget(app.radio, "Tariff scenario").set_value("Peak / off-peak").run()
                widget(app.number_input, "Base / off-peak rate per kWh").set_value(1.5).run()
                widget(app.number_input, "Peak rate per kWh").set_value(4.25).run()
                widget(app.number_input, "Peak period starts (inclusive hour)").set_value(22).run()
                widget(app.number_input, "Peak period ends (exclusive hour)").set_value(6).run()
                widget(app.number_input, "Hypothetical additional energy (kWh)").set_value(2.8).run()
                widget(app.button, "Calculate planning scenario").click().run()
                healthy(app)
                cost = download_data(downloads, "Download hourly cost scenario CSV")
                windows = download_data(downloads, "Download additional-load windows CSV")
                np.testing.assert_allclose(cost.estimated_energy_cost, cost.predicted_kwh * cost.tariff_per_kwh)
                assert len(cost) == 24 and len(windows) == 23
                assert cost.peak_tariff_period.sum() == 8
                assert windows.dataset_id.eq(dataset_id).all()
                for row in windows.itertuples():
                    expected_cost = cost.tariff_per_kwh.iloc[row.start_index:row.start_index + 2].sum() * 1.4
                    assert np.isclose(row.incremental_energy_cost, expected_cost)
                assert windows.incremental_energy_cost.is_monotonic_increasing
                before = app.session_state["forecast_result"]["data"].copy()
                widget(app.radio, "Workspace").set_value("Data Hub").run()
                widget(app.radio, "Workspace").set_value("Forecast Studio").run()
                healthy(app)
                assert widget(app.number_input, "Base / off-peak rate per kWh").value == 1.5
                pd.testing.assert_frame_equal(before, app.session_state["forecast_result"]["data"])
                records.append({"dataset_id": dataset_id, "input_mode": mode, "profile_rows": expected_profile["rows"],
                                "replay_origins": [str(origin) for origin in replay["origins"]], "replay_prediction_rows": len(predictions),
                                "all_replay_predictions_match_shared_inference": True,
                                "replay_metrics": metrics[["model", "mae_kwh", "rmse_kwh", "bias_kwh"]].to_dict("records"),
                                "scenario_hourly_rows": len(cost), "additional_load_windows": len(windows),
                                "clock": source.attrs["timestamp_clock"], "artifact_release_id": cost.artifact_release_id.iloc[0],
                                "energy_only_scenario_cost": float(cost.estimated_energy_cost.sum()),
                                "rates_are_integration_test_inputs": True, "roundtrip_controls_and_forecast": "passed",
                                "journey_seconds": round(time.perf_counter() - started, 3)})
                print(f"PASS [{dataset_id} / {mode}]: profile, 216 replay predictions, energy cost, 23 added-load windows, downloads and preserved state.")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = verify()
    evidence = {"verified_at_utc": datetime.now(timezone.utc).isoformat(),
                "method": "Streamlit AppTest with actual fitted bundles; not visual browser testing.",
                "scope": "Offset labels and tariff rates are test inputs, not new household measurements or real market tariffs. Replay errors are not new held-out benchmark scores.",
                "journeys": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"PASS: {len(results)} real-bundle decision-support journeys.")
