"""Five working Streamlit views, sharing one validated input and forecast state."""

from __future__ import annotations

import io
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import ui
from .artifacts import ArtifactReleaseError, verified_bytes
from .evaluation import regression_metrics
from .diagnostics import load_diagnostics
from .forecasting import ModelArtifactError, forecast_24h
from .uncertainty import add_interval, load_interval_offsets, seasonal_median_forecast
from .input_contract import INPUT_CONTRACT
from .visualization import forecast_figure, importance_figure, model_comparison_figure
from .workspace import Workspace, forecast_selection, forecast_key as make_forecast_key, matching_result
from .advanced_views import render_consumption_profile, render_personal_evaluation, render_planning

MIN_HISTORY = INPUT_CONTRACT["minimum_history_rows"]
MIN_BACKTEST = INPUT_CONTRACT["minimum_backtest_rows"]


# About the four cached loaders below: Streamlit saves their result (cache) and
# reuses it, so the files are not read again on every click. The release_id
# argument is not used inside the functions. It is there only so the cache
# loads the file again when the model release changes.
@st.cache_data
def load_importance(path, release_id: str, expected_sha256: str | None) -> pd.DataFrame:
    """Read the feature importance table (which inputs mattered most)."""
    return pd.read_csv(io.BytesIO(verified_bytes(path, expected_sha256)))


@st.cache_data
def cached_interval_offsets(folder: str, model_name: str, release_id: str):
    """Build the 80% range offsets once per model and release (validation days only)."""
    return load_interval_offsets(folder, model_name)


@st.cache_data
def load_baseline_comparison(path: str, release_id: str) -> pd.DataFrame | None:
    """Read the small table made by scripts/run_baseline_check.py, if it exists."""
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return None


@st.cache_data
def load_test_origins(path, release_id: str, expected_sha256: str | None) -> int:
    """Expose the verified experiment's sample size beside its reported errors."""
    # "origins" are the forecast start points. Each test origin is one test day.
    metadata = json.loads(verified_bytes(path, expected_sha256))
    count = metadata["origins"]["test"]
    if type(count) is not int or count < 1:
        raise ValueError("Held-out origin count must be a positive integer.")
    return count


def benchmark_scope(w: Workspace) -> None:
    """Show how many held-out test days were used for the reported scores."""
    try:
        count = load_test_origins(w.spec.artifact_dir / "training_metadata.json", w.release_id,
                                  w.artifact_digests.get("training_metadata.json"))
    except (OSError, ValueError, KeyError, TypeError):
        st.caption("Held-out sample count unavailable for this experiment snapshot. Review the experiment metadata before interpreting its benchmark.")
    else:
        st.caption(f"Evaluation scope: {count} held-out daily origins · {count * 24:,} hourly targets per method. The same target blocks are used for all three methods; a larger sample does not guarantee transfer to another home.")


def source_context(w: Workspace) -> None:
    """Show a short summary strip: benchmark, input dates, model and time clock."""
    if w.hourly is None:
        return
    ui.source_strip([
        ("Reference benchmark", w.spec.short_name),
        ("Current input records", f"{w.hourly.index.min():%d %b %Y} – {w.hourly.index.max():%d %b %Y}"),
        ("Forecast model", w.model_name),
        ("Time reference", w.hourly.attrs["timestamp_clock"]),
    ])
    st.caption(w.hourly.attrs["timestamp_normalization"])
    st.caption("Recorded-data workspace · Forecasts follow the selected last-known record, not today's date. Built-in data is not a live feed.")


def input_ready(w: Workspace) -> bool:
    """Return True if valid hourly data is loaded.

    Otherwise show the reason (an error or a hint) with an example CSV
    download, and return False so the screen can stop early.
    """
    if w.input_error:
        st.error(f"Data validation failed: {w.input_error}")
    elif w.hourly is None:
        st.info(f"Choose a CSV file with at least {MIN_HISTORY} consecutive hourly readings to continue.")
    else:
        return True
    sample_download(w)
    st.caption("Use Data Hub for the schema and recovery steps, or choose Built-in dataset demo in the sidebar.")
    return False


def sample_download(w: Workspace) -> None:
    """Offer a small example CSV that shows the correct file format."""
    try:
        content = w.spec.demo_path.read_bytes()
    except OSError:
        st.warning("The attributed example is unavailable. Run the dataset preparation and demo-excerpt commands in Help.")
        return
    st.download_button("Download valid dataset example CSV", content, f"ecowatt_{w.spec.id}_hourly_example.csv", "text/csv")
    st.caption(f"This attributed {w.spec.short_name} excerpt is a format example. Replace its rows with your own hourly kWh readings.")


def open_page(label: str, page: str, *, key: str, primary: bool = False) -> None:
    """Show a button that moves the user to another screen."""
    st.button(label, on_click=ui.navigate, args=(page,), key=key,
              type="primary" if primary else "secondary", use_container_width=True)


def percentage(value) -> str:
    """Format a ratio like 0.25 as "25.00%". Missing values give "Not defined"."""
    return "Not defined" if pd.isna(value) else f"{float(value):.2%}"


def available_diagnostics(w: Workspace) -> dict | None:
    """Load the reliability tables, or return None if they are missing or invalid."""
    try:
        return load_diagnostics(w.spec.artifact_dir / "diagnostics", w.spec.id, w.release_id)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def render_reliability(w: Workspace) -> None:
    """Show the Reliability part of Model Lab: high-demand detection and daily errors.

    If the tables are not available, show the command that creates them.
    """
    ui.section("Reliability *checks*", "Separate average energy error, high-demand detection and performance across time.")
    diagnostic = available_diagnostics(w)
    if diagnostic is None:
        st.info("Additional reliability evidence is not available for this experiment version. It may be missing, outdated or invalid. The published benchmark above is unchanged; forecasting remains available.")
        st.code(f"python scripts/run_diagnostics.py --dataset {w.spec.id}")
        return
    summary = diagnostic["test_event_metrics.csv"]
    st.markdown("**High-demand detection · unchanged source test set and training-only threshold**")
    st.caption("Precision is the fraction of flagged hours that were high demand. Recall is the fraction of recorded high-demand hours detected. Undefined ratios are left missing when their denominator is zero. These describe reference households, not your upload or operational alerts.")
    display = summary[["model", "test_days", "actual_events", "true_positives", "false_positives", "false_negatives", "precision", "recall"]].rename(columns={
        "model": "Model", "test_days": "Test days", "actual_events": "High-demand hours", "true_positives": "Detected",
        "false_positives": "False alarms", "false_negatives": "Missed", "precision": "Precision", "recall": "Recall"})
    st.dataframe(display.style.format({"Precision": "{:.2%}", "Recall": "{:.2%}"}, na_rep="Not defined"), hide_index=True, use_container_width=True)
    selected = summary.loc[summary.model.eq(w.model_name)].iloc[0]
    st.info(f"{w.model_name}: detected {int(selected.true_positives)} of {int(selected.actual_events)} high-demand hours, missed {int(selected.false_negatives)}, and produced {int(selected.false_positives)} false alarms at {selected.threshold_kwh:.4f} kWh. The threshold was not tuned on these test results.")
    st.download_button("Download high-demand diagnostics CSV", summary.to_csv(index=False).encode(), f"ecowatt_{w.spec.id}_high_demand_diagnostics.csv", "text/csv")
    with st.expander("Daily energy and peak timing errors"):
        st.caption("Daily-total error compares the sum of 24 forecasts with recorded energy. Peak time uses the earliest maximum when values tie and measures absolute timestamp distance within the same forecast window; it is not circular clock-hour distance.")
        st.dataframe(summary[["model", "daily_total_mae_kwh", "mean_peak_magnitude_error_kwh", "mean_peak_time_error_hours"]], hide_index=True, use_container_width=True)
        st.dataframe(diagnostic["test_daily_metrics.csv"], hide_index=True, use_container_width=True)
    fold = diagnostic["fold_manifest.json"]
    rolling = diagnostic["rolling_fold_metrics.csv"]
    st.markdown("**Development-period expanding-window checks**")
    st.caption(f"{fold['development_origins']} eligible development origins; {len(fold['folds'])} expanding folds. Training targets end before evaluation targets. Reserved test targets beginning {fold['reserved_test_first_target']} are excluded. Parameters are fixed; this diagnostic did not replace the published model or choose new settings.")
    figure = go.Figure()
    for model, rows in rolling.groupby("model", sort=True):
        figure.add_trace(go.Scatter(x=rows.fold, y=rows.mae_kwh, mode="lines+markers", name=model,
                                   hovertemplate="Fold %{x}<br>MAE %{y:.4f} kWh<extra>%{fullData.name}</extra>"))
    figure.update_layout(title={"text":"Development folds · MAE", "font":{"size":14}, "x":.04, "xref":"container", "xanchor":"left"}, template="simple_white", height=390,
                         margin={"l":50,"r":20,"t":60,"b":110}, xaxis_title="Expanding fold", yaxis_title="Hourly MAE (kWh)",
                         legend={"orientation":"h","y":-.3,"x":0,"font":{"size":10}})
    figure.update_xaxes(dtick=1)
    ui.chart(figure, w.palette, w.reduce_motion)
    st.dataframe(rolling, hide_index=True, use_container_width=True)
    comparisons = rolling.pivot(index="fold", columns="model", values="mae_kwh")
    if {"HistGradientBoosting", "Seasonal Naive"}.issubset(comparisons.columns):
        wins = int((comparisons.HistGradientBoosting < comparisons["Seasonal Naive"]).sum())
        selected_mean = rolling.loc[rolling.model.eq(w.model_name), "mae_kwh"].mean()
        st.info(f"Development diagnostics: HGB has lower MAE than Seasonal Naive in {wins} of {len(comparisons)} folds. {w.model_name}'s mean fold MAE is {selected_mean:.4f} kWh. A published holdout advantage is not evidence of uniform performance across periods.")
    with st.expander("Fold dates and leakage controls"):
        rows = [{"Fold": row["fold"], "First training origin": row["train_origins"][0],
                 "Last training origin": row["train_origins"][-1], "Last training target": row["last_training_target"],
                 "First evaluation target": row["first_evaluation_target"], "Last evaluation target": row["last_evaluation_target"]}
                for row in fold["folds"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(w.spec.timezone_note)
    with st.expander("Held-out error by forecast horizon"):
        horizon = diagnostic["test_horizon_metrics.csv"]
        figure = go.Figure()
        for model, rows in horizon.groupby("model", sort=True):
            figure.add_trace(go.Scatter(x=rows.horizon, y=rows.mae_kwh, mode="lines+markers", name=model))
        figure.update_layout(title={"text":"Test error by horizon", "font":{"size":14}, "x":.04, "xref":"container", "xanchor":"left"}, template="simple_white", height=390,
                             margin={"l":50,"r":20,"t":60,"b":110}, xaxis_title="Forecast horizon (hours)",
                             yaxis_title="Hourly MAE (kWh)", legend={"orientation":"h","y":-.3,"x":0,"font":{"size":10}})
        ui.chart(figure, w.palette, w.reduce_motion)
        st.dataframe(horizon[["model", "horizon", "mae_kwh", "rmse_kwh"]], hide_index=True, use_container_width=True)
    st.caption("Periods and households differ; lower errors across different sources do not establish a better model. Folds share training history and are not independent replications. XKY's published test still has only 12 days.")
    st.download_button("Download temporal validation CSV", rolling.to_csv(index=False).encode(), f"ecowatt_{w.spec.id}_development_folds.csv", "text/csv")


def render_forecast(w: Workspace) -> None:
    """Screen 2: choose a start hour, run the model, and show the 24-hour result."""
    ui.page_header("Forecast Studio", "Turn validated hourly readings into a real 24-hour energy forecast.")
    guided_demo = st.session_state.pop("guided_demo_requested", False)
    if not input_ready(w):
        return
    if w.bundle is None:
        st.info("Prepare the selected benchmark model using the commands shown above. Data Hub and Help remain available.")
        return
    source_context(w)
    dataset_id, dataset_spec = w.spec.id, w.spec
    source_mode, source_label = w.source_mode, w.source_label
    hourly, bundle, model_name, metrics = w.hourly, w.bundle, w.model_name, w.metrics
    # Remember the user's choices between clicks.
    controls = st.session_state.setdefault("forecast_controls", {})
    # The guided demo picks the latest full day and runs the forecast for the user.
    if guided_demo:
        if len(hourly) < MIN_BACKTEST:
            st.error(f"The guided historical example needs at least {MIN_BACKTEST} rows. Prepare the complete attributed example shown in Help.")
            return
        choices = hourly.index[MIN_HISTORY - 1:-24]
        daily = choices[choices.hour == 23]
        origin = (daily if not daily.empty else choices)[-1]
        controls[w.widget_scope] = {"view": "Historical backtest", "origin": str(origin)}
        for key in [f"view_{w.widget_scope}", f"origin_{w.widget_scope}"]:
            st.session_state.pop(key, None)
    # A historical test needs 24 extra hours, so we can compare with real values.
    can_backtest = len(hourly) >= MIN_BACKTEST
    forecast_views = ["Historical backtest", "Future forecast"] if can_backtest else ["Future forecast"]
    # Section 1: choose the view and the start hour.
    with st.container(key="workspace"):
        ui.section("Set your *forecast*", "Compare predictions against recorded energy, or forecast beyond the last available reading.")
        restored_view, restored_origin, _ = forecast_selection(w, controls)
        forecast_view = st.radio("Forecast view", forecast_views, horizontal=True,
                                 index=forecast_views.index(restored_view), key=f"view_{w.widget_scope}")
        historical_backtest = forecast_view == "Historical backtest"
        if historical_backtest:
            candidate_origins = hourly.index[MIN_HISTORY - 1:-24]
            daily_origins = candidate_origins[candidate_origins.hour == 23]
            if not daily_origins.empty:
                candidate_origins = daily_origins
            origin = st.selectbox(
                "Historical forecast origin (last known hour)",
                options=list(candidate_origins),
                index=candidate_origins.get_loc(restored_origin) if restored_origin in candidate_origins else len(candidate_origins) - 1,
                key=f"origin_{w.widget_scope}",
                format_func=lambda value: pd.Timestamp(value).strftime("%d %b %Y, %H:%M"),
            )
        else:
            origin = hourly.index[-1]
            st.info(
                f"The last known hour is {origin:%d %b %Y, %H:%M}. "
                "The forecast covers the following 24 hours; observed targets are not available."
            )
            if not can_backtest:
                st.caption(f"Historical backtesting becomes available with at least {MIN_BACKTEST} consecutive hourly rows.")

        controls[w.widget_scope] = {"view": forecast_view, "origin": str(origin)}
        st.caption(
            "Timestamps label the start of each hourly interval. The selected origin is the last completed interval; "
            f"this forecast is issued after {(pd.Timestamp(origin) + pd.Timedelta(hours=1)):%d %b %Y, %H:%M}."
        )
        if pd.Timestamp(origin).hour != 23:
            st.warning(
                "Exploratory origin: training and reported backtest metrics use 23:00 origins. "
                "Accuracy at other issue schedules has not been evaluated."
            )
        # A benchmark cutoff classifies only that benchmark's own recorded excerpt.
        # An uploaded household has no known relationship to the training sample.
        trained_through = pd.Timestamp(bundle.get("trained_through", "1900-01-01"))
        if source_mode == "Built-in dataset demo" and pd.Timestamp(origin) < trained_through:
            st.warning(
                "This origin predates the saved model's training cutoff. Its chart is an in-sample replay, "
                "not a held-out historical evaluation."
            )

        forecast_key = make_forecast_key(w, forecast_view, origin)
        # First visit in this session: name the next action before the button.
        # We keep an empty slot for the hint, so we can remove it again as soon as
        # the forecast is computed in this same run. The guided demo needs no hint,
        # because it computes the forecast by itself.
        first_time_hint = st.empty()
        if st.session_state.get("forecast_result") is None and not guided_demo:
            first_time_hint.caption("First time here? The settings above are already valid. Press the Generate 24-hour forecast button to compute a real 24-hour forecast.")
        generate = st.button("Generate 24-hour forecast", type="primary", use_container_width=True)

    with st.expander("About this data · dates, source and units"):
        st.markdown(f"**Benchmark:** {dataset_spec.label}. **Target:** {dataset_spec.target_description}.")
        st.write(dataset_spec.source_note)
        st.caption(dataset_spec.timezone_note)
        if source_mode == "Upload hourly CSV":
            st.caption(f"Upload clock: {hourly.attrs['timestamp_clock']}. {hourly.attrs['timestamp_normalization']}")
        st.caption(f"Input source: {source_label}. {len(hourly):,} consecutive hourly records are loaded.")
        st.markdown(f"[Dataset source and attribution]({dataset_spec.citation_url}) · {dataset_spec.license}")

    # Section 2: run the model when the button is clicked.
    if generate or guided_demo:
        try:
            with st.spinner("Calculating your 24-hour energy outlook…"):
                result = forecast_24h(hourly, pd.Timestamp(origin), bundle, model_name)
        except (ValueError, ModelArtifactError) as exc:
            st.error(f"Forecast could not be generated: {exc}")
            st.stop()

        # Add the data source, clock and release to the result, so the
        # downloaded CSV file says clearly where its numbers come from.
        result["dataset_id"] = dataset_id
        result["timestamp_clock"] = hourly.attrs["timestamp_clock"]
        result["timestamp_input_clock"] = hourly.attrs["timestamp_input_clock"]
        result["timestamp_normalization"] = hourly.attrs["timestamp_normalization"]
        result["artifact_release_id"] = w.release_id
        # Save the result with its key. Later we show it only while the key
        # still matches the current settings.
        st.session_state["forecast_result"] = {"key": forecast_key, "data": result}
        # The forecast exists now, so the "First time here" hint is out of date.
        first_time_hint.empty()
        if guided_demo:
            st.success("Guided demo computed: 24 real predictions from the attributed XKY 2025 example. This is recorded data, not a live meter feed.")

    # Section 3: show the result (numbers, summary, chart, table and download).
    saved_result = st.session_state.get("forecast_result")
    with st.container(key="forecast_results"):
        ui.section("Your energy *outlook*", "A focused view of forecast demand, peak hours and the selected model's benchmark error.")
        if saved_result and saved_result["key"] == forecast_key:
            result = saved_result["data"]

            total = result["predicted_kwh"].sum()
            peak_row = result.loc[result["predicted_kwh"].idxmax()]
            model_metric = metrics.loc[metrics["model"] == model_name] if not metrics.empty else pd.DataFrame()
            # Keep the daily total and its supporting values in one readout row.
            with st.container(key="outlook_metrics"):
                main, c1, c2, c3 = st.columns([1.4, 1, 1, 1])
                with main:
                    with st.container(key="headline_metric"):
                        st.metric("24-hour forecast", f"{total:.2f} kWh", help="Sum of the next 24 predicted hourly energy values.")
                c1.metric("Forecast peak", f"{peak_row['predicted_kwh']:.2f} kWh", help="The highest predicted hourly energy value in this forecast.")
                c2.metric("Peak hour", pd.Timestamp(peak_row["target_timestamp"]).strftime("%H:%M"), help=f"Forecast peak interval starts {pd.Timestamp(peak_row['target_timestamp']):%d %b %Y, %H:%M}.")
                c3.metric(f"{dataset_spec.short_name} test MAE", f"{model_metric.iloc[0]['mae_kwh']:.3f} kWh" if not model_metric.empty else "Unavailable", help="Average absolute error on the benchmark's held-out test set. This is not a measured error on your uploaded data.")
            st.caption("The first three values describe this forecast. Test MAE describes the model's past accuracy on the reference household, not this forecast.")

            # One easy sentence for people who do not read charts.
            plain_words = (
                f"In the next 24 hours, {model_name} expects about {total:.1f} kWh in total. "
                f"The busiest hour is {pd.Timestamp(peak_row['target_timestamp']).strftime('%H:%M')}, "
                f"with about {peak_row['predicted_kwh']:.2f} kWh."
            )
            ui.summary([
                ("In plain words", plain_words),
                ("Forecast window", f"{result['target_timestamp'].min():%d %b, %H:%M} → {result['target_timestamp'].max():%d %b %Y, %H:%M}"),
                ("High-demand estimates", f"{int(result['is_high_demand'].sum())} of 24 hours exceed the training-only threshold of {bundle['peak_threshold_kwh']:.3f} kWh."),
            ])
            with st.expander("Understand these results", expanded=guided_demo):
                st.markdown(
                    "**24-hour forecast:** the sum of 24 predicted hourly energy values, in kWh. It is not a bill estimate.\n\n"
                    "**Forecast peak / peak hour:** the largest predicted hourly value and its interval start. This is a planning estimate, not a guaranteed event or an identified appliance.\n\n"
                    "**Test MAE:** average absolute hourly error on the reference household's held-out test set. It does not measure the accuracy of your uploaded home or give a confidence interval for this forecast.\n\n"
                    f"**Dates and clock:** the window follows your selected completed record, using {hourly.attrs['timestamp_clock']}. Recorded dates are preserved."
                )
            diagnostic = available_diagnostics(w)
            if diagnostic is not None:
                indicator = diagnostic["test_event_metrics.csv"].loc[lambda frame: frame.model.eq(model_name)].iloc[0]
                recall = percentage(indicator.recall)
                st.caption(f"Reference-household high-demand detection ({model_name}): {int(indicator.true_positives)} of {int(indicator.actual_events)} recorded high-demand hours detected ({recall} recall); {int(indicator.false_negatives)} missed and {int(indicator.false_positives)} false alarms. See Model Lab for full scope; this does not validate an uploaded household.")
            # Optional extras: a likely range and a simple comparison rule.
            show_extras = st.checkbox(
                "Show likely range (80%) and a simple comparison rule",
                value=False, key=f"extras_{w.widget_scope}",
                help="The range comes from the errors the model made on validation days. The simple rule is the median of the same hour over the last 7 days.",
            )
            chart_data = result
            offsets = None
            if show_extras:
                offsets = cached_interval_offsets(str(dataset_spec.artifact_dir), model_name, w.release_id)
                # Add the simple-rule line to the table. how="left" keeps all 24 forecast rows.
                chart_data = result.merge(
                    seasonal_median_forecast(hourly, pd.Timestamp(origin))[["target_timestamp", "baseline_kwh"]],
                    on="target_timestamp", how="left",
                )
                if offsets is not None:
                    chart_data = add_interval(chart_data, offsets)
                else:
                    st.caption("The likely range is unavailable because the validation predictions file was not found.")
            ui.chart(forecast_figure(chart_data, historical_backtest=historical_backtest), w.palette, w.reduce_motion)
            if show_extras and offsets is not None:
                risky_hours = int((chart_data["upper_kwh"] >= float(bundle["peak_threshold_kwh"])).sum())
                st.caption(
                    f"About 8 in 10 real hours fell inside this kind of range on the reference household's test days. "
                    f"Careful view: {risky_hours} of 24 hours could reach the high-demand level of "
                    f"{bundle['peak_threshold_kwh']:.3f} kWh if the real value is at the top of its range. "
                    "This will raise more false alarms than the single forecast line. The range is learned from the reference household and is not checked on an uploaded home."
                )
            if dataset_id == "xky_2025" and source_mode == "Built-in dataset demo" and historical_backtest:
                st.caption("XKY comparison targets labeled Observed are recorded counter-boundary estimates, not exact clock-hour integrals.")
            if historical_backtest and result["actual_kwh"].notna().all():
                live_metrics = regression_metrics(result["actual_kwh"], result["predicted_kwh"])
                st.caption(
                    f"This selected historical origin has observed targets: MAE {live_metrics['mae_kwh']:.3f} kWh; "
                    f"RMSE {live_metrics['rmse_kwh']:.3f} kWh. These single-origin values are not the full test score."
                )

            with st.expander("Explore all 24 hourly predictions", expanded=True):
                display_columns = ["Timestamp", "Horizon", "Forecast kWh"]
                if historical_backtest:
                    display_columns.append("Observed kWh")
                display_columns.append("High demand estimate")
                display = result.rename(
                    columns={
                        "target_timestamp": "Timestamp",
                        "horizon": "Horizon",
                        "predicted_kwh": "Forecast kWh",
                        "actual_kwh": "Observed kWh",
                        "is_high_demand": "High demand estimate",
                    }
                )[display_columns]
                st.dataframe(
                    display, hide_index=True, use_container_width=True,
                    column_config={
                        "Forecast kWh": st.column_config.NumberColumn(format="%.3f"),
                        "Observed kWh": st.column_config.NumberColumn(format="%.3f"),
                    },
                )
                st.download_button(
                    "Download forecast CSV",
                    result.to_csv(index=False).encode("utf-8"),
                    f"ecowatt_{dataset_id}_24h_forecast.csv",
                    "text/csv",
                )
            st.caption("High-demand estimates are forecast-based indicators, not guaranteed alerts. Unless you switch on the likely range above, the shaded chart area shows predicted energy, not an uncertainty interval.")
            render_planning(w, result)
        else:
            if saved_result:
                st.caption("Inputs have changed. Generate a forecast to update the results.")
            ui.empty_state(stale=bool(saved_result))


    first, second = st.columns(2)
    with first:
        open_page("Explore model evidence", "Model Lab", key="studio_lab")
    with second:
        open_page("Inspect your data", "Data Hub", key="studio_data")


def render_model_lab(w: Workspace) -> None:
    """Screen 4: compare the three models and explain how they work and where they fail."""
    ui.page_header("Model Lab", "Inspect held-out benchmark results, feature sensitivity and scientific limits.")
    st.caption(f"Experiment version: {w.release_id[:12]} · {'Matched manifest' if w.managed_release else 'Local artifact snapshot'}.")
    benchmark_scope(w)
    dataset_spec, metrics, model_name = w.spec, w.metrics, w.model_name
    if metrics.empty:
        st.warning(w.metrics_error or "Benchmark metrics are unavailable. Prepare the selected dataset and train its models.")
        render_personal_evaluation(w)
        return
    with st.container(key="research_tabs"):
        ui.section("Benchmark *evidence*", "Compare the three methods, see which inputs matter, and read the limits.")
        tab_results, tab_explain, tab_method = st.tabs(["Compare models", "Key inputs", "Method and limits"])
        with tab_results:
            ui.chart(model_comparison_figure(metrics), w.palette, w.reduce_motion)
            st.dataframe(metrics.style.format({"mae_kwh": "{:.4f}", "rmse_kwh": "{:.4f}", "r2": "{:.4f}"}), hide_index=True, use_container_width=True)
            st.caption(f"{dataset_spec.short_name} benchmark only: all methods use identical chronological held-out origins. Lower MAE and RMSE are better. Scores from different households are separate experiments.")
            baseline_table = load_baseline_comparison(str(dataset_spec.artifact_dir / "baseline_comparison.csv"), w.release_id)
            if baseline_table is not None:
                st.markdown("**A fair check against a very simple rule**")
                st.dataframe(
                    baseline_table.rename(columns={
                        "model": "Method", "mae_kwh": "MAE (kWh)", "rmse_kwh": "RMSE (kWh)",
                        "mae_vs_naive": "MAE vs Seasonal Naive", "mae_percent_of_mean": "MAE as % of average use",
                    }).style.format({"MAE (kWh)": "{:.4f}", "RMSE (kWh)": "{:.4f}",
                                     "MAE vs Seasonal Naive": "{:.2f}", "MAE as % of average use": "{:.1f}"}),
                    hide_index=True, use_container_width=True,
                )
                st.caption(
                    "The 7-day same-hour median needs no training. Below 1.00 means better than Seasonal Naive. "
                    "With one household and a small test period, a simple rule can be as good as a trained model. "
                    "That is why we report it openly. MAE as % of average use shows the error size relative to how much energy the home uses."
                )
        with tab_explain:
            st.markdown("**HistGradientBoosting · validation feature sensitivity**")
            st.caption(
                f"Analysis model: HistGradientBoosting, evaluated on the {dataset_spec.short_name} validation set. "
                f"The selected forecast model is {model_name}. This saved analysis always describes HistGradientBoosting."
            )
            if model_name == "Seasonal Naive":
                st.info("Selected-model explanation: Seasonal Naive copies the previous day's same-hour energy. The chart below describes HistGradientBoosting, not Seasonal Naive.")
            elif model_name == "Ridge":
                st.info("Selected-model explanation: Ridge combines standardized features with fitted linear weights. The chart below describes HistGradientBoosting; a Ridge coefficient analysis has not been generated.")
            try:
                importance = load_importance(dataset_spec.artifact_dir / "permutation_importance.csv", w.release_id,
                                             w.artifact_digests.get("permutation_importance.csv"))
                ui.chart(importance_figure(importance), w.palette, w.reduce_motion)
                st.caption(
                    "Permutation importance measures validation-MAE sensitivity over five repeats. Bars show the mean "
                    "MAE increase; error bars show repeat standard deviation. Correlated features can share importance; "
                    "this is not causal evidence. At the evaluated 23:00 origins, horizon and target hour encode the same "
                    "daily position, so their individual ranks do not isolate forecast-distance effects."
                )
            except (OSError, pd.errors.ParserError, ArtifactReleaseError) as exc:
                st.warning(f"Feature sensitivity artifact unavailable: {exc}")
        with tab_method:
            with st.container(key="context_panel"):
                st.markdown(
                    "**Forecast design.** Each row predicts one of the next 24 hours. Consumption-derived features are frozen "
                    "at or before the chosen origin; only target calendar fields and the horizon are known in advance. "
                    "This direct multi-horizon design prevents observed future consumption from entering longer-horizon features."
                )
                st.markdown("**Models.** Seasonal Naive repeats the same hour from the previous day. Ridge is a regularized linear model. HistGradientBoosting combines shallow regression trees to learn nonlinear relationships efficiently.")
                st.markdown(f"**Selected dataset limits.** {dataset_spec.limitations}")
                st.markdown("Weather, occupancy, tariff, and appliance-state inputs are not used by these models. Forecasts support analysis and require validation before operational use.")
            # A plain-language explanation of the three models, for people new to machine learning.
            with st.expander("How the models work (simple explanation)"):
                st.markdown(
                    "**1. Seasonal Naive (the baseline).** It copies the value from the same hour yesterday. "
                    "It needs no training. A real model must beat this simple rule to be useful.\n\n"
                    "**2. Ridge regression.** It gives each of the 16 inputs a weight and adds them up: "
                    "prediction = weight1 x input1 + weight2 x input2 + ... Training finds the weights with the "
                    "smallest error. A small penalty (alpha = 10) keeps the weights small, so the model stays simple. "
                    "It is fast and easy to read, but it can only learn straight-line patterns.\n\n"
                    "**3. HistGradientBoosting.** It builds 220 small decision trees, one after another. Each tree "
                    "asks simple yes/no questions, such as *was the usage 24 hours ago above 0.5 kWh?*, and learns to "
                    "fix the mistakes of the trees before it. The final prediction is the sum of all these small "
                    "corrections. It can learn curved patterns, but it is harder to explain than Ridge."
                )
                st.markdown(
                    "**How training works.** The days are put in time order. The oldest 70% are used to train, the next 15% "
                    "to choose the best model (validation), and the newest 15% to test it once at the end. "
                    "The days are never shuffled, so the model never learns from the future."
                )
                st.markdown(
                    "**What it can do.** Predict the next 24 hours for the household in the data, show how large the "
                    "errors were on unseen days, and help plan when to use heavy appliances.\n\n"
                    "**What it cannot do.** It does not know the weather, who is at home, or new appliances. "
                    "It is trained on one household, so it is not proven for other homes. It can miss rare spikes: "
                    "on the UCI household it found only 4 of 203 high-demand hours."
                )
            # Why the error looks large for one home, in simple words.
            with st.expander("Why is one home hard to predict?"):
                st.markdown(
                    "A big building or a whole city has many people, so ups and downs cancel out and the pattern is smooth. "
                    "One home is different: one person switching on a heater, an oven or a washing machine at a new time "
                    "changes the whole hour. This is why the error as a share of average use (MAE as % of average use in "
                    "the comparison table) is much larger for one home than for a large group of homes.\n\n"
                    "**What this means for you.** Use the forecast to plan, not as a promise. "
                    "The *likely range* in Forecast Studio shows how wrong a forecast can be, and a simple rule can be "
                    "almost as good as a trained model on a short test period.\n\n"
                    "**What could improve it later.** Weather, public holidays, how many people are at home, and more homes "
                    "and more months of data. We did not use these, so we make no claim about them."
                )


    st.download_button("Download benchmark metrics CSV", metrics.to_csv(index=False).encode(),
                       f"ecowatt_{w.spec.id}_test_metrics.csv", "text/csv")
    st.caption("These are source-household benchmark scores. They do not measure the accuracy of an uploaded household.")
    render_reliability(w)
    render_personal_evaluation(w)


def render_overview(w: Workspace) -> None:
    """Screen 1: a full-height opening, then the recorded day, then the evidence."""
    ready = input_ready(w)
    with st.container(key="opening"):
        ui.html('<p class="ew-eyebrow ew-hero-eyebrow">Household electricity forecasting</p>')
        ui.hero_title("A clearer view", "of *tomorrow.*")
        ui.hero_lede("Understand the next 24 hours of a home’s electricity use. "
                     "Explore the forecast, one hour at a time.")
        with st.container(key="opening_action"):
            st.button("Run the 24-hour forecast", on_click=ui.start_guided_demo, key="guided_demo",
                      type="primary", use_container_width=True,
                      help="Selects the XKY 2025 benchmark, HistGradientBoosting and a 23:00 start hour, then computes a real forecast.")
        ui.opening_details()
    ui.home_story()
    if ready:
        with st.container(key="opening_data"):
            ui.section("From hours to *an outlook.*",
                       "One bar for each hour, drawn from the current input or your matching forecast. Exact values remain available below.")
            recent = w.hourly.tail(24)
            saved = matching_result(w, st.session_state.get("forecast_controls", {}),
                                    st.session_state.get("forecast_result"))
            if saved is not None:
                threshold = float(w.bundle["peak_threshold_kwh"]) if w.bundle else None
                ui.hour_ribbon(saved.predicted_kwh, int(pd.Timestamp(saved.target_timestamp.iloc[0]).hour),
                               threshold=threshold, measured=False,
                               caption=f"Forecast hourly energy in kWh for {pd.Timestamp(saved.target_timestamp.iloc[0]):%d %b %Y}, "
                                       f"from {w.model_name}. Totals and exact values are in Forecast Studio.")
            else:
                if st.session_state.get("forecast_result"):
                    st.caption("Inputs have changed, so the saved forecast no longer matches them. "
                               "Run the forecast again in Forecast Studio.")
                ui.recorded_bars(recent)
            model_mae = w.metrics.loc[w.metrics.model.eq(w.model_name), "mae_kwh"]
            ui.spec([
                ("Benchmark", w.spec.label),
                ("Recorded hours loaded", f"{len(w.hourly):,}"),
                ("Forecast model", w.model_name),
                ("Held-out test MAE", f"{model_mae.iloc[0]:.3f} kWh" if not model_mae.empty else "Unavailable"),
            ])
            st.caption("Recorded measurements, not a live meter feed. Test MAE is the model's past error on the "
                       "reference household's held-out days; it is not a guarantee for any other home.")
    render_overview_detail(w, ready)


def render_overview_detail(w: Workspace, ready: bool) -> None:
    """The rest of the opening screen: recorded detail, evidence and sources."""
    if not ready:
        return
    recent = w.hourly.tail(24)
    peak_time = recent.energy_kwh.idxmax()

    ui.section("The day that was *recorded*",
               "The last 24 measured hours of the loaded input, before any model runs.")
    first, second = st.columns(2)
    first.metric("Energy in those 24 hours", f"{recent.energy_kwh.sum():.2f} kWh",
                 help="Sum of recorded input energy, not a forecast.")
    second.metric("Busiest recorded hour", f"{recent.energy_kwh.max():.2f} kWh",
                  help=f"Starts {peak_time:%d %b %Y, %H:%M}.")
    st.caption(f"Peak interval starts {peak_time:%d %b %Y, %H:%M} in the {w.hourly.attrs['timestamp_clock']}. "
               "This is a measured peak, not a forecast alert.")
    with st.expander("Read all 24 recorded hours"):
        st.dataframe(recent.reset_index(), hide_index=True, use_container_width=True)
        st.download_button("Download last 24 recorded hours CSV", recent.reset_index().to_csv(index=False).encode(),
                           f"ecowatt_{w.spec.id}_last_24_inputs.csv", "text/csv")

    ui.section("How the three methods *compare*",
               "Held-out test error for the same days and the same target hours.")
    benchmark_scope(w)
    if not w.metrics.empty:
        ui.overview_evidence(w.metrics)
    else:
        st.info(w.metrics_error or "Benchmark evidence is unavailable for this experiment.")
    st.write(w.spec.limitations)

    ui.section("Where the data *comes from*", "One real household, recorded and published by its source.")
    st.write(w.spec.source_note)
    st.caption(w.spec.timezone_note)
    st.markdown(f"[Official dataset source]({w.spec.citation_url}) — {w.spec.license}")
    st.caption("An uploaded household is separate from the reference benchmark. Transfer accuracy is unverified. "
               "A forecast supports planning; cost savings and peak-alert accuracy are not guaranteed.")
    ui.section("A simple path to *the next day.*", "Begin with history. Explore the hours ahead. Check the evidence.")
    first, second, third = st.columns(3)
    with first:
        ui.card("Begin with recorded hours.", "Inspect the input, its clock and its source before making a prediction.", "01 / History")
        open_page("Open Data Hub", "Data Hub", key="overview_data")
    with second:
        ui.card("Look one day ahead.", "Read the daily total, find the peak hour and download 24 computed predictions.", "02 / Forecast")
        open_page("Open Forecast Studio", "Forecast Studio", key="overview_studio", primary=True)
    with third:
        ui.card("Understand the error.", "Compare all three methods on the same held-out days and read the limitations.", "03 / Evidence")
        open_page("Open Model Lab", "Model Lab", key="overview_lab")
    with st.container(key="overview_closing"):
        ui.html('<p class="ew-eyebrow">Your next 24 hours</p><h2>Make room<br>for <em>foresight.</em></h2>')
        open_page("Explore the next 24 hours ↗", "Forecast Studio", key="overview_closing_studio", primary=True)


def render_data_hub(w: Workspace) -> None:
    """Screen 3: upload or inspect the hourly data, with clear error messages."""
    ui.page_header("Data Hub", "Connect your hourly CSV, understand its clock and inspect the data used for prediction.")
    ui.section("Connect your *readings*", "Choose Built-in dataset demo or Upload hourly CSV in the sidebar. The same input follows you between screens.")
    if input_ready(w):
        source_context(w)
        hourly = w.hourly
        a, b = st.columns(2)
        a.metric("Validated hourly rows", f"{len(hourly):,}")
        b.metric("Input mean energy", f"{hourly.energy_kwh.mean():.3f} kWh")
        ui.summary([
            ("Validation", "Ascending, unique, finite, non-negative and hourly in the declared clock."),
            ("Forecast readiness", "Historical and future views available" if len(hourly) >= MIN_BACKTEST else f"Future view available; historical comparison needs {MIN_BACKTEST} rows."),
        ])
        recent = hourly.tail(168)
        figure = go.Figure(go.Scatter(
            x=recent.index, y=recent.energy_kwh, mode="lines", name="Recorded input",
            hovertemplate="%{x|%d %b %Y, %H:%M}<br>%{y:.3f} kWh<extra>Recorded input</extra>",
        ))
        figure.update_layout(title="Recorded input · latest available week", template="simple_white",
                             height=340, margin={"l":50,"r":20,"t":65,"b":55},
                             xaxis_title="Interval start", yaxis_title="Hourly energy (kWh)")
        ui.chart(figure, w.palette, w.reduce_motion)
        st.caption("This chart shows actual uploaded or attributed input values. No missing values are filled or fabricated.")
        if w.source_mode == "Built-in dataset demo" and w.spec.id == "xky_2025":
            st.caption("XKY recorded energy values are past-only counter-boundary estimates, not exact clock-hour integrals.")
        with st.expander("Inspect validated input rows", expanded=True):
            st.dataframe(hourly.tail(500).reset_index(), hide_index=True, use_container_width=True)
            st.caption(f"Showing the most recent {min(500,len(hourly)):,} of {len(hourly):,} validated rows.")
            normalized = hourly.reset_index().assign(timestamp_clock=hourly.attrs["timestamp_clock"])
            st.download_button("Download validated input CSV", normalized.to_csv(index=False).encode(),
                               f"ecowatt_{w.spec.id}_validated_input.csv", "text/csv")
        render_consumption_profile(w)
        open_page("Forecast with this data", "Forecast Studio", key="data_studio", primary=True)
    ui.section("The CSV *contract*", "One documented schema is enforced by the working Python app.")
    st.code("timestamp,energy_kwh\n2026-01-01 00:00:00,0.52\n2026-01-01 01:00:00,0.47", language="csv")
    st.caption("Two rows illustrate formatting only. A usable file needs at least 169 rows.")
    st.markdown(
        f"- **History:** {MIN_HISTORY} consecutive hours for future forecasts; {MIN_BACKTEST} for a historical comparison.\n"
        "- **Columns:** timestamp / datetime and energy_kwh / consumption_kwh; case-insensitive. Exactly one alias per field.\n"
        "- **Values:** finite, non-negative decimal kWh. No gaps, duplicates, reordering or silent aggregation.\n"
        "- **Limits:** UTF-8 CSV, 20 MiB, 32 columns and 100,000 data rows.\n"
        "- **Time:** whole hours after normalization; one fixed offset on all rows or no offset on all rows."
    )
    st.info("XKY: explicit offsets convert to UTC; no offset means UTC. UCI: supplied wall-clock labels and fixed offsets are preserved; its original timezone is unconfirmed.")
    if w.hourly is not None:
        sample_download(w)
    with st.expander("Source, attribution and input privacy"):
        st.markdown(f"**Reference benchmark:** {w.spec.label}. **Training target:** {w.spec.target_description}.")
        st.write(w.spec.source_note)
        st.caption(w.spec.timezone_note)
        st.markdown(f"[Official dataset source]({w.spec.citation_url}) · {w.spec.license}")
        st.write("Uploaded CSV data is processed by the Streamlit server for this session. This app does not save uploaded rows to project files or retrain the shared models.")


def render_help(w: Workspace) -> None:
    """Screen 5: step-by-step guide and answers to common questions."""
    ui.page_header("Help", "A practical guide to the forecast, the data and your live demonstration.")
    with st.expander("Start here · your first forecast", expanded=True):
        st.write("For the fastest start, choose Overview → Run the 24-hour forecast. It selects the attributed XKY 2025 example and computes 24 real historical predictions; open Understand these results for a plain explanation.")
        st.markdown(
            "1. Choose a reference benchmark in the sidebar. Keep Built-in dataset demo for an offline walkthrough.\n"
            "2. Open Forecast Studio and choose Historical backtest to compare against recorded targets.\n"
            "3. Choose a model, then press Generate 24-hour forecast.\n"
            "4. Read the total and peak, inspect all 24 rows, and download the computed CSV.\n"
            "5. Open Data Hub for input details or Model Lab for benchmark evidence."
        )
        open_page("Try Forecast Studio", "Forecast Studio", key="help_studio", primary=True)
    questions = [
        ("Why are the dates older than today?", "The app forecasts the next 24 hours after the last selected reading. XKY contains authentic 2025 measurements released in 2026; UCI contains 2006–2010 measurements. Neither is a live meter feed, and dates are never shifted to imply current measurements."),
        ("What is the difference between a backtest and a future forecast?", "A historical backtest has 24 recorded target hours available for comparison. A future forecast follows the final input hour; its actual targets are unknown. Only past consumption enters the predictors."),
        ("What does kWh mean?", "kWh is energy used during an interval. This app predicts hourly energy. kW is power; the two units must not be substituted."),
        ("Which model should I choose?", "The recommended model has the lowest validation MAE for the reference household. Seasonal Naive repeats yesterday's same hour; Ridge is regularized linear regression; HistGradientBoosting learns nonlinear patterns."),
        ("Do the benchmark scores describe my uploaded home?", "No. Uploaded data uses a pretrained reference-household model. Its transfer accuracy is unverified. A selected historical origin can have its own observed MAE/RMSE, but that single window is not the full benchmark."),
        ("Can I trust high-demand indicators or calculate a guaranteed bill saving?", "Indicators compare predictions with a training-only threshold and are not validated operational alerts. The cost planner uses your explicit rates for an energy-only scenario. It excludes tax, standing charges and tiered tariffs. Additional-load windows describe new energy, not an identified appliance; there is no switching control or guaranteed saving."),
        ("Can I evaluate my own recorded data?", "Model Lab → Evaluate your recorded days compares all three frozen reference models on up to 14 complete 23:00 windows. It does not retrain or change the recommended model. Uploaded household identity and training overlap are unknown. These retrospective errors do not establish independent future performance or replace the official benchmark."),
        ("Where can I understand consumption patterns?", "Data Hub → Consumption patterns shows actual mean energy by clock hour and a day/hour heatmap from at most 28 days. Partial days keep their recorded-hour counts; blanks are not zero-filled. Two non-overlapping 168-hour totals are compared only when at least 336 hours exist. A change is descriptive, not an appliance or anomaly diagnosis."),
        ("Where are false alarms and checks across time?", "Model Lab → Reliability checks shows source-test precision, recall, misses and false alarms at the unchanged training-only threshold, plus three development-only expanding folds. UCI HGB detects only 4 of 203 high-demand hours. XKY HGB does not consistently outperform Seasonal Naive across periods. These checks did not replace the production model or tune on reserved test targets."),
        ("Can I stop animations?", "Open Colour & motion in the sidebar and turn on Reduce motion. The OS reduced-motion preference also disables decoration. Replay page entrance does not recompute forecasts."),
        ("Can I read charts without hover or colour?", "Below each chart, open Read chart data as a table. Native HTML tables expose series names, axis labels, dates, plotted values and displayed error-bar lengths. Missing heatmap cells say Not available rather than zero. Numbers are rounded to eight significant digits for reading; use the CSV downloads for exported precision. Tab can focus the table's scroll region. Full screen-reader and keyboard verification remains pending."),
        ("Why did my previous result disappear?", "Changing data, clock provenance, dataset, model or forecast controls invalidates matching output. Return to Forecast Studio and generate a new forecast. Moving between screens alone preserves your selection and result."),
        ("Is the company domain connected?", "The proposed ecowatt.nihaoint.com subdomain is not connected. Hosting, DNS, HTTPS and public verification remain separate deployment work."),
    ]
    for question, answer in questions:
        with st.expander(question):
            st.write(answer)
    with st.expander("Missing data or model files"):
        st.code(f"{w.spec.prepare_command}\n{w.spec.train_command}\npython scripts/create_demo_excerpt.py --dataset {w.spec.id}\npython scripts/verify_release.py --dataset {w.spec.id} --write-manifests")
        st.caption("Run these commands from the project root, with the virtual environment active. Never upload a joblib model from an untrusted source.")
    with st.expander("Project scope and demonstration checklist"):
        st.write(w.spec.limitations)
        st.markdown("Rehearse a historical forecast, switch model, explain MAE/RMSE, upload a valid CSV, show the clock notice and download 24 computed rows. Be ready to explain leakage prevention and the limitations of one household.")
        st.caption("The assignment report remains on hold pending final design/build confirmation.")
