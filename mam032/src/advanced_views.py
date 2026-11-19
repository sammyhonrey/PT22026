"""Decision-support panels using genuine input, fixed models and explicit rates."""

import hashlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import ui
from .decision_support import MAX_REPLAY_DAYS, additional_load_windows, consumption_profile, energy_costs, replay_models, replay_origins
from .forecasting import ModelArtifactError
from .workspace import Workspace


def _keep_open(panel: str, scope: str) -> None:
    """Widget callbacks preserve an advanced panel before Streamlit reruns it."""
    st.session_state[panel] = scope


def render_personal_evaluation(w: Workspace) -> None:
    """Show a panel that scores all three models on the user's recent recorded days."""
    with st.expander("Evaluate your recorded days · all three models",
                     expanded=st.session_state.get("active_replay_panel") == w.widget_scope):
        st.write("Compare the fixed pretrained models on the latest complete 23:00 windows in the current input. This is a retrospective replay, not a new held-out benchmark, retraining, or automatic model selection.")
        if w.hourly is None or w.input_error or w.bundle is None:
            st.info("Load valid hourly data and a matching model bundle to evaluate recorded windows.")
            return
        # The user cannot choose more days than the data (or the limit) allows.
        limit = min(MAX_REPLAY_DAYS, len(replay_origins(w.hourly)))
        if limit == 0:
            st.info("No complete daily replay is available. Each 23:00 origin needs 169 known history hours and the following 24 recorded targets. Future-only input cannot be evaluated.")
            return
        settings = st.session_state.setdefault("replay_settings", {})
        days = int(st.number_input("Recorded days to evaluate", min_value=1, max_value=limit,
                                   value=min(settings.get(w.widget_scope, 7), limit), step=1,
                                   key=f"replay_days_{w.widget_scope}", on_change=_keep_open,
                                   args=("active_replay_panel", w.widget_scope)))
        settings[w.widget_scope] = days
        # This key describes the current input. A saved result is shown only
        # if its key is the same, so old results never match new data.
        key = (w.input_key, w.release_id, days)
        st.caption(f"Up to {limit} complete days available in this input; limited to {MAX_REPLAY_DAYS} per run. All methods use identical target hours. No prediction window overlaps another.")
        if st.button("Evaluate recorded days", key=f"run_replay_{w.widget_scope}", use_container_width=True,
                     on_click=_keep_open, args=("active_replay_panel", w.widget_scope)):
            st.session_state.pop("personal_replay", None)
            try:
                with st.spinner("Comparing fixed models on recorded windows…"):
                    result = replay_models(w.hourly, w.bundle, days)
                for name in ["summary", "daily", "predictions"]:
                    result[name] = result[name].assign(dataset_id=w.spec.id, artifact_release_id=w.release_id,
                                                      timestamp_clock=w.hourly.attrs["timestamp_clock"],
                                                      timestamp_input_clock=w.hourly.attrs["timestamp_input_clock"],
                                                      timestamp_normalization=w.hourly.attrs["timestamp_normalization"],
                                                      evaluation_scope=result["scope"])
                st.session_state["personal_replay"] = {"key": key, "data": result}
            except (ValueError, ModelArtifactError) as exc:
                st.error(f"Recorded-window evaluation failed: {exc}")
                return
        saved = st.session_state.get("personal_replay")
        if not saved or saved["key"] != key:
            st.caption("Choose the day count and evaluate to view current-input results. Changes to data, clock, release or day count hide earlier results.")
            return
        result = saved["data"]
        st.caption(f"Replay origins: {result['origins'][0]:%d %b %Y, %H:%M} → {result['origins'][-1]:%d %b %Y, %H:%M}. Clock: {w.hourly.attrs['timestamp_clock']}.")
        if w.source_mode == "Built-in dataset demo":
            overlap = int((result["origins"] < pd.Timestamp(w.bundle.get("trained_through", "1900-01-01"))).sum())
            st.caption(f"{overlap} replay origins predate the saved training cutoff. This input replay is separate from the published reserved-test evaluation.")
        else:
            st.warning("These errors describe only the uploaded recorded windows. Household identity and overlap with reference-model training are unknown; independent future accuracy is not established.")
        st.dataframe(result["summary"][["model", "days", "hours", "mae_kwh", "rmse_kwh", "bias_kwh", "mae_reduction_vs_naive_pct"]],
                     hide_index=True, use_container_width=True)
        best = result["summary"].iloc[0]
        st.info(f"Lowest MAE on these {days} recorded windows: {best.model}, {best.mae_kwh:.4f} kWh. The reference recommendation remains {w.bundle['selected_model']}, chosen from the original validation period. This replay does not change your selected forecast model.")
        st.caption("Positive bias means overprediction. Positive MAE reduction means improvement over Seasonal Naive on these windows; it is undefined when baseline MAE is zero. Correlated hourly errors do not establish an independent sample size.")
        figure = go.Figure()
        for model, rows in result["daily"].groupby("model", sort=True):
            figure.add_trace(go.Scatter(x=rows.forecast_origin, y=rows.mae_kwh, mode="lines+markers", name=model))
        figure.update_layout(title={"text":"Replay · daily MAE", "font":{"size":14}, "x":.04, "xref":"container", "xanchor":"left"}, height=390,
                             margin={"l":50,"r":20,"t":60,"b":110}, xaxis_title="Origin date · 23:00",
                             yaxis_title="MAE (kWh)", legend={"orientation":"h","y":-.3,"font":{"size":10}})
        origins = result["origins"]
        ticks = [origins[index] for index in sorted({round(i * (len(origins) - 1) / 3) for i in range(4)})]
        figure.update_xaxes(tickmode="array", tickvals=ticks, ticktext=[origin.strftime("%d %b") for origin in ticks])
        ui.chart(figure, w.palette, w.reduce_motion)
        st.download_button("Download replay metrics CSV", result["summary"].to_csv(index=False).encode(),
                           f"ecowatt_{w.spec.id}_recorded_replay_metrics.csv", "text/csv")
        st.download_button("Download replay predictions CSV", result["predictions"].to_csv(index=False).encode(),
                           f"ecowatt_{w.spec.id}_recorded_replay_predictions.csv", "text/csv")


def render_consumption_profile(w: Workspace) -> None:
    """Show a panel with the average use per hour, a heatmap and daily totals."""
    with st.expander("Consumption patterns · recorded input",
                     expanded=st.session_state.get("active_profile_panel") == w.widget_scope):
        data = consumption_profile(w.hourly)
        st.caption(f"{data['rows']} recorded hours: {data['first_timestamp']:%d %b %Y, %H:%M} → {data['last_timestamp']:%d %b %Y, %H:%M}. At most 28 days; clock: {w.hourly.attrs['timestamp_clock']}. This describes input energy, not appliances, causal effects or forecast accuracy.")
        hourly = data["hourly"]
        figure = go.Figure(go.Bar(x=hourly.hour, y=hourly.mean_kwh, customdata=hourly[["observations"]],
                                 name="Recorded mean", hovertemplate="Hour %{x}:00<br>Mean %{y:.3f} kWh<br>%{customdata[0]} readings<extra></extra>"))
        figure.update_layout(title={"text":"Average energy by hour", "font":{"size":14}, "x":.04, "xref":"container", "xanchor":"left"}, height=320, margin={"l":50,"r":20,"t":55,"b":55},
                             xaxis_title="Hour in the declared input clock", yaxis_title="Mean hourly energy (kWh)")
        figure.update_xaxes(dtick=3)
        ui.chart(figure, w.palette, w.reduce_motion)
        matrix = data["heatmap"]
        figure = go.Figure(go.Heatmap(x=list(matrix.columns), y=list(matrix.index), z=matrix.to_numpy(),
                                     colorscale=[[0, ui.PALETTES[w.palette]["panel"]], [1, ui.PALETTES[w.palette]["primary"]]],
                                     colorbar={"title":"kWh"}, hoverongaps=False,
                                     hovertemplate="%{y}, %{x}:00<br>%{z:.3f} kWh<extra>Recorded input</extra>"))
        figure.update_layout(title={"text":"Recorded energy heatmap", "font":{"size":14}, "x":.04, "xref":"container", "xanchor":"left"}, height=390, margin={"l":95,"r":35,"t":55,"b":55},
                             xaxis_title="Clock hour", yaxis_title="Recorded date")
        ui.chart(figure, w.palette, w.reduce_motion)
        st.caption("Blank cells are outside this input window; they are not zero consumption. Partial-day totals remain labeled with their recorded hour count.")
        st.dataframe(data["daily"], hide_index=True, use_container_width=True)
        if data["change"] is not None:
            change = data["change"]
            percent = "undefined (previous block is zero)" if change["change_percent"] is None else f"{change['change_percent']:+.2f}%"
            st.info(f"Latest 168 recorded hours: {change['recent_168h_kwh']:.2f} kWh; previous 168: {change['previous_168h_kwh']:.2f} kWh. Difference {change['difference_kwh']:+.2f} kWh; relative change {percent}. This descriptive change is not a model-drift or anomaly diagnosis.")
            st.caption(f"Previous block: {change['previous_start']} → {change['previous_end']}; latest block: {change['recent_start']} → {change['recent_end']}.")
        else:
            st.info("At least 336 recorded hours are needed for two non-overlapping 168-hour energy comparisons.")
        st.download_button("Download consumption profile CSV", hourly.assign(dataset_id=w.spec.id,
                           timestamp_clock=w.hourly.attrs["timestamp_clock"], window_first=data["first_timestamp"],
                           window_last=data["last_timestamp"]).to_csv(index=False).encode(),
                           f"ecowatt_{w.spec.id}_consumption_profile.csv", "text/csv",
                           on_click=_keep_open, args=("active_profile_panel", w.widget_scope))


def render_planning(w: Workspace, forecast: pd.DataFrame) -> None:
    """Show a panel where the user types electricity prices to estimate cost.

    It also compares start times for an extra load (for example a washing
    machine). The prices are always typed by the user, never fetched.
    """
    # A short ID for this forecast, so each forecast keeps its own settings.
    scope = hashlib.sha256(repr((w.input_key, w.release_id, w.model_name, str(forecast.forecast_origin.iloc[0]))).encode()).hexdigest()[:16]
    keep_open = {"on_change": _keep_open, "args": ("active_planning_panel", scope)}
    with st.expander("Energy cost and additional-load planner",
                     expanded=st.session_state.get("active_planning_panel") == scope):
        st.write("Enter your own scenario rates. Costs include energy only: no taxes, standing charges, tiered tariffs or guaranteed bill savings. This tool does not fetch a tariff or control appliances.")
        st.info(f"Tariff hours use the forecast's displayed clock: {w.hourly.attrs['timestamp_clock']}. Convert your tariff schedule into this clock first. Recorded dates remain unchanged.")
        preferences = st.session_state.setdefault("planning_settings", {})
        settings = preferences.get(scope, {})
        currencies = ["BDT", "CNY", "EUR", "USD", "GBP"]
        currency = st.selectbox("Scenario currency", currencies, index=currencies.index(settings.get("currency", "BDT")), key=f"currency_{scope}", **keep_open)
        modes = ["Flat rate", "Peak / off-peak"]
        mode = st.radio("Tariff scenario", modes, index=modes.index(settings.get("mode", "Flat rate")), horizontal=True, key=f"tariff_mode_{scope}", **keep_open)
        rate = st.number_input("Base / off-peak rate per kWh", min_value=0.0, value=settings.get("rate"), step=.01,
                               placeholder="Enter your own rate", key=f"base_rate_{scope}", **keep_open)
        peak_rate, start, end = None, settings.get("start", 17), settings.get("end", 22)
        if mode == "Peak / off-peak":
            peak_rate = st.number_input("Peak rate per kWh", min_value=0.0, value=settings.get("peak_rate"), step=.01,
                                        placeholder="Enter your own peak rate", key=f"peak_rate_{scope}", **keep_open)
            start = int(st.number_input("Peak period starts (inclusive hour)", min_value=0, max_value=23, value=start, key=f"peak_start_{scope}", **keep_open))
            end = int(st.number_input("Peak period ends (exclusive hour)", min_value=0, max_value=23, value=end, key=f"peak_end_{scope}", **keep_open))
            st.caption("Start is included; end is excluded. A start later than end crosses midnight. Equal hours are invalid.")
        energy = float(st.number_input("Hypothetical additional energy (kWh)", min_value=0.0, value=settings.get("energy", 1.0), step=.1, key=f"load_energy_{scope}", **keep_open))
        duration = int(st.number_input("Additional load duration (hours)", min_value=1, max_value=24, value=settings.get("duration", 2), key=f"load_duration_{scope}", **keep_open))
        reference = st.selectbox("Compare with additional load starting at", list(range(25 - duration)),
                                 index=min(settings.get("reference", 0), 24 - duration),
                                 format_func=lambda n: forecast.target_timestamp.iloc[n].strftime("%d %b %Y, %H:%M"), key=f"reference_start_{scope}", **keep_open)
        st.caption("This is new energy spread evenly across consecutive hours. The app cannot isolate or move an appliance already included in the household forecast. Cheapest placement uses your rates only; it does not establish feasibility or safety.")
        preferences[scope] = dict(currency=currency, mode=mode, rate=rate,
                                  peak_rate=peak_rate if mode == "Peak / off-peak" else settings.get("peak_rate"), start=start, end=end,
                                  energy=energy, duration=duration, reference=reference)
        key = (scope, currency, mode, rate, peak_rate, start, end, energy, duration, reference)
        # The button stays disabled until the user has typed every price needed.
        missing = rate is None or (mode == "Peak / off-peak" and peak_rate is None)
        if st.button("Calculate planning scenario", key=f"calculate_plan_{scope}", disabled=missing, use_container_width=True,
                     on_click=_keep_open, args=("active_planning_panel", scope)):
            st.session_state.pop("planning_result", None)
            try:
                costs = energy_costs(forecast, rate, peak_rate=peak_rate, peak_start=start, peak_end=end).assign(
                    scenario_currency=currency, cost_scope="User-supplied energy-only scenario; not a bill quote.")
                windows = additional_load_windows(costs, energy, duration).assign(scenario_currency=currency,
                    timestamp_clock=w.hourly.attrs["timestamp_clock"], artifact_release_id=w.release_id, dataset_id=w.spec.id,
                    forecast_origin=forecast.forecast_origin.iloc[0], base_rate_per_kwh=rate, peak_rate_per_kwh=peak_rate,
                    peak_start_inclusive=start if peak_rate is not None else None, peak_end_exclusive=end if peak_rate is not None else None,
                    load_scope="Hypothetical additional energy; not identified existing appliance load.")
                st.session_state["planning_result"] = {"key": key, "costs": costs, "windows": windows}
            except ValueError as exc:
                st.error(f"Planning scenario could not be calculated: {exc}")
                return
        saved = st.session_state.get("planning_result")
        if saved is None or saved["key"] != key:
            st.caption("Enter rates and calculate. Changing the forecast or scenario hides previous cost results until recalculated.")
            return
        costs, windows = saved["costs"], saved["windows"]
        best = windows.iloc[0]
        comparison = windows.loc[windows.start_index.eq(reference)].iloc[0]
        ui.summary([
            ("Forecast energy-only scenario", f"{costs.estimated_energy_cost.sum():.2f} {currency}"),
            ("Lowest-cost additional-load window", f"{best.start:%d %b, %H:%M} → {best.end_exclusive:%d %b, %H:%M}"),
            ("Additional-load energy cost", f"{best.incremental_energy_cost:.2f} {currency}"),
            ("Difference versus your selected start", f"{comparison.incremental_energy_cost - best.incremental_energy_cost:.2f} {currency} (scenario only)"),
        ])
        st.dataframe(windows, hide_index=True, use_container_width=True)
        st.download_button("Download hourly cost scenario CSV", costs.to_csv(index=False).encode(), f"ecowatt_{w.spec.id}_energy_cost_scenario.csv", "text/csv")
        st.download_button("Download additional-load windows CSV", windows.to_csv(index=False).encode(), f"ecowatt_{w.spec.id}_additional_load_windows.csv", "text/csv")
