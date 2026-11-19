"""Plotly charts for EcoWatt, without changing forecast values or evidence."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# Build-time defaults. ui.chart() replaces these with the active palette, so the
# meaning stays fixed: PRIMARY is a model prediction, GRAPHITE is a measurement.
# Note: some names are old and do not match the colour. EMERALD is now a blue
# and CYAN is a grey. Read the comment on each line for the real meaning.
NAVY ="#edf0f1"
PANEL = "#ffffff"
EMERALD = "#1b3fd8"   # prediction
CYAN = "#566770"      # measurement
OFF_WHITE = "#0f2027"
MUTED = "#4c5f68"
GRID = "rgba(15, 32, 39, 0.10)"

# Readable names for the 16 model inputs. The charts show these instead of
# the short code names. If a name is missing here, the code name is used.
FEATURE_LABELS = {
    "horizon": "Forecast horizon",
    "last_value": "Last observed hour",
    "origin_lag_24": "Origin lag · 24 hours",
    "origin_lag_48": "Origin lag · 48 hours",
    "origin_lag_168": "Origin lag · 7 days",
    "target_lag_24": "Same hour · previous day",
    "target_lag_168": "Same hour · previous week",
    "rolling_mean_24": "Daily average",
    "rolling_std_24": "Daily variability",
    "rolling_mean_168": "Weekly average",
    "rolling_std_168": "Weekly variability",
    "target_hour_sin": "Target hour · sine",
    "target_hour_cos": "Target hour · cosine",
    "target_dow_sin": "Target weekday · sine",
    "target_dow_cos": "Target weekday · cosine",
    "target_is_weekend": "Weekend indicator",
}


def _layout(fig: go.Figure, title: str, y_title: str = "Hourly energy (kWh)", subtitle: str | None = None) -> go.Figure:
    """Apply the restrained panel style and transition only real chart updates.

    The subtitle carries the unit and the scope, so the chart does not depend on
    a long rotated axis label to explain what is plotted.
    """
    fig.update_layout(
        title={
            "text": title, "font": {"size": 15, "color": OFF_WHITE},
            "subtitle": {"text": subtitle, "font": {"size": 12, "color": MUTED}} if subtitle else None,
            "x": 0.04, "xref": "container", "xanchor": "left",
            "y": 0.98, "yref": "container", "yanchor": "top",
        },
        template="simple_white",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font={"family": "Inter, Arial, sans-serif", "color": OFF_WHITE, "size": 12},
        height=370,
        autosize=True,
        margin={"l": 56, "r": 28, "t": 86, "b": 58},
        xaxis_title="Target hour",
        yaxis_title=y_title,
        # "x unified": hovering on one hour shows all lines for that hour together.
        hovermode="x unified",
        hoverlabel={"bgcolor": "#142734", "bordercolor": "#243B4B", "font": {"color": OFF_WHITE, "size": 12}},
        legend={"orientation": "h", "y": 1.08, "yanchor": "bottom", "x": 0, "font": {"color": MUTED, "size": 11}},
        transition={"duration": 350, "easing": "cubic-in-out"},
        showlegend=True,
    )
    fig.update_xaxes(
        gridcolor=GRID, showgrid=False, zeroline=False, showline=False,
        tickfont={"color": MUTED, "size": 11}, title_font={"color": MUTED, "size": 11},
        automargin=True,
    )
    # rangemode="tozero": the y axis always starts at 0, so the heights of the
    # bars and lines are not made to look bigger than they are.
    fig.update_yaxes(
        gridcolor=GRID, zeroline=False, showline=False, rangemode="tozero",
        tickfont={"color": MUTED, "size": 11}, title_font={"color": MUTED, "size": 11},
        automargin=True,
    )
    return fig


def forecast_figure(forecast: pd.DataFrame, historical_backtest: bool) -> go.Figure:
    """Show computed forecasts, genuine observations when available, and the real peak.

    Input: the forecast table (from forecast_24h) and True or False for a
    historical backtest. Optional columns add more lines: lower_kwh and
    upper_kwh (the likely range) and baseline_kwh (the simple rule).
    Output: a Plotly figure. The data values are never changed here.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=forecast["target_timestamp"],
            y=forecast["predicted_kwh"],
            customdata=forecast[["horizon"]].to_numpy(),
            mode="lines+markers",
            name="Forecast",
            line={"color": EMERALD, "width": 3},
            marker={"color": EMERALD, "size": 4},
            fill="tozeroy",
            fillcolor="rgba(121, 232, 186, 0.08)",
            hovertemplate="Forecast · hour %{customdata[0]}<br>%{x|%d %b %Y, %H:%M}<br><b>%{y:.3f} kWh</b><extra></extra>",
        )
    )
    # Optional: a shaded 80% range around the forecast (drawn only when the columns exist).
    # How the shading works: the first line (upper) is invisible (width 0).
    # The second line (lower) is also invisible, but "tonexty" fills the area
    # between it and the line before it.
    if {"lower_kwh", "upper_kwh"}.issubset(forecast.columns):
        fig.add_trace(go.Scatter(
            x=forecast["target_timestamp"], y=forecast["upper_kwh"], mode="lines",
            line={"width": 0}, showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=forecast["target_timestamp"], y=forecast["lower_kwh"], mode="lines",
            line={"width": 0}, fill="tonexty", fillcolor="rgba(27, 63, 216, 0.14)",
            name="Likely range (80%)", hoverinfo="skip",
        ))
    # Optional: the simple 7-day same-hour median rule, for comparison.
    if "baseline_kwh" in forecast.columns:
        fig.add_trace(go.Scatter(
            x=forecast["target_timestamp"], y=forecast["baseline_kwh"], mode="lines",
            name="Simple rule (7-day median)", line={"color": MUTED, "width": 2, "dash": "dash"},
            hovertemplate="Simple rule<br>%{x|%d %b %Y, %H:%M}<br><b>%{y:.3f} kWh</b><extra></extra>",
        ))
    # Real (observed) values are shown only in a historical backtest, and only
    # if every one of the 24 hours has a real value.
    if historical_backtest and "actual_kwh" in forecast and forecast["actual_kwh"].notna().all():
        fig.add_trace(
            go.Scatter(
                x=forecast["target_timestamp"],
                y=forecast["actual_kwh"],
                mode="lines",
                name="Observed",
                line={"color": CYAN, "width": 2, "dash": "dot"},
                hovertemplate="Observed<br>%{x|%d %b %Y, %H:%M}<br><b>%{y:.3f} kWh</b><extra></extra>",
            )
        )
    if forecast["predicted_kwh"].notna().any():
        peak = forecast.loc[forecast["predicted_kwh"].idxmax()]
        fig.add_trace(go.Scatter(
            x=[peak["target_timestamp"]], y=[peak["predicted_kwh"]],
            mode="markers", name="Forecast peak",
            marker={"symbol": "diamond", "size": 10, "color": OFF_WHITE, "line": {"color": PANEL, "width": 2}},
            hovertemplate="Forecast peak<br>%{x|%d %b %Y, %H:%M}<br><b>%{y:.3f} kWh</b><extra></extra>",
        ))
        # Label the peak on the chart itself, so the reader does not have to
        # match a legend entry to a point. The text repeats plotted values only.
        # If the peak is in the first half of the day, the label goes to the
        # right of it. Otherwise it goes to the left, so it stays inside the chart.
        early = pd.Timestamp(peak["target_timestamp"]) < forecast["target_timestamp"].median()
        fig.add_annotation(
            x=peak["target_timestamp"], y=peak["predicted_kwh"],
            text=f"Peak {peak['predicted_kwh']:.2f} kWh · {pd.Timestamp(peak['target_timestamp']):%H:%M}",
            showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=MUTED, ax=38 if early else -38, ay=-30,
            xanchor="left" if early else "right", font={"color": OFF_WHITE, "size": 11},
            bgcolor="rgba(255, 255, 255, 0.88)", borderpad=4,
        )
    title = "Historical backtest" if historical_backtest else "Next 24 hours · energy forecast"
    # Short subtitles, so the whole line also fits on a phone screen.
    subtitle = ("Forecast and recorded energy, in kWh"
                if historical_backtest else "Predicted energy for the next 24 hours, in kWh")
    _layout(fig, title, subtitle=subtitle)
    # Keep the legend out of the title band. It may wrap to two rows on a
    # narrow phone; the bottom margin reserves room for both rows and ticks.
    fig.update_layout(
        height=400,
        margin={"l": 50, "r": 18, "t": 78, "b": 110},
        legend={
            "orientation": "h", "x": 0, "xanchor": "left",
            "y": -0.32, "yanchor": "top", "font": {"color": MUTED, "size": 10},
        },
    )
    fig.update_xaxes(tickformat="%H:%M", nticks=8)
    fig.update_yaxes(tickformat=".1f")
    return fig


def model_comparison_figure(metrics: pd.DataFrame) -> go.Figure:
    """Rank actual held-out MAE values, with the best model shown first.

    Input: the test scores table (columns model and mae_kwh).
    Output: a horizontal bar chart. The best model (lowest MAE) is blue and
    the other models are grey.
    """
    ordered = metrics.sort_values("mae_kwh", ascending=True, kind="stable")
    labels = ordered["model"].replace({"HistGradientBoosting": "HGB"})
    colors = [EMERALD] + ["rgba(86, 103, 112, 0.55)"] * max(0, len(ordered) - 1)
    fig = go.Figure(
        go.Bar(
            x=ordered["mae_kwh"],
            y=labels,
            customdata=ordered[["model"]].to_numpy(),
            orientation="h",
            marker={"color": colors, "cornerradius": 5},
            width=0.46,
            text=ordered["mae_kwh"].map(lambda value: f"{value:.3f}"),
            textposition="outside",
            textfont={"color": OFF_WHITE, "size": 12},
            cliponaxis=False,
            hovertemplate="%{customdata[0]}<br>Held-out MAE: <b>%{x:.4f} kWh</b><extra></extra>",
        )
    )
    _layout(fig, "Held-out MAE · lower is better",
            subtitle="Average absolute error per hour, in kWh")
    fig.update_layout(showlegend=False, hovermode="closest", margin={"l": 116, "r": 42, "t": 84, "b": 54})
    fig.update_xaxes(title="MAE (kWh)", showgrid=True, rangemode="tozero", nticks=4)
    # Leave 28% extra space on the right, so the number text after the longest bar fits.
    if not ordered.empty:
        fig.update_xaxes(range=[0, max(float(ordered["mae_kwh"].max()) * 1.28, 0.01)])
    # "reversed" puts the first (best) model at the top of the chart.
    fig.update_yaxes(title=None, autorange="reversed", showgrid=False, rangemode="normal")
    return fig


def importance_figure(importance: pd.DataFrame) -> go.Figure:
    """Show actual permutation means and repeat SDs with readable feature labels.

    Input: the importance table (feature, importance_mean, importance_std).
    Each bar shows how much the error grew when that input was shuffled.
    A longer bar means the model depends more on that input. The thin error
    bar shows how much the result changed between the repeats (SD).
    Only the 10 largest values are shown.
    """
    top = importance.nlargest(10, "importance_mean").sort_values("importance_mean", ascending=False, kind="stable")
    labels = top["feature"].map(lambda name: FEATURE_LABELS.get(name, name.replace("_", " ")))
    fig = go.Figure(
        go.Bar(
            x=top["importance_mean"],
            y=labels,
            customdata=top[["feature", "importance_std"]].to_numpy(),
            orientation="h",
            width=0.5,
            marker={"color": EMERALD, "cornerradius": 3},
            error_x={"type": "data", "array": top["importance_std"], "visible": True, "color": CYAN, "thickness": 1.2, "width": 3},
            hovertemplate="%{y}<br>%{customdata[0]}<br>MAE increase: <b>%{x:.4f} kWh</b><br>Repeat SD: %{customdata[1]:.4f} kWh<extra></extra>",
        )
    )
    _layout(fig, "Feature sensitivity",
            subtitle="Error increase when one input is shuffled, in kWh")
    fig.update_layout(showlegend=False, hovermode="closest", height=424, margin={"l": 155, "r": 20, "t": 84, "b": 58})
    fig.update_xaxes(title="MAE increase (kWh)", showgrid=True, rangemode="tozero", tickformat=".3f", nticks=4)
    fig.update_yaxes(title=None, autorange="reversed", showgrid=False, rangemode="normal", tickfont={"size": 10})
    return fig
