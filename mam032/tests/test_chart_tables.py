"""Chart alternatives must retain evidence and never turn missing values into zero."""

from html.parser import HTMLParser

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.chart_tables import chart_table_markup
from src.visualization import forecast_figure, importance_figure


class TableReader(HTMLParser):
    def __init__(self, markup):
        super().__init__()
        self.rows, self.headers, self.tags = [], [], []
        self.row = None
        self.cell = None
        self.feed(markup)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        if tag == "tr":
            self.row = []
        elif tag in {"td", "th"}:
            self.cell = ""
            if tag == "th":
                self.headers.append(dict(attrs))

    def handle_data(self, text):
        if self.cell is not None:
            self.cell += text

    def handle_endtag(self, tag):
        if tag in {"td", "th"}:
            self.row.append(self.cell)
            self.cell = None
        elif tag == "tr":
            self.rows.append(self.row)
            self.row = None


@pytest.mark.parametrize("historical", [True, False])
@pytest.mark.parametrize("clock", ["UTC", "Etc/GMT-6"])
def test_forecast_tables_keep_plotted_values_clock_and_observation_scope(historical, clock):
    frame = pd.DataFrame({"target_timestamp": pd.date_range("2025-01-01", periods=24, freq="h", tz=clock),
                          "horizon": range(1, 25), "predicted_kwh": np.arange(24) / 10,
                          "actual_kwh": np.arange(24) / 5})
    figure = forecast_figure(frame, historical)
    before = figure.to_json()
    markup = chart_table_markup(figure)
    reader = TableReader(markup)
    assert figure.to_json() == before
    assert ("Observed" in markup) == historical
    assert all({"scope": "col"} == header for header in reader.headers)
    rows = [row for row in reader.rows if row[0].startswith("2025-")]
    assert len(rows) == (49 if historical else 25)  # 24 forecasts, optional 24 targets, one peak
    assert [float(row[1]) for row in rows[:24]] == pytest.approx(frame.predicted_kwh)
    # Plotly represents UTC Series as naive numpy datetime64, but retains fixed
    # offsets for other clocks. Interpret naive plotted labels in the source clock.
    timestamps = [pd.Timestamp(row[0]) for row in rows[:24]]
    timestamps = [stamp.tz_localize(clock) if stamp.tz is None else stamp for stamp in timestamps]
    assert timestamps == list(frame.target_timestamp)
    assert any(tag == "div" and attrs.get("tabindex") == "0" and attrs.get("role") == "region"
               for tag, attrs in reader.tags)


def test_heatmap_preserves_blank_cells_zeroes_and_axis_units():
    figure = go.Figure(go.Heatmap(x=[0, 1], y=["2025-01-01", "2025-01-02"],
                                 z=[[0.0, np.nan], [1.25, 2.0]], colorbar_title="kWh"))
    figure.update_layout(xaxis_title="Clock hour", yaxis_title="Recorded date")
    reader = TableReader(chart_table_markup(figure))
    assert reader.rows == [["Recorded date", "Clock hour", "kWh"],
                           ["2025-01-01", "0", "0"], ["2025-01-01", "1", "Not available"],
                           ["2025-01-02", "0", "1.25"], ["2025-01-02", "1", "2"]]


def test_importance_table_preserves_negative_mean_and_plotted_repeat_sd():
    figure = importance_figure(pd.DataFrame({"feature": ["horizon", "last_value"],
                                            "importance_mean": [.1, -.02], "importance_std": [.005, .003]}))
    reader = TableReader(chart_table_markup(figure))
    assert reader.rows[0] == ["MAE increase (kWh)", "Y / category", "X error (+)"]
    assert reader.rows[1:] == [["0.1", "Forecast horizon", "0.005"], ["-0.02", "Last observed hour", "0.003"]]


def test_asymmetric_errors_keep_both_lengths():
    figure = go.Figure(go.Scatter(x=[1], y=[.4], error_y={"array": [.2], "arrayminus": [.1], "symmetric": False, "visible": True}))
    reader = TableReader(chart_table_markup(figure))
    assert reader.rows[-1] == ["1", "0.4", "0.2", "0.1"]
    assert reader.rows[0][-2:] == ["Y error (+)", "Y error (-)"]


def test_labels_and_categories_cannot_inject_html():
    figure = go.Figure(go.Bar(x=[1], y=['<img src=x onerror="alert(1)">'], name="<script>bad</script>"))
    figure.update_layout(title='Title " onmouseover="bad')
    markup = chart_table_markup(figure)
    tags = TableReader(markup).tags
    assert not any(tag in {"img", "script"} for tag, _ in tags)
    assert not any("onmouseover" in attrs or "onerror" in attrs for _, attrs in tags)
    assert "&lt;script&gt;" in markup and "&lt;img" in markup


def test_unknown_trace_fails_explicitly_instead_of_inventing_coordinates():
    with pytest.raises(ValueError, match="trace type pie"):
        chart_table_markup(go.Figure(go.Pie(values=[1, 2])))
