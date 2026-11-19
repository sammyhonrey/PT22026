"""Semantic HTML alternatives derived from the plotted points, without refitting."""

from html import escape
from numbers import Real

import pandas as pd


def _cell(value) -> str:
    """Turn one value into safe HTML text. Missing values say "Not available"."""
    if value is None or pd.isna(value):
        return "Not available"
    if isinstance(value, Real):
        return escape(format(value, ".8g"))
    return escape(str(value))


def _table(caption: str, headers: list[str], rows: list[list]) -> str:
    """Build one HTML table. escape() makes text safe so it cannot add HTML code."""
    head = "".join(f'<th scope="col">{escape(label)}</th>' for label in headers)
    body = "".join("<tr>" + "".join(f"<td>{_cell(value)}</td>" for value in row) + "</tr>" for row in rows)
    return f'<table><caption>{escape(caption)}</caption><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def chart_table_markup(figure) -> str:
    """Expose scatter/bar/heatmap coordinates, including gaps and plotted errors.

    Native details/table markup avoids nested Streamlit expanders and canvas-only
    access. Display rounds numbers to eight significant digits; CSV exports keep
    their existing precision. Never interpolate, fill gaps or mutate the figure.
    """
    title = figure.layout.title.text or "Chart"
    x_label = figure.layout.xaxis.title.text or "X / category"
    y_label = figure.layout.yaxis.title.text or "Y / category"
    tables = []
    for index, trace in enumerate(figure.data, start=1):
        caption = f"{title} · {trace.name or f'Series {index}'}"
        # Each chart line (trace) becomes one table. Screen readers can read
        # tables, but they cannot read a picture of a chart.
        if trace.type == "heatmap":
            z = trace.z
            xs = list(trace.x) if trace.x is not None else list(range(len(z[0])))
            ys = list(trace.y) if trace.y is not None else list(range(len(z)))
            rows = [[y, x, z[j][i]] for j, y in enumerate(ys) for i, x in enumerate(xs)]
            headers = [y_label, x_label, trace.colorbar.title.text or "Value"]
        elif trace.type in {"scatter", "bar"}:
            count = len(trace.x) if trace.x is not None else len(trace.y)
            xs = list(trace.x) if trace.x is not None else list(range(count))
            ys = list(trace.y) if trace.y is not None else list(range(count))
            rows = [[x, y] for x, y in zip(xs, ys, strict=True)]
            headers = [x_label, y_label]
            # These are plotted error-bar lengths, not prediction intervals.
            for axis in ["x", "y"]:
                errors = getattr(trace, f"error_{axis}")
                if errors.visible and errors.array is not None:
                    headers.append(f"{axis.upper()} error (+)")
                    asymmetric = errors.symmetric is False and errors.arrayminus is not None
                    if asymmetric:
                        headers.append(f"{axis.upper()} error (-)")
                    for row_index, row in enumerate(rows):
                        row.append(errors.array[row_index])
                        if asymmetric:
                            row.append(errors.arrayminus[row_index])
        else:
            raise ValueError(f"No HTML table alternative for trace type {trace.type}.")
        tables.append(_table(caption, headers, rows))
    return (
        '<details class="ew-chart-data"><summary>Read chart data as a table</summary>'
        '<p>Values are the plotted points, rounded to eight significant digits. '
        'Not available means missing, not zero. Dates use the displayed source clock. '
        'Use the CSV downloads for full exported precision.</p>'
        f'<div class="ew-chart-table-scroll" tabindex="0" role="region" aria-label="{escape(title, quote=True)} chart data">'
        + "".join(tables) + "</div></details>"
    )
