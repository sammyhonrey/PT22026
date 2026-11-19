"""Chart builders.

Every figure reads a completed :class:`~mmsa.schemas.AnalysisResult`; nothing here
recomputes a probability. That is deliberate: the number a user sees on a bar is
the number in the JSON that the report's figures are drawn from, so the interface
and the document cannot drift apart.

All figures share one registered Plotly template so the application, the report and
the slides look like one product.
"""

from __future__ import annotations

from typing import Mapping

import plotly.graph_objects as go
import plotly.io as pio

from .schemas import AnalysisResult

__all__ = [
    "MMSA_TEMPLATE",
    "PALETTE",
    "EMOTION_COLOURS",
    "emotion_bars",
    "radar_overlay",
    "stress_waterfall",
    "confidence_ring",
    "latency_bars",
    "empty_figure",
]

PALETTE: dict[str, str] = {
    "cyan": "#22D3EE",
    "violet": "#A78BFA",
    "magenta": "#F472B6",
    "amber": "#FBBF24",
    "green": "#34D399",
    "rose": "#FB7185",
    "slate": "#94A3B8",
    "text": "#E8EEF9",
    "muted": "#93A4C3",
    "grid": "rgba(255,255,255,0.07)",
}

EMOTION_COLOURS: dict[str, str] = {
    "happy": "#FBBF24",
    "sad": "#60A5FA",
    "angry": "#FB7185",
    "fear": "#A78BFA",
    "disgust": "#34D399",
    "neutral": "#94A3B8",
}

_FONT = "Inter, Segoe UI, system-ui, sans-serif"

MMSA_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": _FONT, "color": PALETTE["text"], "size": 12},
        colorway=[
            PALETTE["cyan"],
            PALETTE["violet"],
            PALETTE["magenta"],
            PALETTE["amber"],
            PALETTE["green"],
            PALETTE["slate"],
        ],
        margin={"l": 8, "r": 8, "t": 28, "b": 8},
        hoverlabel={"bgcolor": "#0B1226", "bordercolor": PALETTE["cyan"], "font": {"color": "#E8EEF9"}},
        xaxis={
            "gridcolor": PALETTE["grid"],
            "zerolinecolor": "rgba(255,255,255,0.16)",
            "tickfont": {"size": 11},
            "title": {"font": {"size": 11, "color": PALETTE["muted"]}},
        },
        yaxis={
            "gridcolor": PALETTE["grid"],
            "zerolinecolor": "rgba(255,255,255,0.16)",
            "tickfont": {"size": 11},
            "title": {"font": {"size": 11, "color": PALETTE["muted"]}},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "font": {"size": 10},
            "bgcolor": "rgba(0,0,0,0)",
        },
    )
)

pio.templates["mmsa"] = MMSA_TEMPLATE
pio.templates.default = "mmsa"


def empty_figure(message: str = "Awaiting a clip") -> go.Figure:
    """Neutral placeholder so the layout does not jump before the first result."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 12, "color": PALETTE["muted"]},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=220, margin={"l": 8, "r": 8, "t": 8, "b": 8})
    return fig


def emotion_bars(result: AnalysisResult, height: int = 236) -> go.Figure:
    """Grouped horizontal bars: acoustic mass, text mass and the fused distribution.

    Acoustic and text bars are plotted as *mapped mass*, which sums to their
    coverage and may therefore fall short of 1.0. That shortfall is the point --
    it is the part of the target space the branch cannot express.
    """
    labels = list(result.fused_probs.keys())
    acoustic = [result.acoustic.mapped_probs.get(k, 0.0) for k in labels]
    text = [result.text.mapped_probs.get(k, 0.0) for k in labels]
    fused = [result.fused_probs.get(k, 0.0) for k in labels]

    fig = go.Figure()
    fig.add_bar(
        name=f"Acoustic (coverage {result.acoustic.coverage:.0%})",
        y=labels,
        x=acoustic,
        orientation="h",
        marker_color=PALETTE["cyan"],
        opacity=0.85,
        hovertemplate="%{y}: %{x:.3f}<extra>acoustic</extra>",
    )
    fig.add_bar(
        name=f"Text (coverage {result.text.coverage:.0%})",
        y=labels,
        x=text,
        orientation="h",
        marker_color=PALETTE["violet"],
        opacity=0.85,
        hovertemplate="%{y}: %{x:.3f}<extra>text</extra>",
    )
    fig.add_bar(
        name="Fused",
        y=labels,
        x=fused,
        orientation="h",
        marker={"color": [EMOTION_COLOURS.get(k, PALETTE["slate"]) for k in labels],
                "line": {"color": "rgba(255,255,255,0.35)", "width": 1}},
        hovertemplate="%{y}: %{x:.3f}<extra>fused</extra>",
    )
    fig.update_layout(
        barmode="group",
        bargap=0.22,
        bargroupgap=0.06,
        height=height,
        xaxis={"range": [0, max(1.0, max(acoustic + text + fused) * 1.12)], "tickformat": ".0%"},
        yaxis={"autorange": "reversed", "title": None},
        showlegend=True,
    )
    return fig


def radar_overlay(result: AnalysisResult) -> go.Figure:
    """Three-way radar so the *shape* of disagreement is visible at a glance."""
    labels = list(result.fused_probs.keys())
    closed = labels + [labels[0]]

    def _close(values: list[float]) -> list[float]:
        return values + [values[0]]

    fig = go.Figure()
    for name, values, colour in (
        ("Acoustic", [result.acoustic.mapped_probs.get(k, 0.0) for k in labels], PALETTE["cyan"]),
        ("Text", [result.text.mapped_probs.get(k, 0.0) for k in labels], PALETTE["violet"]),
        ("Fused", [result.fused_probs.get(k, 0.0) for k in labels], PALETTE["magenta"]),
    ):
        fig.add_trace(
            go.Scatterpolar(
                r=_close(values),
                theta=closed,
                fill="toself",
                name=name,
                line={"color": colour, "width": 2},
                fillcolor=colour,
                opacity=0.22,
                hovertemplate="%{theta}: %{r:.3f}<extra>" + name + "</extra>",
            )
        )
    fig.update_layout(
        height=300,
        polar={
            "bgcolor": "rgba(255,255,255,0.02)",
            "radialaxis": {"range": [0, 1], "gridcolor": PALETTE["grid"], "tickfont": {"size": 9}},
            "angularaxis": {"gridcolor": PALETTE["grid"], "tickfont": {"size": 10}},
        },
        showlegend=True,
        margin={"l": 30, "r": 30, "t": 34, "b": 10},
    )
    return fig


def stress_waterfall(result: AnalysisResult, height: int = 280) -> go.Figure:
    """Signed contribution of each stress factor.

    Note the honest framing: the bars are contributions to the *logit*, not to the
    score. They sum with the bias to ``z``, which the sigmoid maps to the score
    shown in the annotation. Calling this a decomposition of the score directly
    would be wrong arithmetic.
    """
    contributors = dict(result.stress.contributors)
    names = list(contributors.keys())
    values = [contributors[n] for n in names]

    order = sorted(range(len(names)), key=lambda i: values[i])
    names = [names[i] for i in order]
    values = [values[i] for i in order]

    colours = [PALETTE["rose"] if v > 0 else PALETTE["cyan"] for v in values]
    pretty = [n.replace("_", " ").capitalize() for n in names]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=pretty,
            orientation="h",
            marker_color=colours,
            marker_line={"color": "rgba(255,255,255,0.25)", "width": 1},
            hovertemplate="%{y}: %{x:+.3f} logit<extra></extra>",
        )
    )
    fig.add_vline(x=0, line={"color": "rgba(255,255,255,0.35)", "width": 1})
    fig.update_layout(
        height=height,
        xaxis={"title": "contribution to logit (signed)"},
        yaxis={"title": None, "tickfont": {"size": 10}},
        showlegend=False,
        margin={"l": 8, "r": 16, "t": 30, "b": 34},
    )
    # ``logit`` is only present if stress.py recorded it; otherwise the signed
    # contributions already sum to it, so derive rather than show a bogus 0.00.
    logit = result.stress.features.get("logit", result.stress.features.get("z"))
    if logit is None:
        logit = sum(values)

    fig.add_annotation(
        text=(
            f"z = {logit:+.2f}  →  "
            f"stress {result.stress.score:.2f} · {result.stress.band_label}"
        ),
        xref="paper",
        yref="paper",
        x=0.0,
        y=1.16,
        showarrow=False,
        font={"size": 11, "color": PALETTE["muted"]},
        align="left",
    )
    return fig


def confidence_ring(result: AnalysisResult, height: int = 190) -> go.Figure:
    """Donut showing the fused confidence for the winning class."""
    value = float(result.fused_confidence)
    colour = EMOTION_COLOURS.get(result.fused_emotion, PALETTE["cyan"])

    fig = go.Figure(
        go.Pie(
            values=[value, max(0.0, 1.0 - value)],
            labels=[result.fused_emotion, "remainder"],
            hole=0.72,
            sort=False,
            direction="clockwise",
            textinfo="none",
            marker={"colors": [colour, "rgba(255,255,255,0.06)"],
                    "line": {"color": "rgba(0,0,0,0)", "width": 0}},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        height=height,
        margin={"l": 6, "r": 6, "t": 6, "b": 6},
        annotations=[
            {
                "text": f"<b>{result.fused_emotion}</b><br><span style='font-size:11px;color:{PALETTE['muted']}'>"
                        f"{value:.0%} confidence</span>",
                "showarrow": False,
                "font": {"size": 15, "color": PALETTE["text"]},
            }
        ],
    )
    return fig


def latency_bars(result: AnalysisResult, height: int = 210) -> go.Figure:
    """Per-stage latency, excluding the aggregate ``total`` entry."""
    timings: Mapping[str, float] = {
        k: v for k, v in result.timings_ms.items() if k != "total"
    }
    names = list(timings.keys())
    values = [timings[n] for n in names]

    fig = go.Figure(
        go.Bar(
            x=names,
            y=values,
            marker_color=PALETTE["cyan"],
            marker_line={"color": "rgba(255,255,255,0.2)", "width": 1},
            text=[f"{v:.0f}" for v in values],
            textposition="outside",
            hovertemplate="%{x}: %{y:.0f} ms<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        yaxis={"title": "ms"},
        xaxis={"title": None, "tickfont": {"size": 10}},
        showlegend=False,
    )
    return fig
