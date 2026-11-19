#!/usr/bin/env python
"""Render the report figures as PNGs with matplotlib.

Figures come from a fixture analysis that runs the real fusion, stress and
narration code over illustrative branch inputs; every caption says so.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _fixtures import build_fixture_pipeline, write_tone  # noqa: E402
from mmsa.config import load_config  # noqa: E402
from mmsa.pipeline import safe_analyse  # noqa: E402

FIG_DIR = ROOT / "report" / "figures"
RESULTS = ROOT / "results"

BG = "#0B1226"
PANEL = "#101A38"
TEXT = "#E8EEF9"
MUTED = "#93A4C3"
CYAN = "#22D3EE"
VIOLET = "#A78BFA"
MAGENTA = "#F472B6"
AMBER = "#FBBF24"
ROSE = "#FB7185"
GREEN = "#34D399"

EMOTION_COLOURS = {
    "happy": AMBER,
    "sad": "#60A5FA",
    "angry": ROSE,
    "fear": VIOLET,
    "disgust": GREEN,
    "neutral": "#94A3B8",
}

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": TEXT,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": "#26324D",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.grid": True,
        "grid.color": "#1E2A45",
        "grid.linewidth": 0.8,
        "figure.dpi": 100,
    }
)


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    target = FIG_DIR / name
    fig.savefig(target, dpi=200, bbox_inches="tight", pad_inches=0.28, facecolor=BG)
    plt.close(fig)
    print(f"  {name}")
    return target


def fig_branch_bars(result) -> Path:
    """Grouped horizontal bars: acoustic mass, text mass, fused distribution."""
    labels = list(result.fused_probs.keys())
    acoustic = [result.acoustic.mapped_probs.get(k, 0.0) for k in labels]
    text = [result.text.mapped_probs.get(k, 0.0) for k in labels]
    fused = [result.fused_probs.get(k, 0.0) for k in labels]

    y = np.arange(len(labels))
    h = 0.26

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.barh(y - h, acoustic, h, label=f"Acoustic (coverage {result.acoustic.coverage:.0%})",
            color=CYAN, alpha=0.88)
    ax.barh(y, text, h, label=f"Text (coverage {result.text.coverage:.0%})",
            color=VIOLET, alpha=0.88)
    ax.barh(y + h, fused, h, label="Fused",
            color=[EMOTION_COLOURS.get(k, MUTED) for k in labels], edgecolor="#FFFFFF22")

    for values, offset in ((acoustic, -h), (text, 0.0), (fused, h)):
        for yi, v in zip(y, values):
            ax.text(v + 0.012, yi + offset, f"{v:.2f}", va="center", fontsize=7.5, color=MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels([k.capitalize() for k in labels])
    ax.invert_yaxis()
    ax.set_xlim(0, max(acoustic + text + fused + [0.1]) * 1.16)
    ax.set_xlabel("probability mass")
    ax.set_title("Branch opinions and the fused distribution", color=TEXT, fontsize=12, pad=12, loc="left")
    ax.legend(facecolor=PANEL, edgecolor="#26324D", labelcolor=TEXT, fontsize=8, loc="lower right")
    return _save(fig, "fig_branch_bars.png")


def fig_radar(result) -> Path:
    """Polar overlay: the *shape* of disagreement."""
    labels = list(result.fused_probs.keys())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(5.4, 5.0))
    ax = fig.add_subplot(111, polar=True)

    for name, values, colour in (
        ("Acoustic", [result.acoustic.mapped_probs.get(k, 0.0) for k in labels], CYAN),
        ("Text", [result.text.mapped_probs.get(k, 0.0) for k in labels], VIOLET),
        ("Fused", [result.fused_probs.get(k, 0.0) for k in labels], MAGENTA),
    ):
        closed = values + [values[0]]
        ax.plot(angles, closed, color=colour, linewidth=1.9, label=name)
        ax.fill(angles, closed, color=colour, alpha=0.16)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([k.capitalize() for k in labels], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "0.5", "", "1.0"], fontsize=7, color=MUTED)
    ax.set_title("Where the channels disagree", color=TEXT, fontsize=11, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.12), frameon=False, labelcolor=TEXT, fontsize=8)
    ax.grid(color="#1E2A45")
    ax.spines["polar"].set_color("#26324D")
    return _save(fig, "fig_radar.png")


def fig_stress(result) -> Path:
    """Signed contributors to the stress logit."""
    contributors = dict(result.stress.contributors)
    names = sorted(contributors, key=lambda k: contributors[k])
    values = [contributors[n] for n in names]

    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    colours = [ROSE if v > 0 else CYAN for v in values]
    ax.barh([n.replace("_", " ").capitalize() for n in names], values, color=colours, height=0.62)
    ax.axvline(0, color="#FFFFFF55", linewidth=1)

    span = max(abs(v) for v in values) or 0.1
    for i, v in enumerate(values):
        ax.text(v + (0.012 if v >= 0 else -0.012) * span, i, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8, color=MUTED)

    ax.set_xlabel("contribution to logit (signed)")
    ax.set_title(
        f"Stress decomposition — score {result.stress.score:.2f} ({result.stress.band_label})",
        color=TEXT, fontsize=11, pad=12, loc="left",
    )
    ax.grid(axis="y", visible=False)
    return _save(fig, "fig_stress.png")


def fig_latency(result) -> Path:
    """Per-stage latency."""
    timings = {k: v for k, v in result.timings_ms.items() if k != "total"}
    names = list(timings)
    values = [timings[n] for n in names]

    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    bars = ax.bar(names, values, color=CYAN, width=0.58)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom",
                fontsize=8, color=MUTED)
    ax.set_ylabel("milliseconds")
    ax.set_title("Orchestration latency by stage (model inference excluded)",
                 color=TEXT, fontsize=11, pad=12, loc="left")
    ax.grid(axis="x", visible=False)
    return _save(fig, "fig_latency.png")


def fig_zeroshot_results() -> Path | None:
    """Measured per-class F1 for the two branches, if a measurement exists."""
    path = RESULTS / "zeroshot.json"
    if not path.exists():
        print("  (no results/zeroshot.json — skipping measured branch figure)")
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload["acoustic"]["labels"]
    ac = [payload["acoustic"]["per_class"][label]["f1"] for label in labels]
    tx = [payload["text"]["per_class"][label]["f1"] for label in labels]

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.bar(x - w / 2, ac, w, label="Acoustic", color=CYAN)
    ax.bar(x + w / 2, tx, w, label="Text", color=VIOLET)
    ax.set_xticks(x)
    ax.set_xticklabels([l.capitalize() for l in labels])
    ax.set_ylabel("per-class F1")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Zero-shot transfer on RAVDESS — acoustic {payload['acoustic']['macro_f1']:.3f} macro-F1, "
        f"text {payload['text']['macro_f1']:.3f}",
        color=TEXT, fontsize=10.5, pad=12, loc="left",
    )
    ax.legend(facecolor=PANEL, edgecolor="#26324D", labelcolor=TEXT, fontsize=8)
    return _save(fig, "fig_zeroshot.png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-results", action="store_true", help="also render measured figures")
    parser.add_argument("--emotion", default="neutral", help="fixture profile (default: %(default)s)")
    args = parser.parse_args(argv)

    cfg = load_config()
    pipeline = build_fixture_pipeline(cfg, emotion=args.emotion)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"rendering figures into {FIG_DIR}")

    with tempfile.TemporaryDirectory() as tmp:
        clip = write_tone(Path(tmp) / "figure_clip.wav", seconds=3.0)
        result, error = safe_analyse(pipeline, str(clip))
        if error is not None or result is None:
            print(f"fixture analysis failed: {getattr(error, 'code', '?')}")
            return 1

        fig_branch_bars(result)
        fig_radar(result)
        fig_stress(result)
        fig_latency(result)

    if args.from_results:
        fig_zeroshot_results()

    print("\ndone. Figures are embedded by report/report.html.")
    print("Captions mark the fixture-derived figures as illustrative.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
