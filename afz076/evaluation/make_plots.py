"""Turn the evaluation JSON files into report-ready figures (PNG).

Usage:  python evaluation/make_plots.py                (reads evaluation/results/*.json)
Output: evaluation/figures/{robustness,reliability,coverage,training_curves}.png
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CONDITIONS  # noqa: E402

RES = Path("evaluation/results")
FIG = Path("evaluation/figures")


def load_results():
    return [json.loads(p.read_text()) for p in sorted(RES.glob("*.json"))]


def plot_robustness(results):
    """Accuracy vs severity, one panel per condition, one line per trained run."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for ax, cond in zip(axes.ravel(), CONDITIONS):
        for r in results:
            pts = [(0, r["clean"]["accuracy"])] + [(c["severity"], c["accuracy"]) for c in r["conditions"]
                                                    if c["condition"] == cond]
            pts.sort()
            ax.plot(*zip(*pts), marker="o", label=r["name"])
        ax.set_title(cond + (" (never seen in training)" if cond == "glare" else ""))
        ax.set_xlabel("severity (0 = clean)")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("test accuracy")
    axes[1, 0].set_ylabel("test accuracy")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "robustness.png", dpi=150)


def plot_reliability(results):
    """Reliability diagram on clean test data, before and after temperature scaling."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, key, title in zip(axes, ("reliability_clean_raw", "reliability_clean_temp"),
                              ("Before temperature scaling", "After temperature scaling")):
        ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
        for r in results:
            pts = [(b["confidence"], b["accuracy"]) for b in r[key] if b["confidence"] is not None]
            if pts:
                ax.plot(*zip(*pts), marker="o", label=r["name"])
        ax.set_title(title)
        ax.set_xlabel("confidence")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "reliability.png", dpi=150)


def plot_coverage(results):
    """Accuracy on answered inputs vs the share of inputs answered (reject option), clean and weather."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, key, title in zip(axes, ("selective_clean", "selective_sev3_pooled"),
                              ("Clean test images", "All conditions, severity 3")):
        for r in results:
            rows = [x for x in r.get(key, []) if x["accuracy_on_answered"] is not None]
            if rows:
                ax.plot([x["coverage"] for x in rows], [x["accuracy_on_answered"] for x in rows],
                        marker="o", label=r["name"])
        ax.set_title(title)
        ax.set_xlabel("coverage (share of images answered)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy on answered images")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "coverage.png", dpi=150)


def plot_training_curves():
    """Training loss and validation accuracy per epoch for every run found in outputs/."""
    files = sorted(Path("outputs").glob("*/history.json"))
    if not files:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for f in files:
        h = json.loads(f.read_text())["history"]
        axes[0].plot([r["epoch"] for r in h], [r["train_loss"] for r in h], marker="o", label=f.parent.name)
        axes[1].plot([r["epoch"] for r in h], [r["val_acc"] for r in h], marker="o", label=f.parent.name)
    axes[0].set(title="Training loss", xlabel="epoch")
    axes[1].set(title="Validation accuracy (clean)", xlabel="epoch")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "training_curves.png", dpi=150)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    results = load_results()
    if not results:
        raise SystemExit("No result files found in evaluation/results/. Run evaluation/evaluate.py first.")
    plot_robustness(results)
    plot_reliability(results)
    plot_coverage(results)
    plot_training_curves()
    print(f"Saved figures to {FIG}/")


if __name__ == "__main__":
    main()
