"""Build the Phase 7 report figures and the final results table from the saved evaluation files.

Reads the files written by evaluate.py (results/eval) and writes:
    results/final_results.csv, results/final_results.md   test vs real-photo numbers, E1 to E4
    report_figures/test_vs_real.png                       macro-F1 on test vs real photos
    report_figures/confidence_hist_<final>.png            confidence of correct vs wrong predictions
    report_figures/threshold_curve_<final>.png            coverage vs accuracy for several thresholds
    report_figures/error_panel_<final>_real.jpg           the most confident wrong real-photo predictions

Run from the ans082 folder, for example:
    python src/make_report_figures.py --real-root "F:/Studies/Projects/SmartSort AI/real_photos"
"""
import argparse
import json
from pathlib import Path, PurePosixPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

EXPERIMENTS = ["E1_resnet50", "E2_vit_head_only", "E3_vit_full", "E4_vit_full_strong_aug"]


def load_final_table(eval_dir, experiments=EXPERIMENTS) -> pd.DataFrame:
    """One row per experiment: accuracy and macro-F1 on the test split and on the real photos."""
    rows = []
    for name in experiments:
        row = {"experiment": name}
        for split in ("test", "real"):
            metrics = json.loads((Path(eval_dir) / f"{name}_{split}_metrics.json").read_text())
            row[f"{split}_acc"] = round(metrics["accuracy"], 4)
            row[f"{split}_macro_f1"] = round(metrics["macro_f1"], 4)
        row["gap_macro_f1"] = round(row["test_macro_f1"] - row["real_macro_f1"], 4)
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(table: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(table.columns) + " |",
             "| " + " | ".join("---" for _ in table.columns) + " |"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def plot_test_vs_real(table: pd.DataFrame, out_png) -> None:
    """Grouped bars: macro-F1 on the test split and on the real photos (axis starts at zero)."""
    positions = np.arange(len(table))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    test_bars = ax.bar(positions - width / 2, table["test_macro_f1"], width, label="Test split")
    real_bars = ax.bar(positions + width / 2, table["real_macro_f1"], width, label="Real phone photos")
    ax.bar_label(test_bars, fmt="%.3f", fontsize=8)
    ax.bar_label(real_bars, fmt="%.3f", fontsize=8)
    ax.set_xticks(positions)
    ax.set_xticklabels(table["experiment"], rotation=20, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Domain-shift gap: test split vs real phone photos")
    ax.legend(loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_confidence_hist(pred_val: pd.DataFrame, pred_real: pd.DataFrame, out_png,
                         threshold: float, title: str) -> None:
    """Histogram of the model's confidence for correct and wrong predictions."""
    bins = np.linspace(0, 1, 21)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, table, label in zip(axes, (pred_val, pred_real), ("Validation split", "Real phone photos")):
        ax.hist(table.loc[table["correct"], "confidence"], bins=bins, alpha=0.7,
                label=f"correct (n={int(table['correct'].sum())})")
        ax.hist(table.loc[~table["correct"], "confidence"], bins=bins, alpha=0.7,
                label=f"wrong (n={int((~table['correct']).sum())})")
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1,
                   label=f"threshold {threshold:.2f}")
        ax.set_title(label)
        ax.set_xlabel("Confidence of the predicted class")
        ax.set_ylabel("Images")
        ax.legend(loc="upper left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_threshold_curve(thr_val: pd.DataFrame, thr_real: pd.DataFrame, out_png, title: str) -> None:
    """Coverage (share of images answered) against accuracy on the answered images."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for table, label in ((thr_val, "validation"), (thr_real, "real photos")):
        table = table.dropna(subset=["accuracy_on_accepted"])
        ax.plot(table["coverage"], table["accuracy_on_accepted"], marker="o", label=label)
        for _, row in table.iterrows():
            if label == "validation" and row["threshold"] < 0.8:
                continue
            ax.annotate(f"{row['threshold']:.2f}", (row["coverage"], row["accuracy_on_accepted"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("Coverage (share of images the model answers)")
    ax.set_ylabel("Accuracy on the answered images")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def resolve_photo(path_text: str, real_root) -> Path:
    """Map a Colab/Drive path from the results onto the local photo folder (class/file)."""
    parts = PurePosixPath(str(path_text).replace("\\", "/")).parts
    return Path(real_root) / parts[-2] / parts[-1]


def plot_error_panel(top_errors: pd.DataFrame, real_root, out_png, title: str, columns: int = 4) -> None:
    """Grid of the most confident wrong predictions with true class, prediction and confidence."""
    rows = int(np.ceil(len(top_errors) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, row) in zip(axes.ravel(), top_errors.iterrows()):
        photo = resolve_photo(row["path"], real_root)
        if photo.exists():
            with Image.open(photo) as img:
                thumb = img.convert("RGB")
                thumb.thumbnail((400, 400))          # small thumbnails keep the figure file small
                ax.imshow(thumb)
        else:
            ax.text(0.5, 0.5, f"missing\n{photo.name}", ha="center", va="center", fontsize=8)
        ax.set_title(f"{photo.stem}\n{row['true']} -> {row['pred']} ({row['confidence']:.2f})", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, pil_kwargs={"quality": 85})   # JPEG keeps the photo grid small
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-dir", type=Path, default=Path("results/eval"))
    parser.add_argument("--real-root", type=Path, required=True, help="local folder with the resized real photos")
    parser.add_argument("--out-dir", type=Path, default=Path("report_figures"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--final", default="E3_vit_full", help="the selected final model")
    parser.add_argument("--threshold", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    table = load_final_table(args.eval_dir)
    table.to_csv(args.results_dir / "final_results.csv", index=False)
    (args.results_dir / "final_results.md").write_text(to_markdown(table), encoding="utf-8")
    print(to_markdown(table))
    plot_test_vs_real(table, args.out_dir / "test_vs_real.png")

    pred_val = pd.read_csv(args.eval_dir / f"{args.final}_val_predictions.csv")
    pred_real = pd.read_csv(args.eval_dir / f"{args.final}_real_predictions.csv")
    plot_confidence_hist(pred_val, pred_real, args.out_dir / f"confidence_hist_{args.final}.png",
                         args.threshold, f"{args.final}: confidence of correct and wrong predictions")

    thr_val = pd.read_csv(args.eval_dir / f"{args.final}_val_thresholds.csv")
    thr_real = pd.read_csv(args.eval_dir / f"{args.final}_real_thresholds.csv")
    plot_threshold_curve(thr_val, thr_real, args.out_dir / f"threshold_curve_{args.final}.png",
                         f"{args.final}: coverage vs accuracy")

    errors = pd.read_csv(args.eval_dir / f"{args.final}_real_top_errors.csv")
    plot_error_panel(errors, args.real_root, args.out_dir / f"error_panel_{args.final}_real.jpg",
                     f"{args.final}: most confident wrong predictions on real photos")
    print(f"Saved figures to {args.out_dir}")


if __name__ == "__main__":
    main()
