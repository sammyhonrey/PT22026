"""Collect the *_summary.json files of all experiments into one comparison table.

Run from the ans082 folder, for example:
    python src/summarize_runs.py --results-dir results

It writes experiments_val.csv and experiments_val.md next to the summaries and a bar chart
to figures/experiments_val.png. These are the validation results only. Test and real-photo
results are added in Phase 7.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

COLUMNS = [
    "experiment", "model", "trainable_params_M", "total_params_M", "best_epoch",
    "best_val_acc", "best_val_macro_f1", "total_train_min", "mean_epoch_s",
]


def collect(results_dir) -> pd.DataFrame:
    """Read every *_summary.json in results_dir and return one row per experiment."""
    rows = []
    for path in sorted(Path(results_dir).glob("*_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "experiment": summary["experiment"],
            "model": summary["model"],
            "trainable_params_M": round(summary["trainable_parameters"] / 1e6, 3),
            "total_params_M": round(summary["total_parameters"] / 1e6, 2),
            "best_epoch": summary["best_epoch"],
            "best_val_acc": round(summary["best_val_acc"], 4),
            "best_val_macro_f1": round(summary["best_val_macro_f1"], 4),
            "total_train_min": round(summary["total_train_time_s"] / 60, 1),
            "mean_epoch_s": round(summary["mean_epoch_time_s"], 1),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def to_markdown(table: pd.DataFrame) -> str:
    """Render the table as a Markdown table (no extra packages needed)."""
    lines = ["| " + " | ".join(table.columns) + " |",
             "| " + " | ".join("---" for _ in table.columns) + " |"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def plot_comparison(table: pd.DataFrame, out_png) -> None:
    """Bar chart of validation accuracy and macro-F1 for every experiment."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = list(range(len(table)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    acc_bars = ax.bar([p - width / 2 for p in positions], table["best_val_acc"], width,
                      label="Validation accuracy")
    f1_bars = ax.bar([p + width / 2 for p in positions], table["best_val_macro_f1"], width,
                     label="Validation macro-F1")
    ax.bar_label(acc_bars, fmt="%.3f", fontsize=8)
    ax.bar_label(f1_bars, fmt="%.3f", fontsize=8)
    ax.set_xticks(positions)
    ax.set_xticklabels(table["experiment"], rotation=20, ha="right")
    lowest = min(table["best_val_acc"].min(), table["best_val_macro_f1"].min())
    ax.set_ylim(max(0.0, lowest - 0.03), 1.0)
    ax.set_ylabel("Score (the axis does not start at zero)")
    ax.set_title("Validation results by experiment")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    table = collect(args.results_dir)
    if table.empty:
        raise SystemExit(f"No *_summary.json files found in {args.results_dir}")
    out_dir = Path(args.results_dir)
    table.to_csv(out_dir / "experiments_val.csv", index=False)
    (out_dir / "experiments_val.md").write_text(to_markdown(table), encoding="utf-8")
    plot_comparison(table, out_dir / "figures" / "experiments_val.png")
    print(to_markdown(table))


if __name__ == "__main__":
    main()
