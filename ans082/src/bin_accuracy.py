"""Class-level and bin-level accuracy from the saved predictions.

The app tells the user which BIN to use. Many mistakes between classes (for example plastic and
metal) still lead to the same bin, so the bin-level accuracy shows how often the advice would
still have been right.

Run from the ans082 folder, for example:
    python src/bin_accuracy.py
Writes results/bin_level_accuracy.csv and results/bin_level_accuracy.md.
"""
import argparse
from pathlib import Path

import pandas as pd

from bins import bin_of

EXPERIMENTS = ["E1_resnet50", "E2_vit_head_only", "E3_vit_full", "E4_vit_full_strong_aug"]


def bin_level_accuracy(predictions: pd.DataFrame) -> dict:
    """Class accuracy and bin accuracy for one predictions table (columns true, pred)."""
    class_ok = predictions["true"] == predictions["pred"]
    bin_ok = predictions["true"].map(bin_of) == predictions["pred"].map(bin_of)
    return {
        "n_images": int(len(predictions)),
        "class_accuracy": round(float(class_ok.mean()), 4),
        "bin_accuracy": round(float(bin_ok.mean()), 4),
    }


def to_markdown(table: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(table.columns) + " |",
             "| " + " | ".join("---" for _ in table.columns) + " |"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-dir", type=Path, default=Path("results/eval"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    rows = []
    for name in EXPERIMENTS:
        for split in ("test", "real"):
            predictions = pd.read_csv(args.eval_dir / f"{name}_{split}_predictions.csv")
            rows.append({"experiment": name, "split": split, **bin_level_accuracy(predictions)})
    table = pd.DataFrame(rows)
    table.to_csv(args.results_dir / "bin_level_accuracy.csv", index=False)
    (args.results_dir / "bin_level_accuracy.md").write_text(to_markdown(table), encoding="utf-8")
    print(to_markdown(table))


if __name__ == "__main__":
    main()
