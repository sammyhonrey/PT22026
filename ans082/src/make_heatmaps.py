"""Make the Grad-CAM examples for the report: 3 correct and 3 wrong predictions on real photos.

Selection rules (fixed in advance, so the examples are not hand-picked):
  * correct: the most confident correct prediction of each class, then the 3 most confident of those
  * wrong:   the 3 most confident wrong predictions
The heatmap always explains the class the model predicted.

Writes into report_figures/:
    heatmaps/<kind>_<photo>.jpg   one small overlay image per example (6 files)
    heatmap_panel_<experiment>.jpg   all six, with the original crop above each heatmap

Run from the ans082 folder (CPU is enough), for example:
    python src/make_heatmaps.py --real-root "F:/Studies/Projects/SmartSort AI/real_photos"
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from data import CLASS_TO_IDX  # noqa: E402
from evaluate import load_checkpoint  # noqa: E402
from explain import ViTExplainer  # noqa: E402
from make_report_figures import resolve_photo  # noqa: E402


def choose_examples(predictions: pd.DataFrame, n: int = 3):
    """Return (correct, wrong) tables chosen by the fixed rules above."""
    correct = (predictions[predictions["correct"]].sort_values("confidence", ascending=False)
               .drop_duplicates("true").head(n))
    wrong = predictions[~predictions["correct"]].sort_values("confidence", ascending=False).head(n)
    return correct, wrong


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="E3_vit_full")
    parser.add_argument("--ckpt", type=Path, default=Path("models/E3_vit_full/best.pt"))
    parser.add_argument("--eval-dir", type=Path, default=Path("results/eval"))
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("report_figures"))
    parser.add_argument("--n", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.eval_dir / f"{args.experiment}_real_predictions.csv")
    correct, wrong = choose_examples(predictions, args.n)
    examples = [("correct", row) for _, row in correct.iterrows()] + \
               [("wrong", row) for _, row in wrong.iterrows()]

    model, ckpt = load_checkpoint(args.ckpt, torch.device("cpu"))
    explainer = ViTExplainer(model, ckpt["mean"], ckpt["std"])
    heat_dir = args.out_dir / "heatmaps"
    heat_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, len(examples), figsize=(2.6 * len(examples), 6.2), squeeze=False)
    for column, (kind, row) in enumerate(examples):
        photo = resolve_photo(row["path"], args.real_root)
        image = Image.open(photo).convert("RGB")
        overlay, _, probs, index = explainer.explain(image, class_index=CLASS_TO_IDX[row["pred"]])
        if int(probs.argmax()) != index:
            print(f"warning: {photo.name}: the CPU prediction differs from the saved one")
        Image.fromarray(overlay).save(heat_dir / f"{kind}_{photo.stem}.jpg", quality=88)
        axes[0, column].imshow(explainer.view(image))
        axes[1, column].imshow(overlay)
        title = f"{photo.stem}\n" + (f"correct: {row['true']}" if kind == "correct"
                                      else f"{row['true']} -> {row['pred']}") + f" ({row['confidence']:.2f})"
        axes[0, column].set_title(title, fontsize=8)
        for ax in axes[:, column]:
            ax.axis("off")
        print(f"{kind:8s}{photo.name:22s} true {row['true']:11s} pred {row['pred']:11s} "
              f"conf {row['confidence']:.2f}")
    axes[0, 0].text(-0.05, 0.5, "photo", transform=axes[0, 0].transAxes, rotation=90,
                    va="center", ha="right", fontsize=9)
    axes[1, 0].text(-0.05, 0.5, "Grad-CAM", transform=axes[1, 0].transAxes, rotation=90,
                    va="center", ha="right", fontsize=9)
    fig.suptitle(f"{args.experiment}: Grad-CAM on real photos (red = most important)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out_dir / f"heatmap_panel_{args.experiment}.jpg", dpi=130,
                pil_kwargs={"quality": 85})
    plt.close(fig)
    explainer.close()
    print(f"Saved {len(examples)} heatmaps to {heat_dir} and the panel to {args.out_dir}")


if __name__ == "__main__":
    main()
