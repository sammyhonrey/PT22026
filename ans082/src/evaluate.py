"""Evaluate a trained SmartSort checkpoint on the validation split, the test split or real photos.

Every run writes, into the output folder, files that start with <experiment>_<split>:
    _predictions.csv   one row per image: path, true, pred, confidence, correct, p_<class>...
    _metrics.json      accuracy, macro precision / recall / F1 and the per-class report
    _confusion.png     normalised confusion matrix
    _thresholds.csv    coverage and accuracy for several confidence thresholds
    _top_errors.csv    the most confident wrong predictions

The test split and the real photos may be used ONCE, at the very end, so they need the
--confirm-final flag. Use --split val for every dry run.

Run from the ans082 folder, for example:
    python src/evaluate.py --experiment E3_vit_full --split val
"""
import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from torch.utils.data import Dataset

from data import CLASS_TO_IDX, CLASSES, SmartSortDataset, build_loader, build_transforms, load_rgb
from model_loader import build_model

logger = logging.getLogger(__name__)

# Where the Colab runs keep their files. Set these environment variables to use other folders.
DRIVE_ROOT = Path(os.environ.get("SMARTSORT_DRIVE_ROOT", "/content/drive/MyDrive/SmartSort_AI"))
DATA_ROOT = os.environ.get("SMARTSORT_DATA_ROOT", "/content/data/garbage_classification")

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
DEFAULT_THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
FINAL_SPLITS = ("test", "real")


class ImageListDataset(Dataset):
    """A dataset built from explicit lists of image paths and numeric labels."""

    def __init__(self, paths, labels, transform=None):
        self.paths = [Path(p) for p in paths]
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        image = load_rgb(self.paths[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]


def list_real_photos(root):
    """Return (paths, labels) for a folder with one sub-folder per class name."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Real-photo folder not found: {root}")
    paths, labels = [], []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if class_dir.name not in CLASS_TO_IDX:
            raise ValueError(f"Folder '{class_dir.name}' is not one of the classes {CLASSES}")
        for file in sorted(class_dir.iterdir()):
            if file.suffix.lower() in IMAGE_EXTS:
                paths.append(file)
                labels.append(CLASS_TO_IDX[class_dir.name])
    if not paths:
        raise ValueError(f"No images found under {root}")
    return paths, labels


def load_checkpoint(ckpt_path, device):
    """Rebuild the model saved in best.pt and return (model, checkpoint dictionary)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if list(ckpt["classes"]) != list(CLASSES):
        raise ValueError("The checkpoint's class order differs from data.CLASSES")
    model_cfg = dict(ckpt["model_cfg"])
    model_cfg["pretrained"] = False          # the trained weights are loaded below
    model_cfg["mean"], model_cfg["std"] = ckpt["mean"], ckpt["std"]
    model, _, _ = build_model(model_cfg)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def predict_probs(model, loader, device, use_amp=False) -> np.ndarray:
    """Return softmax probabilities, shape (number of images, number of classes)."""
    model.eval()
    chunks = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        chunks.append(torch.softmax(logits.float(), dim=1).cpu())
    return torch.cat(chunks).numpy()


def make_predictions_table(paths, labels, probs) -> pd.DataFrame:
    """One row per image with the true class, the predicted class and the confidence."""
    labels = np.asarray(labels)
    preds = probs.argmax(axis=1)
    table = pd.DataFrame({
        "path": [str(p) for p in paths],
        "true": [CLASSES[i] for i in labels],
        "pred": [CLASSES[i] for i in preds],
        "confidence": probs.max(axis=1).round(4),
    })
    table["correct"] = table["true"] == table["pred"]
    for idx, name in enumerate(CLASSES):
        table[f"p_{name}"] = probs[:, idx].round(4)
    return table


def compute_metrics(labels, preds) -> dict:
    """Accuracy, macro precision / recall / F1 and the per-class report."""
    idx = list(range(len(CLASSES)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=idx, average="macro", zero_division=0)
    report = classification_report(labels, preds, labels=idx, target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    return {
        "n_images": int(len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "per_class": {name: {k: float(v) for k, v in report[name].items()} for name in CLASSES},
    }


def threshold_table(confidence, correct, thresholds=DEFAULT_THRESHOLDS) -> pd.DataFrame:
    """Coverage and accuracy of the predictions that pass each confidence threshold."""
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    rows = []
    for threshold in thresholds:
        accepted = confidence >= threshold
        n_accepted = int(accepted.sum())
        rows.append({
            "threshold": threshold,
            "n_accepted": n_accepted,
            "coverage": round(n_accepted / len(confidence), 4),
            "accuracy_on_accepted": round(float(correct[accepted].mean()), 4) if n_accepted else None,
        })
    return pd.DataFrame(rows)


def top_confident_errors(table: pd.DataFrame, k: int = 12) -> pd.DataFrame:
    """The k wrong predictions the model was most sure about."""
    wrong = table[~table["correct"]].sort_values("confidence", ascending=False)
    return wrong.head(k)[["path", "true", "pred", "confidence"]].reset_index(drop=True)


def plot_confusion(labels, preds, out_png, title="") -> None:
    """Save a row-normalised confusion matrix (each row sums to 1)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = list(range(len(CLASSES)))
    matrix = confusion_matrix(labels, preds, labels=idx, normalize="true")
    matrix = np.nan_to_num(matrix)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(idx)
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(idx)
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    for i in idx:
        for j in idx:
            if matrix[i, j] >= 0.005:
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if matrix[i, j] > 0.5 else "black")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def run_evaluation(ckpt_path, split, split_csv, data_root, real_root, out_dir, experiment,
                   image_size=224, batch_size=32, workers=2, max_images=None) -> dict:
    """Evaluate one checkpoint on one split and write every output file. Returns the metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(ckpt_path, device)
    _, eval_tf = build_transforms(ckpt["mean"], ckpt["std"], image_size)

    if split == "real":
        paths, labels = list_real_photos(real_root)
        dataset = ImageListDataset(paths, labels, eval_tf)
    else:
        dataset = SmartSortDataset(split_csv, data_root, split, eval_tf, max_samples=max_images)
        paths, labels = dataset.paths, dataset.labels
    loader = build_loader(dataset, batch_size, False, workers)

    probs = predict_probs(model, loader, device, use_amp=device.type == "cuda")
    table = make_predictions_table(paths, labels, probs)
    metrics = compute_metrics(labels, probs.argmax(axis=1))
    metrics.update({"experiment": experiment, "split": split,
                    "checkpoint_epoch": int(ckpt["epoch"]), "device": device.type})

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{experiment}_{split}"
    table.to_csv(out_dir / f"{prefix}_predictions.csv", index=False)
    threshold_table(table["confidence"], table["correct"]).to_csv(
        out_dir / f"{prefix}_thresholds.csv", index=False)
    top_confident_errors(table).to_csv(out_dir / f"{prefix}_top_errors.csv", index=False)
    plot_confusion(labels, probs.argmax(axis=1), out_dir / f"{prefix}_confusion.png",
                   title=f"{experiment} on {split} (n={len(labels)})")
    with open(out_dir / f"{prefix}_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    logger.info("%s on %s: n=%d accuracy %.4f macro-F1 %.4f", experiment, split,
                metrics["n_images"], metrics["accuracy"], metrics["macro_f1"])
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, help="e.g. E3_vit_full")
    parser.add_argument("--split", required=True, choices=["val", "test", "real"])
    parser.add_argument("--confirm-final", action="store_true",
                        help="required for the test split and the real photos (single use)")
    parser.add_argument("--ckpt-dir", default=str(DRIVE_ROOT / "checkpoints"))
    parser.add_argument("--split-csv", default="splits/split_v1.csv")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--real-root", default=str(DRIVE_ROOT / "real_photos"))
    parser.add_argument("--out-dir", default=str(DRIVE_ROOT / "outputs" / "eval"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-images", type=int, help="use only N images (smoke test, val only)")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.split in FINAL_SPLITS and not args.confirm_final:
        raise SystemExit(f"The '{args.split}' data is for the final evaluation only. "
                         "Dry runs must use --split val. Add --confirm-final when you are sure.")
    if args.split in FINAL_SPLITS and args.max_images:
        raise SystemExit("--max-images is for smoke tests on the validation split only.")
    ckpt_path = Path(args.ckpt_dir) / args.experiment / "best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    run_evaluation(ckpt_path, args.split, args.split_csv, args.data_root, args.real_root,
                   args.out_dir, args.experiment, args.image_size, args.batch_size,
                   args.workers, args.max_images)


if __name__ == "__main__":
    main()