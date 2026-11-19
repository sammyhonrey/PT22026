from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# I have to use "Agg" because we are running this on a server/terminal without a display.
# It saves the plots directly to files instead of trying to open a window.
import matplotlib

matplotlib.use("Agg")  # must be set before importing pyplot

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import load_dataset
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix

from model_loader import REPO_ID, load_model

LOGGER = logging.getLogger("food_classifier.evaluate")
DATASET_ID = "ethz/food101"

# Dataset helpers
def load_test_split(requested_split: str):
    """Load the Food-101 test split, falling back to ``validation``."""
    candidates = [requested_split]
    if requested_split != "validation":
        candidates.append("validation")

    last_error: Exception | None = None
    for split in candidates:
        try:
            LOGGER.info("Loading %s split '%s' ...", DATASET_ID, split)
            dataset = load_dataset(DATASET_ID, split=split)
            if split != requested_split:
                LOGGER.warning(
                    "Split '%s' is not available for %s; using '%s' instead. "
                    "The validation split is the official Food-101 test set.",
                    requested_split,
                    DATASET_ID,
                    split,
                )
            return dataset, split
        except Exception as exc:  # noqa: BLE001 - try the next candidate
            last_error = exc
            LOGGER.warning("Could not load split '%s': %s", split, exc)

    raise RuntimeError(
        f"Could not load any test split from {DATASET_ID}. "
        "Check your internet connection and the `datasets` installation."
    ) from last_error


def get_dataset_label_names(dataset) -> List[str] | None:
    """Return the dataset's ordered class names, if available."""
    try:
        feature = dataset.features["label"]
        names = list(feature.names)
        if names:
            return names
    except Exception:  # noqa: BLE001 - not a ClassLabel feature
        pass
    return None


def build_dataset_to_model_mapping(
    dataset_names: Sequence[str] | None,
    idx_to_label: Dict[int, str],
    num_classes: int,
) -> Tuple[Dict[int, int], bool]:
    """Map dataset label integers to model output indices by class name."""
    if not dataset_names:
        LOGGER.warning(
            "Dataset label names are unavailable; assuming the dataset index "
            "order matches the model's class mapping."
        )
        return {index: index for index in range(num_classes)}, True

    name_to_model_index = {name: index for index, name in idx_to_label.items()}
    mapping: Dict[int, int] = {}
    missing: List[str] = []

    for dataset_index, name in enumerate(dataset_names):
        model_index = name_to_model_index.get(name)
        if model_index is None:
            missing.append(name)
        else:
            mapping[dataset_index] = model_index

    order_matches = all(
        idx_to_label.get(index) == name
        for index, name in enumerate(dataset_names[:num_classes])
    )
    if not order_matches:
        differences = [
            (index, dataset_names[index], idx_to_label.get(index))
            for index in range(min(len(dataset_names), num_classes))
            if dataset_names[index] != idx_to_label.get(index)
        ]
        LOGGER.warning(
            "Dataset label order differs from the model class mapping at %d "
            "index/indices. Using name-based alignment instead. Examples: %s",
            len(differences),
            differences[:5],
        )
    if missing:
        LOGGER.warning(
            "These dataset classes are not present in the model mapping and "
            "will be skipped: %s",
            missing,
        )

    return mapping, order_matches


def sample_indices(
    labels: Sequence[int], max_images: int | None, seed: int
) -> List[int]:
    """Return a deterministic, class-balanced subset of dataset indices."""
    total = len(labels)
    if max_images is None or max_images <= 0 or max_images >= total:
        return list(range(total))

    by_class: Dict[int, List[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_class[int(label)].append(index)

    rng = random.Random(seed)
    for indices in by_class.values():
        rng.shuffle(indices)

    num_classes = len(by_class)
    if max_images < num_classes:
        chosen_classes = rng.sample(sorted(by_class), max_images)
        return sorted(by_class[class_id][0] for class_id in chosen_classes)

    per_class = max(1, max_images // num_classes)
    selected: List[int] = []
    for class_id in sorted(by_class):
        selected.extend(by_class[class_id][:per_class])

    remainder = max_images - len(selected)
    if remainder > 0:
        extras = [
            by_class[class_id][per_class]
            for class_id in sorted(by_class)
            if len(by_class[class_id]) > per_class
        ]
        rng.shuffle(extras)
        selected.extend(extras[:remainder])

    return sorted(selected[:max_images])\
    
# Inference
def run_inference(
    model,
    transform,
    dataset,
    indices: Sequence[int],
    dataset_to_model: Dict[int, int],
    device: torch.device,
    batch_size: int,
    num_classes: int,
) -> Tuple[List[int], List[int], List[List[int]], int]:
    """Run batched inference and return true labels, predictions and top-5."""
    y_true: List[int] = []
    y_pred: List[int] = []
    top5_predictions: List[List[int]] = []
    skipped = 0
    total = len(indices)
    processed = 0
    next_report = 100
    # I processed images in batches to save memory and speed up CPU inference.
    for start in range(0, total, batch_size):
        batch_indices = indices[start : start + batch_size]
        tensors = []
        batch_labels: List[int] = []

        for dataset_index in batch_indices:
            try:
                item = dataset[int(dataset_index)]
                image = item["image"]
                if not isinstance(image, Image.Image):
                    image = Image.fromarray(np.asarray(image))
                image = ImageOps.exif_transpose(image)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                label = int(item["label"])
                tensors.append(transform(image))
                batch_labels.append(label)
            except Exception as exc:  # noqa: BLE001 - skip corrupt samples
                skipped += 1
                LOGGER.warning("Skipping dataset index %s: %s", dataset_index, exc)

        if tensors:
            # torch.stack turns the list of image tensors into one big tensor batch
            batch = torch.stack(tensors).to(device)
            
            # inference_mode() tells PyTorch we are only testing, not training.
            # This saves memory and speeds up the process.
            with torch.inference_mode():
                output = model(batch)
            # Handle the custom output format of our hybrid model
            if isinstance(output, dict):
                logits = output.get("logits")
            elif isinstance(output, (tuple, list)):
                logits = output[0]
            else:
                logits = output

            if not isinstance(logits, torch.Tensor):
                raise RuntimeError("Model did not return a tensor of logits.")
            if logits.ndim != 2:
                raise RuntimeError(
                    f"Unexpected model output shape {tuple(logits.shape)}; "
                    "expected (batch, classes)."
                )
            if logits.shape[0] != len(tensors):
                raise RuntimeError(
                    f"Batch size mismatch: got {logits.shape[0]} logits for "
                    f"{len(tensors)} images."
                )
            if logits.shape[1] != num_classes:
                raise RuntimeError(
                    f"Model returned {logits.shape[1]} classes but the class "
                    f"mapping contains {num_classes}."
                )
            # Softmax turns the raw logits into probabilities (e.g., 0.95, 0.03, 0.02)
            probabilities = torch.softmax(logits.float(), dim=1)
            k = min(5, probabilities.shape[1])
            # Get the index of the top 5 highest probabilities
            top_indices = torch.topk(probabilities, k=k, dim=1).indices.cpu().numpy()

            for position, dataset_label in enumerate(batch_labels):
                model_label = dataset_to_model.get(dataset_label, -1)
                if model_label < 0:
                    skipped += 1
                    continue
                y_true.append(model_label)
                y_pred.append(int(top_indices[position, 0]))
                top5_predictions.append(
                    [int(value) for value in top_indices[position]]
                )

        processed += len(batch_indices)
        if processed >= next_report or processed == total:
            print(f"Processed {processed}/{total} images", flush=True)
            while next_report <= processed:
                next_report += 100

    return y_true, y_pred, top5_predictions, skipped

# Reporting
def save_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    # We make the figure very large (34x32) because a 101x101 grid is huge.
    figure_size = (34, 32) if len(class_names) > 40 else (16, 14)
    fig, ax = plt.subplots(figsize=figure_size)
    # Seaborn's heatmap makes it look nice
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        annot=False,
        square=False,
    )
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(
        f"Food-101 Confusion Matrix (n={len(y_true)} images)", fontsize=16
    )
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=5)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved %s", output_path)


def save_classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    output_path: Path,
) -> Dict:
    labels = list(range(len(class_names)))
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    )
    output_path.write_text(report_text, encoding="utf-8")
    LOGGER.info("Saved %s", output_path)

    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )


def main(argv: Sequence[str] | None = None) -> int:
    # Setup command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate the hybrid Food-101 classifier."
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=1000,
        help="Maximum number of images to evaluate (default: 1000).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Evaluate the entire test split (slow).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Inference batch size."
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use, e.g. cpu, cuda, mps (default: auto).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to try first (default: test; falls back to validation).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: eval_outputs next to this script).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling."
    )
    parser.add_argument(
        "--sample-mode",
        choices=("stratified", "head"),
        default="stratified",
        help=(
            "How to subsample: 'stratified' (default, class-balanced) or "
            "'head' (first N rows, matching the assignment's fallback)."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    script_dir = Path(__file__).resolve().parent
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else script_dir / "eval_outputs"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Model and class mapping 
    LOGGER.info("Loading model from %s ...", REPO_ID)
    model, transform, idx_to_label = load_model(device=args.device)
    device = getattr(model, "device", torch.device("cpu"))
    num_classes = len(idx_to_label)
    class_names = [idx_to_label.get(i, f"class_{i}") for i in range(num_classes)]
    LOGGER.info("Model loaded on %s with %d classes.", device, num_classes)

    # 2. Dataset 
    dataset, split_used = load_test_split(args.split)
    dataset_names = get_dataset_label_names(dataset)
    dataset_to_model, order_matches = build_dataset_to_model_mapping(
        dataset_names, idx_to_label, num_classes
    )

    labels = dataset["label"]
    max_images = None if args.full else args.max_images
    if args.sample_mode == "head" and max_images is not None and max_images > 0:
        indices = list(range(min(max_images, len(dataset))))
    else:
        indices = sample_indices(labels, max_images, args.seed)
    LOGGER.info(
        "Evaluating %d image(s) from split '%s' (dataset size: %d).",
        len(indices),
        split_used,
        len(dataset),
    )

    # 3. Inference 
    y_true, y_pred, top5_predictions, skipped = run_inference(
        model=model,
        transform=transform,
        dataset=dataset,
        indices=indices,
        dataset_to_model=dataset_to_model,
        device=device,
        batch_size=max(1, args.batch_size),
        num_classes=num_classes,
    )

    if not y_true:
        raise RuntimeError(
            "No images were evaluated. The dataset could not be read or all "
            "samples were skipped."
        )
     # Calculate top-1 and top-5 accuracy
    top1_accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    top5_accuracy = float(
        np.mean(
            [
                true_label in predictions
                for true_label, predictions in zip(y_true, top5_predictions)
            ]
        )
    )

    # 4. Reports 
    report_dict = save_classification_report(
        y_true,
        y_pred,
        class_names,
        output_dir / "classification_report.txt",
    )
    save_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        output_dir / "confusion_matrix.png",
    )
     # Save detailed per-class metrics to JSON
    per_class: Dict[str, Dict[str, float | int]] = {}
    for name in class_names:
        entry = report_dict.get(name, {})
        per_class[name] = {
            "precision": float(entry.get("precision", 0.0)),
            "recall": float(entry.get("recall", 0.0)),
            "f1_score": float(entry.get("f1-score", 0.0)),
            "support": int(entry.get("support", 0)),
        }

    results = {
        "model_repo": REPO_ID,
        "dataset": DATASET_ID,
        "split_used": split_used,
        "requested_split": args.split,
        "num_classes": num_classes,
        "num_images_requested": len(indices),
        "num_images_evaluated": len(y_true),
        "num_skipped": skipped,
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "dataset_label_order_matches_model": bool(order_matches),
        "per_class": per_class,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    results_path = output_dir / "eval_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Saved %s", results_path)

    print("\n" + "=" * 60)
    print(f"Evaluated images : {len(y_true)} (skipped: {skipped})")
    print(f"Top-1 accuracy   : {top1_accuracy * 100:.2f}%")
    print(f"Top-5 accuracy   : {top5_accuracy * 100:.2f}%")
    print(f"Outputs saved to : {output_dir}")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
