"""Generate Grad-CAM heatmaps for the hybrid Food-101 classifier.

This script scans the Food-101 validation dataset to find examples where the 
model is correct and incorrect. It then generates Grad-CAM visual heatmaps 
to show where the model is focusing its attention, saving the results to eval_outputs/.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from PIL import Image, ImageOps
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model_loader import load_model

LOGGER = logging.getLogger("food_classifier.gradcam")
DATASET_ID = "ethz/food101"
IMAGE_SIZE = 224

# Model helpers
class LogitsWrapper(nn.Module):
    """Expose only the logits tensor expected by pytorch-grad-cam.

    The hybrid model returns ``{"logits": ...}``; Grad-CAM expects a plain
    tensor from the forward pass.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        if isinstance(output, dict):
            output = output.get("logits")
        elif isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(
                "The model did not return a tensor of logits; Grad-CAM cannot "
                "continue."
            )
        return output


def print_named_modules(model: nn.Module, reason: str) -> None:
    """Print every named module so a target layer can be chosen manually."""
    print(f"\n{reason}\n")
    print("Available modules (name: type):")
    for name, module in model.named_modules():
        print(f"  {name or '<root>'}: {type(module).__name__}")
    print()


def find_target_layer(model: nn.Module) -> Tuple[List[nn.Module], str]:
    # Find the last Conv2d layer in the ResNet50 backbone to use as the Grad-CAM target.
    # This is typically cnn_branch.backbone.7.2.conv3 (the final conv in layer4).
    # We use this layer because its output is a 7x7 spatial feature map, 
    # which Grad-CAM needs to generate a meaningful heatmap.
    cnn_branch = getattr(model, "cnn_branch", None)
    backbone = getattr(cnn_branch, "backbone", None)

    if backbone is None:
        print_named_modules(
            model,
            "Could not find model.cnn_branch.backbone. "
            "Please choose a target layer from the list below.",
        )
        raise RuntimeError("Unable to locate the ResNet50 backbone.")

    last_conv: nn.Conv2d | None = None
    last_conv_name = ""
    # We need the final convolutional layer because it still contains spatial 
    # information (7x7 grid) which Grad-CAM needs to generate a heatmap.
    for name, module in backbone.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
            last_conv_name = name

    if last_conv is None:
        print_named_modules(
            model,
            "No Conv2d layer was found inside model.cnn_branch.backbone. "
            "Please choose a target layer from the list below.",
        )
        raise RuntimeError("Unable to locate a convolutional target layer.")

    full_name = f"cnn_branch.backbone.{last_conv_name}".rstrip(".")
    return [last_conv], full_name


def extract_logits(output) -> torch.Tensor:
    # Extract a logits tensor from the model's dict/tuple/tensor output.
    if isinstance(output, dict):
        logits = output.get("logits")
    elif isinstance(output, (tuple, list)):
        logits = output[0]
    else:
        logits = output

    if not isinstance(logits, torch.Tensor):
        raise RuntimeError("Model did not return a logits tensor.")
    return logits


def predict_index(
    model: nn.Module, input_tensor: torch.Tensor, num_classes: int
) -> int:
    """Return the top-1 predicted class index."""
    with torch.inference_mode():
        logits = extract_logits(model(input_tensor))

    if logits.ndim != 2:
        raise RuntimeError(
            f"Unexpected model output shape {tuple(logits.shape)}; "
            "expected (batch, classes)."
        )
    if logits.shape[0] != 1:
        raise RuntimeError(
            f"Unexpected batch size {logits.shape[0]}; expected 1."
        )
    if logits.shape[1] != num_classes:
        raise RuntimeError(
            f"Model returned {logits.shape[1]} classes but the class mapping "
            f"contains {num_classes}."
        )
    return int(logits.argmax(dim=1).item())


def generate_cam(
    cam: GradCAM,
    model: nn.Module,
    input_tensor: torch.Tensor,
    predicted_index: int,
) -> np.ndarray:
    """Run Grad-CAM for one image and return a 224x224 heatmap."""
    # The model was loaded in eval mode with gradients disabled. 
    # Grad-CAM needs gradients to calculate the importance of each pixel, 
    # so we temporarily enable them.
    cam_input = input_tensor.clone().detach().requires_grad_(True)
    targets = [ClassifierOutputTarget(predicted_index)]

    model.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    try:
        with torch.enable_grad():
            grayscale_cam = cam(input_tensor=cam_input, targets=targets)[0]
    finally:
        model.requires_grad_(False)
        model.zero_grad(set_to_none=True)

    return grayscale_cam

# Dataset helpers
def load_validation_split():
    # Load the Food-101 validation split (the official test set).
    last_error: Exception | None = None
    for split in ("validation", "test"):
        try:
            LOGGER.info("Loading %s split '%s' ...", DATASET_ID, split)
            dataset = load_dataset(DATASET_ID, split=split)
            if split != "validation":
                LOGGER.warning(
                    "The '%s' split was unavailable; using '%s' instead.",
                    "validation",
                    split,
                )
            return dataset, split
        except Exception as exc:  # noqa: BLE001 - try the next candidate
            last_error = exc
            LOGGER.warning("Could not load split '%s': %s", split, exc)

    raise RuntimeError(
        f"Could not load a validation/test split from {DATASET_ID}."
    ) from last_error


def get_dataset_label_names(dataset) -> List[str] | None:
    # Return the dataset's ordered class names, if available.
    try:
        names = list(dataset.features["label"].names)
        return names or None
    except Exception:  # noqa: BLE001 - not a ClassLabel feature
        return None


def build_label_mapping(
    dataset_names: Sequence[str] | None,
    idx_to_label: Dict[int, str],
) -> Dict[int, int]:
    """Map dataset label integers to model output indices by class name."""
    if not dataset_names:
        LOGGER.warning(
            "Dataset label names are unavailable; assuming the dataset index "
            "order matches the model class mapping."
        )
        return {index: index for index in range(len(idx_to_label))}

    name_to_model_index = {name: index for index, name in idx_to_label.items()}
    mapping: Dict[int, int] = {}
    missing: List[str] = []

    for dataset_index, name in enumerate(dataset_names):
        model_index = name_to_model_index.get(name)
        if model_index is None:
            missing.append(name)
        else:
            mapping[dataset_index] = model_index

    if missing:
        LOGGER.warning(
            "These dataset classes are not in the model mapping and will be "
            "skipped: %s",
            missing,
        )
    # The dataset and the model sometimes have slightly different index orders 
    # (e.g., cheesecake and cheese_plate). Mapping by class name prevents errors.
    differences = [
        (index, dataset_names[index], idx_to_label.get(index))
        for index in range(min(len(dataset_names), len(idx_to_label)))
        if dataset_names[index] != idx_to_label.get(index)
    ]
    if differences:
        LOGGER.warning(
            "Dataset label order differs from the model mapping at %d "
            "index/indices (e.g. %s). Name-based mapping is being used.",
            len(differences),
            differences[:5],
        )

    return mapping


# Visualisation
def save_overlay(
    image: Image.Image,
    grayscale_cam: np.ndarray,
    true_label: str,
    predicted_label: str,
    output_path: Path,
) -> None:
    """Overlay a Grad-CAM heatmap on the image and save it with a title."""
    rgb_image = (
        np.asarray(
            image.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32
        )
        / 255.0
    )
    overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(overlay)
    ax.axis("off")
    ax.set_title(
        f"True: {true_label}\nPredicted: {predicted_label}",
        fontsize=11,
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# Main
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM heatmaps for the Food-101 classifier."
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use, e.g. cpu, cuda, mps (default: auto).",
    )
    parser.add_argument(
        "--max-scan",
        type=int,
        default=1000,
        help="Maximum number of dataset images to scan (default: 1000).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: eval_outputs next to this script).",
    )
    parser.add_argument(
        "--list-modules",
        action="store_true",
        help="Print the model's named modules and exit.",
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

    # 1. Model
    try:
        model, transform, idx_to_label = load_model(device=args.device)
    except Exception as exc:  # noqa: BLE001 - friendly console error
        LOGGER.exception("Failed to load the model")
        print(f"\nERROR: Could not load the model: {exc}", file=sys.stderr)
        return 1

    device = getattr(model, "device", torch.device("cpu"))
    class_names = [
        idx_to_label.get(index, f"class_{index}")
        for index in range(len(idx_to_label))
    ]

    if args.list_modules:
        for name, module in model.named_modules():
            print(f"{name or '<root>'}: {type(module).__name__}")
        return 0

    # 2. Grad-CAM target layer
    try:
        target_layers, target_name = find_target_layer(model)
    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\nUsing Grad-CAM target layer: {target_name}\n")

    wrapped_model = LogitsWrapper(model)
    cam = GradCAM(model=wrapped_model, target_layers=target_layers)

    # 3. Dataset
    try:
        dataset, split_used = load_validation_split()
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to load the dataset")
        print(f"\nERROR: Could not load the dataset: {exc}", file=sys.stderr)
        return 1

    dataset_names = get_dataset_label_names(dataset)
    dataset_to_model = build_label_mapping(dataset_names, idx_to_label)

    # 4. Scan for 1 correct + 2 incorrect predictions
    found_correct = 0
    found_wrong = 0
    scanned = 0
    max_scan = min(max(1, args.max_scan), len(dataset))

    print(
        f"Scanning up to {max_scan} images from '{split_used}' "
        f"for 1 correct and 2 incorrect predictions...\n"
    )
    # We want to find 1 correct prediction to show the model working, 
    # and 2 incorrect predictions to analyze its mistakes.
    for dataset_index in range(max_scan):
        if found_correct >= 1 and found_wrong >= 2:
            break

        scanned += 1
        try:
            item = dataset[dataset_index]
            image = item["image"]
            if not isinstance(image, Image.Image):
                image = Image.fromarray(np.asarray(image))
            image = ImageOps.exif_transpose(image).convert("RGB")

            dataset_label = int(item["label"])
            true_index = dataset_to_model.get(dataset_label, -1)
            if true_index < 0:
                continue

            input_tensor = transform(image).unsqueeze(0).to(device)
            predicted_index = predict_index(
                model, input_tensor, len(idx_to_label)
            )
            is_correct = predicted_index == true_index

            if is_correct and found_correct >= 1:
                continue
            if not is_correct and found_wrong >= 2:
                continue
            # Generate the heatmap for this specific image
            grayscale_cam = generate_cam(
                cam=cam,
                model=model,
                input_tensor=input_tensor,
                predicted_index=predicted_index,
            )

            true_name = class_names[true_index]
            predicted_name = class_names[predicted_index]

            if is_correct:
                output_path = output_dir / "gradcam_correct.png"
            else:
                output_path = output_dir / f"gradcam_wrong_{found_wrong + 1}.png"

            save_overlay(
                image=image,
                grayscale_cam=grayscale_cam,
                true_label=true_name,
                predicted_label=predicted_name,
                output_path=output_path,
            )

            if is_correct:
                found_correct += 1
            else:
                found_wrong += 1

            print(
                f"[{scanned}] Saved {output_path.name} | "
                f"true={true_name} | predicted={predicted_name} | "
                f"correct={is_correct}"
            )

        except Exception as exc:  # noqa: BLE001 - skip bad samples
            LOGGER.warning("Skipping dataset index %s: %s", dataset_index, exc)
            # Skip corrupt or unreadable images so the script doesn't crash
            continue

        if scanned % 25 == 0:
            print(
                f"Scanned {scanned}/{max_scan} images | "
                f"correct found: {found_correct}/1 | "
                f"wrong found: {found_wrong}/2",
                flush=True,
            )

    # 5. Summary
    print("\n" + "=" * 60)
    print(f"Scanned images : {scanned}")
    print(f"Correct found  : {found_correct}/1")
    print(f"Wrong found    : {found_wrong}/2")
    print(f"Output folder  : {output_dir}")
    print("=" * 60 + "\n")

    if found_correct < 1 or found_wrong < 2:
        print(
            "ERROR: Could not find the required examples within the scan "
            "limit. Try increasing --max-scan.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
