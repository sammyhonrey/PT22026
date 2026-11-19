
"""SceneSense image preprocessing and segmentation pipeline."""

from __future__ import annotations

import time

import numpy as np

from PIL import Image, ImageOps

from .analytics import (
    summarize_mask,
    summarize_panoptic_segments,
)

from .config import (
    DEFAULT_OVERLAY_OPACITY,
    INSTANCE_MODEL_ID,
    INSTANCE_SCORE_THRESHOLD,
    INSTANCE_THING_CLASSES,
    MAX_IMAGE_SIDE,
    MAX_INPUT_PIXELS,
    MIN_IMAGE_SIDE,
)

from .model import (
    load_model,
    predict_instance_masks,
    predict_label_map,
    predict_panoptic_segments,
)

from .visualize import (
    build_palette,
    create_color_mask,
    create_overlay,
    create_panoptic_visualization,
)


# Available segmentation modes.
SEMANTIC_MODE = "Semantic Segmentation"
INSTANCE_MODE = "Instance Segmentation"
PANOPTIC_MODE = "Panoptic Segmentation"


# ==========================================================
# IMAGE PREPARATION
# ==========================================================

def prepare_image(image: Image.Image) -> Image.Image:
    """Validate, correct orientation and resize the uploaded image."""

    if image is None:
        raise ValueError("Please upload an image.")

    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL image.")

    width, height = image.size

    # Check the image size before sending it to the model.
    if width < MIN_IMAGE_SIDE or height < MIN_IMAGE_SIDE:
        raise ValueError(
            f"Image must be at least {MIN_IMAGE_SIDE} pixels per side."
        )

    if width * height > MAX_INPUT_PIXELS:
        raise ValueError(
            "This image is too large. Please use a smaller image."
        )

    # Correct orientation and convert the image to RGB.
    prepared = ImageOps.exif_transpose(image).convert("RGB")

    # Resize large images without changing their proportions.
    prepared.thumbnail(
        (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE),
        Image.Resampling.LANCZOS
    )

    return prepared


# ==========================================================
# INSTANCE SEGMENTATION RESULTS
# ==========================================================

def create_instance_results(image, instances, class_names, opacity):
    """Create coloured masks, overlays and statistics for each object."""

    width, height = image.size

    original = np.asarray(
        image.convert("RGB"),
        dtype=np.uint8
    )

    overlay_pixels = original.copy()

    # Use a dark background for areas without detected objects.
    mask_pixels = np.full(
        (height, width, 3),
        30,
        dtype=np.uint8
    )

    palette = build_palette(
        num_classes=max(1, len(instances))
    )

    detected_objects = []

    for instance in instances:

        region = np.asarray(
            instance["mask"],
            dtype=bool
        )

        if region.shape != (height, width):
            raise ValueError(
                "Instance mask dimensions do not match the image."
            )

        # Count how many pixels belong to this object.
        pixels = int(np.count_nonzero(region))

        if pixels == 0:
            continue

        class_id = int(instance["label_id"])

        name = str(
            class_names.get(
                class_id,
                f"class_{class_id}"
            )
        )

        detected_objects.append({
            "mask": region,
            "name": name,
            "pixels": pixels,
            "score": float(instance["score"]) * 100.0
        })

    # Display larger objects first in the results table.
    detected_objects.sort(
        key=lambda item: item["pixels"],
        reverse=True
    )

    rows = []
    category_counts = {}

    for index, instance in enumerate(detected_objects):

        region = instance["mask"]
        name = instance["name"]
        pixels = instance["pixels"]
        score = instance["score"]

        # Assign a separate colour to each detected object.
        color = palette[index].astype(np.float32)

        mask_pixels[region] = color.astype(np.uint8)

        # Blend the object colour with the original photograph.
        overlay_pixels[region] = np.rint(
            (1.0 - opacity) * overlay_pixels[region]
            + opacity * color
        ).clip(0, 255).astype(np.uint8)

        # Number objects of the same category, such as Person 1 and Person 2.
        category_counts[name] = (
            category_counts.get(name, 0) + 1
        )

        number = category_counts[name]

        rows.append([
            f"{name.title()} {number} ({score:.1f}%)",
            pixels,
            round(
                pixels / (width * height) * 100.0,
                2
            )
        ])

    # Convert the generated pixel arrays into displayable images.
    color_mask = Image.fromarray(
        mask_pixels,
        mode="RGB"
    )

    overlay = Image.fromarray(
        overlay_pixels,
        mode="RGB"
    )

    return color_mask, overlay, rows


# ==========================================================
# MAIN SEGMENTATION PIPELINE
# ==========================================================

def run_segmentation(
    image: Image.Image,
    opacity: float = DEFAULT_OVERLAY_OPACITY,
    mode: str = SEMANTIC_MODE,
):
    """Run the selected semantic, instance or panoptic mode."""

    if not 0.0 <= float(opacity) <= 1.0:
        raise ValueError(
            "Overlay strength must be between 0 and 1."
        )

    valid_modes = (
        SEMANTIC_MODE,
        INSTANCE_MODE,
        PANOPTIC_MODE
    )

    if mode not in valid_modes:
        raise ValueError("Unknown segmentation mode.")

    # Prepare the image once for the selected segmentation mode.
    prepared = prepare_image(image)

    # --------------------------------------------------------
    # SEMANTIC SEGMENTATION
    # --------------------------------------------------------

    if mode == SEMANTIC_MODE:

        # Load the pretrained semantic segmentation checkpoint.
        processor, model = load_model()

        start = time.perf_counter()

        label_map = predict_label_map(
            prepared,
            processor,
            model
        )

        # Measure how long the model prediction takes.
        elapsed = time.perf_counter() - start

        # Get the category names stored in the model configuration.
        class_names = {
            int(class_id): str(name).strip()
            for class_id, name in model.config.id2label.items()
        }

        palette = build_palette(
            class_names,
            int(model.config.num_labels)
        )

        # Convert the predicted class IDs into a coloured mask.
        color_mask = create_color_mask(
            label_map,
            palette
        )

        overlay = create_overlay(
            prepared,
            color_mask,
            float(opacity)
        )

        # Calculate pixel counts and area percentages for each category.
        rows = summarize_mask(
            label_map,
            class_names
        )

        summary = (
            f"**Image:** {prepared.width} × {prepared.height}  \n"
            f"**Detected:** {len(rows)}  \n"
            f"**Inference:** {elapsed:.2f}s"
        )

        return color_mask, overlay, rows, summary

    # --------------------------------------------------------
    # LOAD INSTANCE / PANOPTIC MODEL
    # --------------------------------------------------------

    # Instance and Panoptic Segmentation share the Swin Tiny checkpoint.
    processor, model = load_model(INSTANCE_MODEL_ID)

    class_names = {
        int(class_id): str(name).strip()
        for class_id, name in model.config.id2label.items()
    }

    start = time.perf_counter()

    # --------------------------------------------------------
    # INSTANCE SEGMENTATION
    # --------------------------------------------------------

    if mode == INSTANCE_MODE:

        # Predict separate masks for individual objects.
        instances = predict_instance_masks(
            prepared,
            processor,
            model,
            threshold=INSTANCE_SCORE_THRESHOLD
        )

        elapsed = time.perf_counter() - start

        color_mask, overlay, rows = create_instance_results(
            prepared,
            instances,
            class_names,
            float(opacity)
        )

        summary = (
            f"**Image:** {prepared.width} × {prepared.height}  \n"
            f"**Objects:** {len(rows)}  \n"
            f"**Inference:** {elapsed:.2f}s"
        )

        return color_mask, overlay, rows, summary

    # --------------------------------------------------------
    # PANOPTIC SEGMENTATION
    # --------------------------------------------------------

    # Predict individual objects and background regions together.
    segmentation_map, segments_info = predict_panoptic_segments(
        prepared,
        processor,
        model,
        threshold=INSTANCE_SCORE_THRESHOLD
    )

    elapsed = time.perf_counter() - start

    # Give each panoptic segment its own visualization colour.
    color_mask, overlay = create_panoptic_visualization(
        prepared,
        segmentation_map,
        segments_info,
        opacity=float(opacity)
    )

    # Calculate the pixel area of each detected segment.
    rows = summarize_panoptic_segments(
        segmentation_map,
        segments_info,
        class_names,
        INSTANCE_THING_CLASSES
    )

    # Count individual objects and background categories separately.
    thing_count = sum(
        1 for segment in segments_info
        if int(segment["label_id"]) < INSTANCE_THING_CLASSES
        and np.any(segmentation_map == int(segment["id"]))
    )

    stuff_count = sum(
        1 for segment in segments_info
        if int(segment["label_id"]) >= INSTANCE_THING_CLASSES
        and np.any(segmentation_map == int(segment["id"]))
    )

    summary = (
        f"**Image:** {prepared.width} × {prepared.height}  \n"
        f"**Detected segments:** {len(rows)}  \n"
        f"**Things:** {thing_count}  \n"
        f"**Stuff categories:** {stuff_count}  \n"
        f"**Inference:** {elapsed:.2f}s"
    )

    return color_mask, overlay, rows, summary
