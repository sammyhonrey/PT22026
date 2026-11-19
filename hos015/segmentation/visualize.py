
"""
SceneSense - Segmentation Visualization

Creates semantic masks, instance masks and panoptic overlays.
"""

from __future__ import annotations

import colorsys

from collections.abc import Mapping, Sequence

import numpy as np
from PIL import Image


# Fixed colours for common semantic categories.
SPECIAL_COLORS = {
    "person": (230, 105, 99),
    "animal": (234, 166, 83),
    "tree": (69, 151, 114),
    "grass": (156, 188, 104),
    "plant": (99, 173, 137),
    "sky": (102, 162, 215),
    "water": (75, 169, 181),
    "sea": (64, 125, 176),
    "road": (126, 137, 151),
    "sidewalk": (186, 174, 151),
    "wall": (173, 150, 183),
    "floor": (193, 178, 151),
    "ceiling": (216, 190, 153),
    "building": (139, 151, 175),
    "windowpane": (130, 189, 199),
    "door": (192, 139, 107),
    "chair": (183, 121, 104),
    "sofa": (135, 158, 191),
    "table": (186, 153, 111),
    "cabinet": (152, 126, 105),
    "earth": (167, 143, 116),
    "mountain": (122, 145, 124),
    "rock": (133, 134, 147),
    "fence": (180, 141, 175),
}


# Group different animal classes under the same colour.
ANIMALS = {
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "deer", "rabbit", "monkey",
    "fish", "animal",
}


# ==========================================================
# CREATE COLOUR PALETTE
# ==========================================================

def build_palette(
    class_names: Mapping[int | str, str] | None = None,
    num_classes: int = 150,
) -> np.ndarray:
    """Create consistent colours for semantic classes or segment indices."""

    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    names = class_names or {}

    palette = np.empty(
        (num_classes, 3),
        dtype=np.uint8
    )

    for class_id in range(num_classes):

        # Generate different colours for different class IDs.
        hue = (class_id * 0.61803398875) % 1.0

        rgb = colorsys.hsv_to_rgb(
            hue,
            0.43,
            0.82
        )

        color = tuple(
            int(round(value * 255))
            for value in rgb
        )

        label = str(
            names.get(
                class_id,
                names.get(str(class_id), "")
            )
        ).strip().lower()

        # Use predefined colours for common categories.
        if label in ANIMALS:
            color = SPECIAL_COLORS["animal"]

        elif label in SPECIAL_COLORS:
            color = SPECIAL_COLORS[label]

        palette[class_id] = color

    return palette


# ==========================================================
# CREATE SEMANTIC SEGMENTATION MASK
# ==========================================================

def create_color_mask(
    label_map: np.ndarray,
    palette: np.ndarray,
) -> Image.Image:
    """Convert integer semantic class identifiers into an RGB mask."""

    labels = np.asarray(label_map)
    colors = np.asarray(palette)

    if labels.ndim != 2 or labels.size == 0:
        raise ValueError(
            "Label map must be a non-empty 2D array."
        )

    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(
            "Label map must contain integer IDs."
        )

    if colors.ndim != 2 or colors.shape[1] != 3 or len(colors) == 0:
        raise ValueError(
            "Palette must have shape (N, 3)."
        )

    if np.any(labels < 0) or np.any(labels >= len(colors)):
        raise ValueError(
            "Label map contains an invalid class ID."
        )

    # Replace every class ID with its corresponding RGB colour.
    rgb = colors[
        labels.astype(np.int64)
    ].astype(np.uint8)

    return Image.fromarray(rgb, mode="RGB")


# ==========================================================
# CREATE SEMANTIC OVERLAY
# ==========================================================

def create_overlay(
    image: Image.Image,
    color_mask: Image.Image,
    opacity: float = 0.35,
) -> Image.Image:
    """Blend a coloured segmentation mask with the original image."""

    if not 0.0 <= float(opacity) <= 1.0:
        raise ValueError(
            "Opacity must be between 0 and 1."
        )

    original = image.convert("RGB")
    mask = color_mask.convert("RGB")

    if original.size != mask.size:
        raise ValueError(
            "Image and mask dimensions must match."
        )

    # Combine the original image and mask using the selected opacity.
    return Image.blend(
        original,
        mask,
        float(opacity)
    )


# ==========================================================
# CREATE PANOPTIC VISUALIZATION
# ==========================================================

def create_panoptic_visualization(
    image: Image.Image,
    segmentation_map: np.ndarray,
    segments_info: Sequence[Mapping],
    opacity: float = 0.35,
) -> tuple[Image.Image, Image.Image]:
    """Create a unique colour for each panoptic segment."""

    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL image.")

    if not 0.0 <= float(opacity) <= 1.0:
        raise ValueError(
            "Opacity must be between 0 and 1."
        )

    labels = np.asarray(segmentation_map)

    width, height = image.size

    if labels.ndim != 2 or labels.size == 0:
        raise ValueError(
            "Panoptic map must be a non-empty 2D array."
        )

    if labels.shape != (height, width):
        raise ValueError(
            "Panoptic map dimensions must match the image."
        )

    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(
            "Panoptic map must contain integer segment IDs."
        )

    detected_segments = []

    # Find the pixel area of each predicted segment.
    for segment in segments_info:

        segment_id = int(segment["id"])

        region = labels == segment_id

        pixel_count = int(
            np.count_nonzero(region)
        )

        if pixel_count:
            detected_segments.append({
                "id": segment_id,
                "mask": region,
                "pixels": pixel_count,
            })

    # Larger segments receive colours first.
    detected_segments.sort(
        key=lambda item: item["pixels"],
        reverse=True
    )

    palette = build_palette(
        num_classes=max(1, len(detected_segments))
    )

    original = np.asarray(
        image.convert("RGB"),
        dtype=np.uint8
    )

    # Unassigned pixels remain black in the segmentation mask.
    mask_pixels = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    overlay_pixels = original.copy()

    for index, segment in enumerate(detected_segments):

        region = segment["mask"]

        # Give each segment a different colour.
        color = palette[index].astype(np.float32)

        mask_pixels[region] = color.astype(np.uint8)

        # Blend colours only inside the detected segment.
        overlay_pixels[region] = np.rint(
            (1.0 - float(opacity)) * original[region]
            + float(opacity) * color
        ).clip(0, 255).astype(np.uint8)

    # Convert the final NumPy arrays into images for Gradio.
    color_mask = Image.fromarray(
        mask_pixels,
        mode="RGB"
    )

    overlay = Image.fromarray(
        overlay_pixels,
        mode="RGB"
    )

    return color_mask, overlay
