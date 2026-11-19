
"""Calculate pixel counts and area percentages for segmentation results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


# ==========================================================
# SEMANTIC SEGMENTATION STATISTICS
# ==========================================================

def summarize_mask(
    label_map: np.ndarray,
    class_names: Mapping,
) -> list[list]:
    """Calculate pixel counts and area percentages for semantic classes."""

    labels = np.asarray(label_map)

    # Check that the input contains valid pixel class identifiers.
    if labels.ndim != 2:
        raise ValueError("Label map must be a 2-D array.")

    if labels.size == 0:
        raise ValueError("Label map cannot be empty.")

    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Label map must contain integer class IDs.")

    if not isinstance(class_names, Mapping):
        raise TypeError("class_names must be a mapping of class IDs to names.")

    # Find the different classes and count their pixels.
    unique_ids, counts = np.unique(
        labels,
        return_counts=True
    )

    total_pixels = int(labels.size)
    results = []

    for class_id, pixel_count in zip(unique_ids, counts):

        class_id = int(class_id)
        pixel_count = int(pixel_count)

        name = class_names.get(
            class_id,
            class_names.get(str(class_id), f"class_{class_id}")
        )

        # Calculate how much of the image belongs to this class.
        percentage = pixel_count / total_pixels * 100.0

        results.append([
            str(name),
            pixel_count,
            round(percentage, 2)
        ])

    # Show the classes covering the largest area first.
    results.sort(
        key=lambda row: row[1],
        reverse=True
    )

    return results


# ==========================================================
# PANOPTIC SEGMENTATION STATISTICS
# ==========================================================

def summarize_panoptic_segments(
    segmentation_map: np.ndarray,
    segments_info: Sequence[Mapping],
    class_names: Mapping,
    thing_class_count: int,
) -> list[list]:
    """Create table rows for individual objects and stuff categories."""

    labels = np.asarray(segmentation_map)

    if labels.ndim != 2 or labels.size == 0:
        raise ValueError(
            "Panoptic segmentation map must be a non-empty 2-D array."
        )

    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(
            "Panoptic segmentation map must contain integer IDs."
        )

    if not isinstance(class_names, Mapping):
        raise TypeError(
            "class_names must be a mapping of class IDs to names."
        )

    if thing_class_count <= 0:
        raise ValueError("thing_class_count must be positive.")

    total_pixels = int(labels.size)
    detected = []

    # Calculate the pixel area of each detected segment.
    for segment in segments_info:

        segment_id = int(segment["id"])
        class_id = int(segment["label_id"])

        # Find pixels belonging to this particular segment.
        region = labels == segment_id

        pixel_count = int(
            np.count_nonzero(region)
        )

        if pixel_count == 0:
            continue

        class_name = str(
            class_names.get(
                class_id,
                class_names.get(str(class_id), f"class_{class_id}")
            )
        )

        score = float(
            segment.get("score", 0.0)
        ) * 100.0

        # Keep each object separate and identify whether it is a thing or stuff.
        detected.append({
            "class_id": class_id,
            "name": class_name,
            "pixels": pixel_count,
            "score": score,
            "is_thing": class_id < thing_class_count,
        })

    # Sort the detected segments by their pixel area.
    detected.sort(
        key=lambda item: item["pixels"],
        reverse=True
    )

    category_counts = {}
    rows = []

    for segment in detected:

        name = segment["name"]
        pixels = segment["pixels"]
        score = segment["score"]

        if segment["is_thing"]:

            # Number objects of the same category, such as Dog 1 and Dog 2.
            category_counts[name] = (
                category_counts.get(name, 0) + 1
            )

            display_name = (
                f"{name.title()} "
                f"{category_counts[name]} ({score:.1f}%)"
            )

        else:

            # Stuff categories, such as grass and sky, do not need numbers.
            display_name = (
                f"{name.title()} (Stuff, {score:.1f}%)"
            )

        # Prepare the final row for the SceneSense results table.
        rows.append([
            display_name,
            pixels,
            round(
                pixels / total_pixels * 100.0,
                2
            )
        ])

    return rows
