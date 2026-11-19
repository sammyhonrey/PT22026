
"""Load MaskFormer checkpoints and predict semantic, instance or panoptic masks."""

from __future__ import annotations

import gc
import os

import numpy as np
import torch
from PIL import Image

from transformers import (
    AutoImageProcessor,
    MaskFormerForInstanceSegmentation,
)

from .config import (
    DEVICE,
    INSTANCE_THING_CLASSES,
    MIN_INSTANCE_AREA_PERCENT,
    MODEL_ID,
    NUM_CLASSES,
)


# Store the loaded model so it can be reused.
_cached_model_id = None
_cached_objects = None


# ==========================================================
# LOAD THE PRETRAINED MODEL
# ==========================================================

def load_model(model_id: str = MODEL_ID):
    """Load a checkpoint once and reuse it until another is selected."""

    global _cached_model_id, _cached_objects

    # Reuse the model if the same checkpoint is already loaded.
    if _cached_model_id == model_id and _cached_objects is not None:
        return _cached_objects

    # Free the previous model before loading a different checkpoint.
    _cached_model_id = None
    _cached_objects = None
    gc.collect()

    # Limit the number of CPU threads used by PyTorch.
    torch.set_num_threads(
        min(4, os.cpu_count() or 1)
    )

    print("\n" + "=" * 55)
    print("LOADING SCENESENSE MASKFORMER")
    print("Model:", model_id)
    print("Device:", DEVICE)

    # Load the image processor and pretrained model from Hugging Face.
    processor = AutoImageProcessor.from_pretrained(
        model_id,
        use_fast=False
    )

    model = MaskFormerForInstanceSegmentation.from_pretrained(
        model_id
    )

    model.to(torch.device(DEVICE))
    model.eval()  # Set the model to evaluation mode.

    if model_id == MODEL_ID and int(model.config.num_labels) != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_CLASSES} classes, but the model contains "
            f"{model.config.num_labels}."
        )

    print("MaskFormer loaded successfully!")
    print("Classes:", model.config.num_labels)

    # Save the model and processor for later predictions.
    _cached_objects = (processor, model)
    _cached_model_id = model_id

    return _cached_objects


# ==========================================================
# PREPARE MODEL INPUT
# ==========================================================

def _prepare_model_inputs(image: Image.Image):
    """Validate the image and convert it to RGB."""

    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL image.")

    if image.width < 1 or image.height < 1:
        raise ValueError("The image must have positive dimensions.")

    return image.convert("RGB")  # Make sure the image has three colour channels.


# ==========================================================
# SEMANTIC SEGMENTATION
# ==========================================================

def predict_label_map(
    image: Image.Image,
    processor,
    model,
) -> np.ndarray:
    """Predict a semantic category identifier for each pixel."""

    image = _prepare_model_inputs(image)

    # Convert the image into tensors that MaskFormer can process.
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    # Run inference without calculating gradients.
    with torch.inference_mode():

        outputs = model(**inputs)

        # Convert predictions into a class map matching the image size.
        segmentation = processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[
                (image.height, image.width)
            ]
        )[0]

    # Convert the predicted class map into a NumPy array.
    labels = (
        segmentation
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    if labels.shape != (image.height, image.width):
        raise RuntimeError(
            "Mask dimensions do not match the processed image."
        )

    if labels.size == 0 or int(labels.max()) >= NUM_CLASSES:
        raise RuntimeError(
            "The model returned invalid semantic class IDs."
        )

    return labels


# ==========================================================
# INSTANCE SEGMENTATION
# ==========================================================

def predict_instance_masks(
    image: Image.Image,
    processor,
    model,
    threshold: float = 0.5,
) -> list[dict]:
    """Return separate binary masks for individual countable objects."""

    image = _prepare_model_inputs(image)

    if not 0 <= threshold <= 1:
        raise ValueError(
            "Instance score threshold must be between 0 and 1."
        )

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        outputs = model(**inputs)

        # Get a separate binary mask for each detected object.
        result = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            target_sizes=[
                (image.height, image.width)
            ],
            return_binary_maps=True,
        )[0]

    segments = result.get("segments_info", [])

    if not segments:
        return []

    masks = result["segmentation"]

    masks = (
        masks.detach().cpu().numpy()
        if hasattr(masks, "detach")
        else np.asarray(masks)
    )

    if masks.shape != (len(segments), image.height, image.width):
        raise RuntimeError(
            "Instance mask dimensions are invalid."
        )

    instances = []
    total_pixels = image.width * image.height

    for mask, segment in zip(masks, segments):

        class_id = int(segment["label_id"])

        # Ignore background categories because they are not individual objects.
        if not 0 <= class_id < INSTANCE_THING_CLASSES:
            continue

        binary_mask = mask.astype(bool)

        pixel_count = int(
            np.count_nonzero(binary_mask)
        )

        if pixel_count == 0:
            continue

        # Skip objects that cover less than the minimum image area.
        area_percent = pixel_count / total_pixels * 100.0

        if area_percent < MIN_INSTANCE_AREA_PERCENT:
            continue

        # Keep each object's mask, category and prediction score.
        instances.append({
            "mask": binary_mask,
            "label_id": class_id,
            "score": float(segment["score"]),
        })

    return instances


# ==========================================================
# PANOPTIC SEGMENTATION
# ==========================================================

def predict_panoptic_segments(
    image: Image.Image,
    processor,
    model,
    threshold: float = 0.5,
) -> tuple[np.ndarray, list[dict]]:
    """Predict individual objects and background regions together."""

    image = _prepare_model_inputs(image)

    if not 0 <= threshold <= 1:
        raise ValueError(
            "Panoptic score threshold must be between 0 and 1."
        )

    number_of_classes = int(model.config.num_labels)

    if number_of_classes <= INSTANCE_THING_CLASSES:
        raise RuntimeError(
            "The selected checkpoint does not contain both "
            "thing and stuff classes."
        )

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    # Identify the background categories that can be merged.
    stuff_label_ids = set(
        range(
            INSTANCE_THING_CLASSES,
            number_of_classes
        )
    )

    with torch.inference_mode():

        outputs = model(**inputs)

        # Merge repeated background categories while keeping objects separate.
        result = processor.post_process_panoptic_segmentation(
            outputs,
            threshold=threshold,
            target_sizes=[
                (image.height, image.width)
            ],
            label_ids_to_fuse=stuff_label_ids,
        )[0]

    segmentation = result.get("segmentation")

    if segmentation is None:

        # Create an empty map if no segmentation result is returned.
        segmentation_map = np.zeros(
            (image.height, image.width),
            dtype=np.int32
        )

    else:

        segmentation_map = (
            segmentation.detach().cpu().numpy().astype(np.int32)
            if hasattr(segmentation, "detach")
            else np.asarray(segmentation, dtype=np.int32)
        )

    if segmentation_map.shape != (
        image.height,
        image.width
    ):
        raise RuntimeError(
            "Panoptic mask dimensions do not match the processed image."
        )

    segments_info = result.get(
        "segments_info",
        []
    )

    return segmentation_map, segments_info


# ==========================================================
# MODEL INFORMATION
# ==========================================================

def get_model_information():
    """Return information about the semantic segmentation model."""

    return {
        "name": "MaskFormer ResNet-50 COCO-Stuff",
        "checkpoint": MODEL_ID,
        "classes": NUM_CLASSES,
        "device": DEVICE,
        "task": "semantic segmentation",
    }
