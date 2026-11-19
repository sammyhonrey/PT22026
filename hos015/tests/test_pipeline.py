
"""
SceneSense - Pipeline Tests

Tests image preprocessing and application outputs
without loading the real pretrained model.
"""

# SimpleNamespace helps create a fake model configuration for testing.
from types import SimpleNamespace

import numpy as np
import pytest

from PIL import Image

# Import the segmentation pipeline to test its functions.
import segmentation.pipeline as pipeline


def test_rgb_conversion():
    """Check whether an image is converted to Red, Green and Blue."""

    result = pipeline.prepare_image(
        Image.new("RGBA", (100, 80))
    )

    assert result.mode == "RGB"
    assert result.size == (100, 80)


def test_resize_keeps_aspect_ratio():
    """Check whether resizing keeps the original image proportions."""

    result = pipeline.prepare_image(
        Image.new("RGB", (2000, 1000))
    )

    assert result.size == (640, 320)


def test_small_image_rejected():
    """Check whether images smaller than the minimum size are rejected."""

    with pytest.raises(ValueError):
        pipeline.prepare_image(
            Image.new("RGB", (20, 20))
        )


def test_missing_image_rejected():
    """Check whether the pipeline rejects a missing image."""

    with pytest.raises(ValueError):
        pipeline.prepare_image(None)


def test_invalid_input_rejected():
    """Check whether the pipeline rejects an invalid input type."""

    with pytest.raises(TypeError):
        pipeline.prepare_image(
            "not an image"
        )


def test_invalid_opacity_rejected():
    """Check whether overlay strength outside the allowed range is rejected."""

    with pytest.raises(ValueError):
        pipeline.run_segmentation(
            Image.new("RGB", (100, 80)),
            opacity=1.5
        )


def test_full_pipeline_with_fake_model(monkeypatch):
    """
    Test the complete segmentation pipeline using a fake model.

    This allows us to check the pipeline without downloading
    or running the actual pretrained MaskFormer model.
    """

    # Create 171 category names to simulate the model vocabulary.
    names = {
        i: f"category_{i}"
        for i in range(171)
    }

    names[0] = "person"
    names[16] = "dog"

    # SimpleNamespace creates a fake model with the same
    # configuration properties used by the real pipeline.
    fake_model = SimpleNamespace(
        config=SimpleNamespace(
            id2label=names,
            num_labels=171
        )
    )

    # Temporarily replace the real model-loading function
    # so the test does not need to load MaskFormer.
    monkeypatch.setattr(
        pipeline,
        "load_model",
        lambda: (object(), fake_model)
    )

    # Create a fake segmentation mask:
    # the left half represents person and the right half represents dog.
    def fake_prediction(image, processor, model):

        # Start with every pixel assigned to class 0 (person).
        labels = np.zeros(
            (image.height, image.width),
            dtype=np.uint8
        )

        # Assign class 16 (dog) to the right half of the image.
        labels[:, image.width // 2:] = 16

        return labels

    # Replace the real prediction function with our fake prediction.
    monkeypatch.setattr(
        pipeline,
        "predict_label_map",
        fake_prediction
    )

    # Run the complete pipeline using a sample image.
    output = pipeline.run_segmentation(
        Image.new("RGB", (100, 80)),
        0.35
    )

    # The pipeline should return four results:
    # coloured mask, overlay, category statistics and summary.
    assert len(output) == 4

    color_mask, overlay, rows, summary = output

    # Check whether the mask and overlay have the correct dimensions.
    assert color_mask.size == (100, 80)

    assert overlay.size == (100, 80)

    # Check whether both expected categories were detected.
    assert {
        row[0]
        for row in rows
    } == {
        "person",
        "dog"
    }

    # The total number of classified pixels should be 100 × 80 = 8000.
    assert sum(
        row[1]
        for row in rows
    ) == 8000

    # Check whether the summary contains the correct image dimensions,
    # detected category count and inference information.
    assert "**Image:** 100 × 80" in summary
    assert "**Detected:** 2" in summary
    assert "**Inference:**" in summary
