
"""
SceneSense - Instance Segmentation Tests

These tests check instance segmentation using fake predictions,
so the actual pretrained model is not required.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from PIL import Image

import segmentation.pipeline as pipeline
from segmentation.model import predict_instance_masks


# ============================================================
# TEST INSTANCE SEGMENTATION PIPELINE
# ============================================================

def test_instance_mode_with_fake_model(monkeypatch):
    """Check whether separate objects are detected correctly."""

    names = {
        0: "person",
        16: "dog"
    }

    # Create a fake model with category information.
    fake_model = SimpleNamespace(
        config=SimpleNamespace(id2label=names)
    )

    # Replace the real model loader with our fake model.
    monkeypatch.setattr(
        pipeline,
        "load_model",
        lambda model_id: (object(), fake_model)
    )

    def fake_prediction(image, processor, model, threshold=0.5):

        # Create the first dog's mask.
        first = np.zeros(
            (image.height, image.width),
            dtype=bool
        )

        # Create the second dog's mask.
        second = np.zeros_like(first)

        # The first dog occupies 40% of the image width.
        first[:, :40] = True

        # The second dog occupies 60% of the image width.
        second[:, 40:] = True

        return [
            {
                "mask": first,
                "label_id": 16,
                "score": 0.93
            },
            {
                "mask": second,
                "label_id": 16,
                "score": 0.89
            }
        ]

    # Replace actual model prediction with fake predictions.
    monkeypatch.setattr(
        pipeline,
        "predict_instance_masks",
        fake_prediction
    )

    # Run instance segmentation on a sample image.
    mask, overlay, rows, summary = pipeline.run_segmentation(
        Image.new("RGB", (100, 80)),
        0.35,
        pipeline.INSTANCE_MODE
    )

    # Check the output image dimensions.
    assert mask.size == (100, 80)
    assert overlay.size == (100, 80)

    # Check that both dogs are detected separately.
    assert len(rows) == 2

    # The larger dog should receive number 1.
    assert [row[0] for row in rows] == [
        "Dog 1 (89.0%)",
        "Dog 2 (93.0%)"
    ]

    # Check that objects are sorted by pixel area.
    assert [row[1] for row in rows] == [
        4800,
        3200
    ]

    # Check that the summary displays the detected object count.
    assert "**Objects:** 2" in summary


# ============================================================
# TEST IMAGE WITH NO DETECTED OBJECTS
# ============================================================

def test_instance_mode_with_no_objects(monkeypatch):
    """Check that the application handles images without detected objects."""

    fake_model = SimpleNamespace(
        config=SimpleNamespace(id2label={})
    )

    monkeypatch.setattr(
        pipeline,
        "load_model",
        lambda model_id: (object(), fake_model)
    )

    # Simulate a prediction where no objects are detected.
    monkeypatch.setattr(
        pipeline,
        "predict_instance_masks",
        lambda *args, **kwargs: []
    )

    image = Image.new(
        "RGB",
        (100, 80),
        (100, 150, 200)
    )

    mask, overlay, rows, summary = pipeline.run_segmentation(
        image,
        0.35,
        pipeline.INSTANCE_MODE
    )

    # No objects should appear in the results table.
    assert rows == []

    assert "**Objects:** 0" in summary

    # The original image should remain unchanged in the overlay.
    assert overlay.getpixel((0, 0)) == (
        100, 150, 200
    )


# ============================================================
# TEST INVALID SEGMENTATION MODE
# ============================================================

def test_invalid_mode_rejected():
    """Check that unsupported segmentation modes are rejected."""

    with pytest.raises(ValueError, match="mode"):
        pipeline.run_segmentation(
            Image.new("RGB", (100, 80)),
            mode="invalid"
        )


# ============================================================
# TEST BACKGROUND CATEGORY FILTERING
# ============================================================

def test_instance_prediction_ignores_background_categories():
    """Check that background categories are not returned as objects."""

    class FakeProcessor:

        def __call__(self, images, return_tensors):
            return {
                "pixel_values": torch.zeros((1, 3, 8, 8))
            }

        def post_process_instance_segmentation(
            self,
            outputs,
            **kwargs
        ):
            masks = torch.ones((2, 80, 100))

            return [{
                "segmentation": masks,
                "segments_info": [
                    {
                        "label_id": 16,
                        "score": 0.9
                    },
                    {
                        "label_id": 85,
                        "score": 0.9
                    }
                ]
            }]

    class FakeModel:

        def __call__(self, **inputs):
            return object()

    # Class 16 represents an object category.
    # Class 85 represents a background category.
    result = predict_instance_masks(
        Image.new("RGB", (100, 80)),
        FakeProcessor(),
        FakeModel()
    )

    # Only the object category should be returned.
    assert len(result) == 1

    assert result[0]["label_id"] == 16

    assert result[0]["mask"].shape == (80, 100)
