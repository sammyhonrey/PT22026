
"""
Tests for SceneSense segmentation colours and overlay.
"""

# NumPy is used to create sample segmentation masks.
import numpy as np

# Pytest is used to check expected results and errors.
import pytest

from PIL import Image

# Import the visualization functions we want to test.
from segmentation.visualize import (
    SPECIAL_COLORS,
    build_palette,
    create_color_mask,
    create_overlay,
)


def test_palette_is_stable():

    labels = {
        0: "wall",
        12: "person",
        4: "tree"
    }

    # Generate the colour palette twice using the same categories.
    first = build_palette(
        labels,
        150
    )

    second = build_palette(
        labels,
        150
    )

    # Check that the same classes always receive the same colours.
    assert np.array_equal(first, second)

    assert tuple(first[12]) == SPECIAL_COLORS["person"]

    assert tuple(first[4]) == SPECIAL_COLORS["tree"]


def test_color_mask_shape():

    # Create a small two-dimensional segmentation map.
    label_map = np.array(
        [[0, 1], [1, 0]],
        dtype=np.uint8
    )

    palette = build_palette(
        {0: "wall", 1: "person"},
        2
    )

    # Convert the numerical class identifiers into a coloured image.
    mask = create_color_mask(
        label_map,
        palette
    )

    # Check the image mode, dimensions and assigned pixel colour.
    assert mask.mode == "RGB"

    assert mask.size == (2, 2)

    assert mask.getpixel((1, 0)) == tuple(
        palette[1]
    )


def test_overlay_opacity_zero():

    # Create a sample original image and segmentation mask.
    image = Image.new(
        "RGB",
        (5, 5),
        (10, 20, 30)
    )

    mask = Image.new(
        "RGB",
        (5, 5),
        (200, 100, 50)
    )

    result = create_overlay(
        image,
        mask,
        0
    )

    # At zero opacity, only the original image should be visible.
    assert result.getpixel((0, 0)) == (
        10, 20, 30
    )


def test_overlay_opacity_one():

    image = Image.new(
        "RGB",
        (5, 5),
        (10, 20, 30)
    )

    mask = Image.new(
        "RGB",
        (5, 5),
        (200, 100, 50)
    )

    result = create_overlay(
        image,
        mask,
        1
    )

    # At full opacity, the result should show only the mask colours.
    assert result.getpixel((0, 0)) == (
        200, 100, 50
    )


def test_invalid_class_rejected():

    # Class identifier 12 is invalid when the palette
    # contains colours for only two classes.
    with pytest.raises(ValueError):

        create_color_mask(
            np.array([[12]], dtype=np.uint8),
            build_palette(num_classes=2)
        )


def test_mismatched_dimensions_rejected():

    # The original image and segmentation mask must
    # have matching dimensions before blending.
    with pytest.raises(ValueError):

        create_overlay(
            Image.new("RGB", (5, 5)),
            Image.new("RGB", (4, 4))
        )


def test_invalid_opacity_rejected():

    # Overlay opacity must be between zero and one.
    with pytest.raises(ValueError):

        create_overlay(
            Image.new("RGB", (5, 5)),
            Image.new("RGB", (5, 5)),
            1.2
        )
