
"""
Tests for SceneSense semantic class statistics.
"""

# NumPy is used to create sample segmentation masks for testing.
import numpy as np

# Pytest is used to check errors and expected test results.
import pytest

from segmentation.analytics import summarize_mask


def test_class_pixel_counts():
    """Check whether pixels are counted correctly."""

    # Create a small mask with three different class identifiers.
    mask = np.array(
        [
            [12, 12, 0, 0],
            [12, 12, 0, 2],
        ],
        dtype=np.uint8
    )

    # Assign category names to the class identifiers.
    names = {
        0: "wall",
        2: "sky",
        12: "person",
    }

    result = summarize_mask(mask, names)

    # Check the pixel counts and percentages for each category.
    assert result[0] == ["person", 4, 50.0]
    assert result[1] == ["wall", 3, 37.5]
    assert result[2] == ["sky", 1, 12.5]


def test_single_class_coverage():
    """A single predicted class should cover the entire image."""

    # Create a 10 × 10 mask where every pixel belongs to class 2.
    mask = np.full(
        (10, 10),
        2,
        dtype=np.uint8
    )

    result = summarize_mask(
        mask,
        {2: "sky"}
    )

    # All 100 pixels should belong to the sky category.
    assert result == [
        ["sky", 100, 100.0]
    ]


def test_missing_category_name():
    """Unknown class IDs should have fallback names."""

    mask = np.full(
        (4, 4),
        25,
        dtype=np.uint8
    )

    # Test whether an unknown category receives a default name.
    result = summarize_mask(mask, {})

    assert result[0][0] == "class_25"
    assert result[0][1] == 16
    assert result[0][2] == 100.0


def test_invalid_mask_dimensions():
    """The analytics module must reject non-2D masks."""

    # A three-dimensional array is not a valid semantic label map.
    # pytest.raises checks whether the expected error occurs.
    with pytest.raises(ValueError):
        summarize_mask(
            np.zeros((10, 10, 3), dtype=np.uint8),
            {}
        )


def test_empty_mask():
    """Empty predictions are invalid."""

    # Check that an empty segmentation mask raises an error.
    with pytest.raises(ValueError):
        summarize_mask(
            np.empty((0, 0), dtype=np.uint8),
            {}
        )


def test_area_percentages():
    """Reported area percentages should total approximately 100%."""

    # Create a mask containing equal numbers of three categories.
    mask = np.array(
        [
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
        ],
        dtype=np.uint8
    )

    result = summarize_mask(
        mask,
        {0: "wall", 1: "building", 2: "sky"}
    )

    # Add the percentages of all detected categories.
    total = sum(row[2] for row in result)

    # Allow a small difference caused by rounding percentages.
    assert total == pytest.approx(
        100.0,
        abs=0.02
    )
