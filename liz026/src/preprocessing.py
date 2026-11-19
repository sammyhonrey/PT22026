"""
Image validation and preprocessing.

The preprocessing must match what the model saw during training:
RGB, 224x224, pixel values scaled to [0, 1].
"""
import numpy as np
from PIL import Image, ImageOps

from src import config


class InvalidImageError(ValueError):
    """Raised when the uploaded file cannot be used as a chest X-ray."""


def to_rgb(image: Image.Image) -> Image.Image:
    """
    EXIF rotation applied, converted to 8-bit RGB.

    Medical X-rays are often saved as 16-bit PNGs (PIL modes "I;16", "I").
    PIL's own convert("RGB") clips those to almost pure white, so they are
    rescaled to 0-255 from their actual minimum and maximum first.
    """
    image = ImageOps.exif_transpose(image)
    if image.mode.startswith("I") or image.mode == "F":
        arr = np.asarray(image, dtype=np.float64)
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)
        image = Image.fromarray(arr.astype(np.uint8), mode="L")
    return image.convert("RGB")


def validate_image(image: Image.Image) -> list[str]:
    """
    Check that the input is usable. Raises InvalidImageError for hard
    failures and returns a list of soft warnings for the user.
    """
    if image is None:
        raise InvalidImageError("Please upload a chest X-ray image first.")

    w, h = image.size
    if min(w, h) < config.MIN_IMAGE_SIDE:
        raise InvalidImageError(
            f"Image is too small ({w}x{h}). Use at least "
            f"{config.MIN_IMAGE_SIDE}x{config.MIN_IMAGE_SIDE} pixels."
        )

    warnings = []
    rgb = np.asarray(to_rgb(image), dtype=np.float32)
    # X-rays are grayscale: R, G and B channels are (almost) equal.
    color = (np.abs(rgb[..., 0] - rgb[..., 1]).mean() +
              np.abs(rgb[..., 1] - rgb[..., 2]).mean()) / 2
    if color > config.MAX_COLOR_DEVIATION:
        warnings.append(
            "This image looks like a color photo, not a chest X-ray. "
            "The prediction is probably meaningless."
        )
    aspect = max(w, h) / min(w, h)
    if aspect > 2.0:
        warnings.append(
            f"Unusual aspect ratio ({aspect:.1f}:1). Chest X-rays are roughly "
            "square; the image will be distorted when resized to 224x224."
        )
    return warnings


def prepare_display_image(image: Image.Image) -> Image.Image:
    """RGB copy, EXIF rotation applied, resized to the model input size."""
    return to_rgb(image).resize(config.IMAGE_SIZE, Image.Resampling.BILINEAR)


def preprocess(image: Image.Image) -> np.ndarray:
    """PIL image -> float32 array of shape (1, 224, 224, 3) in [0, 1]."""
    arr = np.asarray(prepare_display_image(image), dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]
