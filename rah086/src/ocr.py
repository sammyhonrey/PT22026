"""OCR module: extracts text from an image of a food ingredient label."""

import sys
from functools import lru_cache

import easyocr


@lru_cache(maxsize=1)
def get_reader(languages=("en",)):
    """Load the EasyOCR reader once and reuse it (loading is slow)."""
    return easyocr.Reader(list(languages), gpu=False)


def extract_text(image_path):
    """Read an image and return the detected text as a single string.

    Args:
        image_path: path to an image file (jpg, png, ...).

    Returns:
        The extracted text, or an empty string if nothing was found.

    Raises:
        FileNotFoundError: if the image file does not exist.
    """
    try:
        reader = get_reader()
        lines = reader.readtext(image_path, detail=0, paragraph=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return " ".join(lines).strip()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.ocr <image_path>")
        sys.exit(1)
    print(extract_text(sys.argv[1]))