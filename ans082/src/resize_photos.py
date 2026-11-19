"""Resize phone photos so the long side is at most 1024 px.

Usage (from the ans082 folder, venv active):
    python src/resize_photos.py --src "<raw folder>" --dst "<output folder>"
"""
import argparse
from pathlib import Path

from PIL import Image, ImageOps

MAX_SIDE = 1024
EXTS = {".jpg", ".jpeg", ".png"}


def resize_image(src_path: Path, dst_path: Path, max_side: int = MAX_SIDE) -> tuple:
    """Fix phone rotation, convert to RGB, shrink to max_side, save as JPEG."""
    with Image.open(src_path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path, "JPEG", quality=90)
        return img.size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    args = parser.parse_args()

    counts = {}
    for class_dir in sorted(p for p in args.src.iterdir() if p.is_dir()):
        n = 0
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() not in EXTS:
                print(f"skipped (unsupported type): {f}")
                continue
            n += 1
            resize_image(f, args.dst / class_dir.name / f"{class_dir.name}_{n:03d}.jpg")
        counts[class_dir.name] = n

    print("\nImages per class:")
    for name, n in counts.items():
        print(f"  {name:12s}{n:4d}")
    print(f"  {'TOTAL':12s}{sum(counts.values()):4d}")


if __name__ == "__main__":
    main()