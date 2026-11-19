"""Download the two public sources used for training.

1. Yoga pose images (TensorFlow pose classification tutorial, Apache 2.0):
   five poses performed by seven rendered characters.
2. Dual-View Squat Movement Dataset (github.com/nihar787airobotics/squat_dataset_publication):
   front and side videos of 20 adults doing bodyweight squats, used for
   "standing" and "squat".
"""

import argparse
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xray import config


def fetch(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as fh:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while chunk := response.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {target.name}: {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(target)


def squat_videos():
    for person in range(1, config.SQUAT_PEOPLE + 1):
        yield "front", f"person_{person}"
        yield "side", f"side_person{person}"


def download_yoga(force):
    if (config.YOGA_DIR / "train").is_dir() and not force:
        print(f"Yoga images already in {config.YOGA_DIR}")
        return
    archive = config.RAW_DIR / "yoga_poses.zip"
    fetch(config.YOGA_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(config.YOGA_DIR)
    archive.unlink()
    print(f"  {sum(1 for _ in config.YOGA_DIR.rglob('*.jpg'))} images unpacked")


def download_squats(force):
    for view, name in squat_videos():
        target = config.SQUAT_DIR / view / f"{name}.mp4"
        if target.exists() and not force:
            continue
        fetch(config.SQUAT_URL.format(view=view, name=name), target)
    print(f"Squat videos in {config.SQUAT_DIR}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="download again even if the files exist")
    args = parser.parse_args()
    try:
        download_yoga(args.force)
        download_squats(args.force)
    except (urllib.error.URLError, OSError) as exc:
        sys.exit(f"Download failed: {exc}")


if __name__ == "__main__":
    main()
