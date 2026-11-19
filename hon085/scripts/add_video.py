"""Extract pose samples from a video or a folder of images and add them under one label.

Use it to teach the classifier a new pose from your own recordings:

    python scripts/add_video.py --label squat my_squat.mp4
    python scripts/train.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xray import config, dataset
from xray.pose_estimator import PoseEstimator, PoseEstimatorError

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def video_frames(path, every):
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"cannot open {path}")
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % every == 0:
            yield f"{path.name}#{index}", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        index += 1
    capture.release()


def image_frames(folder):
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in IMAGE_TYPES:
            yield path.name, np.asarray(Image.open(path).convert("RGB"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="video file or folder of images")
    parser.add_argument("--label", required=True)
    parser.add_argument("--subject", default="me")
    parser.add_argument("--every", type=int, default=5, help="use every n-th video frame")
    parser.add_argument("--variant", default="full", choices=("lite", "full", "heavy"))
    parser.add_argument("--out", type=Path, default=config.USER_CSV)
    args = parser.parse_args()

    if not args.source.exists():
        sys.exit(f"{args.source} does not exist")
    frames = image_frames(args.source) if args.source.is_dir() else video_frames(args.source, max(1, args.every))

    rows = []
    try:
        with PoseEstimator(args.variant) as estimator:
            for name, image in frames:
                found = estimator.detect(image)
                if found is not None and found["visibility"].mean() >= config.MIN_VISIBILITY:
                    rows.append(dataset.make_row(
                        args.label, args.subject, "user", name, found["world"], found["visibility"]))
    except (PoseEstimatorError, OSError) as exc:
        sys.exit(str(exc))
    if not rows:
        sys.exit("No person was found in the source.")
    dataset.write_rows(args.out, rows, append=True)
    print(f"{len(rows)} samples of '{args.label}' added to {args.out}")


if __name__ == "__main__":
    main()
