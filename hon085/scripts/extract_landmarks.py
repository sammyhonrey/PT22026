"""Run the pose landmarker over the downloaded images and videos and store the landmarks as CSV.

Yoga images carry their label in the folder name. Squat video frames are
labelled from the knee angle: both knees straight is "standing", both clearly
bent is "squat", and the frames in between are left out. The thresholds follow
the 3D landmarks of this dataset, where a relaxed standing knee reads about
150 degrees and the bottom of a squat about 100.
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xray import config, dataset, features
from xray import landmarks as lm
from xray.pose_estimator import PoseEstimator, PoseEstimatorError

SUBJECT = re.compile(r"^([a-z]+\d*)_")
PERSON = re.compile(r"(\d+)$")
STANDING_KNEE = 142.0
SQUAT_KNEE = 118.0
KNEES = [list(lm.ANGLES).index("left_knee"), list(lm.ANGLES).index("right_knee")]


def yoga_rows(estimator, folder, missed):
    files = sorted(folder.glob("*/*/*.jpg"))
    for i, path in enumerate(files, 1):
        split, label = path.parts[-3], path.parts[-2]
        match = SUBJECT.match(path.name)
        subject = match.group(1) if match else "unknown"
        if subject == "guy":
            subject = "guy1"
        found = estimator.detect(np.asarray(Image.open(path).convert("RGB")))
        if found is None:
            missed[label] += 1
        else:
            yield dataset.make_row(label, subject, split, path.name, found["world"], found["visibility"])
        if i % 100 == 0 or i == len(files):
            print(f"\r  yoga images {i}/{len(files)}", end="", flush=True)
    print()


def squat_label(found):
    knees = features.joint_angles(found["world"])[0, KNEES]
    if knees.min() >= STANDING_KNEE:
        return "standing"
    if knees.max() <= SQUAT_KNEE:
        return "squat"
    return None


def squat_rows(estimator, folder, every, per_label, rng):
    import cv2

    videos = sorted(folder.glob("*/*.mp4"))
    for n, path in enumerate(videos, 1):
        person = int(PERSON.search(path.stem).group(1))
        split = "test" if person in config.SQUAT_TEST_PEOPLE else "train"
        capture = cv2.VideoCapture(str(path))
        candidates = defaultdict(list)
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % every == 0:
                found = estimator.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                label = squat_label(found) if found else None
                if label:
                    candidates[label].append((index, found))
            index += 1
        capture.release()
        for label, items in candidates.items():
            keep = sorted(rng.choice(len(items), size=min(per_label, len(items)), replace=False))
            for k in keep:
                frame_index, found = items[k]
                yield dataset.make_row(label, f"sq{person:02d}", split, f"{path.parent.name}/{path.name}#{frame_index}",
                                       found["world"], found["visibility"])
        print(f"\r  squat videos {n}/{len(videos)}", end="", flush=True)
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=config.LANDMARK_CSV)
    parser.add_argument("--variant", default="heavy", choices=("lite", "full", "heavy"))
    parser.add_argument("--every", type=int, default=6, help="use every n-th video frame")
    parser.add_argument("--per-label", type=int, default=12, help="frames kept per label and video")
    args = parser.parse_args()

    if not (config.YOGA_DIR / "train").is_dir() or not any(config.SQUAT_DIR.glob("*/*.mp4")):
        sys.exit("Raw data is missing. Run scripts/download_data.py first.")

    missed = Counter()
    rng = np.random.default_rng(7)
    try:
        with PoseEstimator(args.variant) as estimator:
            rows = list(yoga_rows(estimator, config.YOGA_DIR, missed))
            rows += list(squat_rows(estimator, config.SQUAT_DIR, max(1, args.every), args.per_label, rng))
    except PoseEstimatorError as exc:
        sys.exit(str(exc))
    dataset.write_rows(args.out, rows)
    counts = Counter(row[0] for row in rows)
    print(f"{len(rows)} poses written to {args.out}")
    print("  per class:", dict(sorted(counts.items())))
    if missed:
        print("  images with no person found:", dict(missed))


if __name__ == "__main__":
    main()
