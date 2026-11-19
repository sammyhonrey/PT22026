import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xray import landmarks as lm


def standing_pose():
    """A simple upright figure in MediaPipe world coordinates (y down, metres)."""
    pts = np.zeros((lm.COUNT, 3))
    place = {
        "nose": (0.0, -0.62, -0.08),
        "left_shoulder": (0.18, -0.45, 0.0), "right_shoulder": (-0.18, -0.45, 0.0),
        "left_elbow": (0.22, -0.18, 0.0), "right_elbow": (-0.22, -0.18, 0.0),
        "left_wrist": (0.24, 0.08, 0.0), "right_wrist": (-0.24, 0.08, 0.0),
        "left_hip": (0.1, 0.0, 0.0), "right_hip": (-0.1, 0.0, 0.0),
        "left_knee": (0.1, 0.42, 0.0), "right_knee": (-0.1, 0.42, 0.0),
        "left_ankle": (0.1, 0.84, 0.0), "right_ankle": (-0.1, 0.84, 0.0),
        "left_heel": (0.1, 0.88, 0.04), "right_heel": (-0.1, 0.88, 0.04),
        "left_foot_index": (0.1, 0.9, -0.14), "right_foot_index": (-0.1, 0.9, -0.14),
    }
    for name, xyz in place.items():
        pts[lm.INDEX[name]] = xyz
    for name in lm.NAMES:
        if name not in place:
            side = 1 if "left" in name else -1
            anchor = "nose" if any(k in name for k in ("eye", "ear", "mouth")) else f"{'left' if side > 0 else 'right'}_wrist"
            pts[lm.INDEX[name]] = pts[lm.INDEX[anchor]] + (0.02 * side, 0.0, 0.0)
    return pts


def arms_up_pose():
    pts = standing_pose()
    for side, sign in (("left", 1), ("right", -1)):
        pts[lm.INDEX[f"{side}_elbow"]] = (0.2 * sign, -0.72, 0.0)
        pts[lm.INDEX[f"{side}_wrist"]] = (0.2 * sign, -0.98, 0.0)
        for finger in ("pinky", "index", "thumb"):
            pts[lm.INDEX[f"{side}_{finger}"]] = (0.2 * sign, -1.02, 0.0)
    return pts


def one_leg_pose():
    pts = standing_pose()
    pts[lm.INDEX["left_knee"]] = (0.38, 0.3, -0.05)
    pts[lm.INDEX["left_ankle"]] = (0.0, 0.38, -0.02)
    pts[lm.INDEX["left_heel"]] = (-0.02, 0.4, 0.0)
    pts[lm.INDEX["left_foot_index"]] = (-0.04, 0.5, -0.05)
    return pts


@pytest.fixture
def poses():
    return {"standing": standing_pose(), "arms_up": arms_up_pose(), "one_leg": one_leg_pose()}


@pytest.fixture
def toy_data(poses):
    rng = np.random.default_rng(0)
    points, labels = [], []
    for label, base in poses.items():
        for _ in range(40):
            points.append(base + rng.normal(0, 0.015, base.shape))
            labels.append(label)
    return np.array(points), np.array(labels)
