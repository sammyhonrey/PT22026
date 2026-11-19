"""Form score and correction hints for a recognised pose.

Built-in poses are checked against a few alignment rules. The limits are in
degrees as measured on the 3D landmarks, where a straight standing knee reads
about 150 to 160 rather than 180. Poses taught by the user have no rules, so
they are compared with the joint angles recorded for them.
"""

from dataclasses import dataclass

import numpy as np

from . import features
from . import landmarks as lm
from .classifier import ANGLE_NAMES, canonical_side

TOLERANCE_DEGREES = 40.0
STABLE_STD = 20.0
HINT_STD = 2.0
HINT_MIN_DEGREES = 20.0
MAX_HINTS = 2
KNEES_IN_RATIO = 0.7

_IDX = {name: i for i, name in enumerate(ANGLE_NAMES)}
_PAIRS = {joint: (_IDX[f"left_{joint}"], _IDX[f"right_{joint}"]) for joint in ("elbow", "shoulder", "hip", "knee")}
_KNEES = (lm.INDEX["left_knee"], lm.INDEX["right_knee"])
_ANKLES = (lm.INDEX["left_ankle"], lm.INDEX["right_ankle"])


@dataclass(frozen=True)
class Check:
    measure: str
    low: float | None
    high: float | None
    hint: str


CHECKS = {
    "standing": [],
    "chair": [
        Check("knee_max", None, 135, "Bend your knees more and sit back"),
        Check("shoulder_min", 115, None, "Reach your arms up beside your ears"),
        Check("trunk_lean", None, 50, "Lift your chest"),
    ],
    "tree": [
        Check("knee_min", None, 105, "Bring the lifted foot higher on the standing leg"),
        Check("knee_max", 140, None, "Straighten your standing leg"),
        Check("trunk_lean", None, 18, "Stand tall, do not lean"),
    ],
    "warrior": [
        Check("trunk_lean", 55, None, "Tip your body further forward"),
        Check("knee_min", 135, None, "Keep both legs straight"),
    ],
    "cobra": [
        Check("trunk_lean", None, 85, "Lift your chest higher"),
        Check("knee_min", 135, None, "Keep your legs long on the floor"),
    ],
    "dog": [
        Check("hip_max", None, 105, "Push your hips up and back"),
        Check("knee_min", 130, None, "Straighten your legs as far as is comfortable"),
        Check("elbow_min", 130, None, "Straighten your arms"),
    ],
    "squat": [
        Check("knee_gap", KNEES_IN_RATIO, None, "Push your knees out over your toes"),
        Check("trunk_lean", None, 70, "Keep your chest up"),
    ],
}


def measures(world):
    """Quantities the checks refer to, from one pose."""
    angles = features.joint_angles(world)[0]
    out = {"trunk_lean": float(features.trunk_lean(world)[0])}
    for joint, (left, right) in _PAIRS.items():
        out[f"{joint}_min"] = float(min(angles[left], angles[right]))
        out[f"{joint}_max"] = float(max(angles[left], angles[right]))
    pts = np.asarray(world, dtype=np.float64).reshape(lm.COUNT, 3)
    ankles = np.linalg.norm(pts[_ANKLES[0]] - pts[_ANKLES[1]])
    knees = np.linalg.norm(pts[_KNEES[0]] - pts[_KNEES[1]])
    out["knee_gap"] = float(knees / ankles) if ankles > 0.05 else 1.0
    return out


def _by_rules(checks, world):
    values = measures(world)
    credit, hints = [], []
    for check in checks:
        value = values[check.measure]
        if check.measure == "knee_gap":
            miss = max(0.0, check.low - value) * 100.0
        else:
            miss = max(0.0, (check.low or value) - value, value - (check.high if check.high is not None else value))
        credit.append(max(0.0, 1.0 - miss / TOLERANCE_DEGREES))
        if miss > 0:
            hints.append((miss, check.hint))
    hints = [text for _, text in sorted(hints, reverse=True)[:MAX_HINTS]]
    return 100.0 * float(np.mean(credit)), hints


def _by_reference(reference, world):
    canon, flipped = canonical_side(world)
    angles = features.joint_angles(canon)[0]
    mean, std = np.array(reference["mean"]), np.array(reference["std"])
    stable = [i for i, name in enumerate(ANGLE_NAMES) if "ankle" not in name and std[i] <= STABLE_STD]
    if not stable:
        return None, []
    deviation = np.abs(angles - mean)
    score = 100.0 * float(np.clip(1.0 - deviation[stable] / (TOLERANCE_DEGREES * 1.5), 0.0, 1.0).mean())
    limit = np.maximum(HINT_STD * std, HINT_MIN_DEGREES)
    hints = []
    for i in sorted(stable, key=lambda k: -deviation[k] / limit[k]):
        if deviation[i] <= limit[i] or len(hints) == MAX_HINTS:
            break
        side, joint = ANGLE_NAMES[i].split("_")
        if flipped[0]:
            side = "right" if side == "left" else "left"
        action = "Open" if angles[i] < mean[i] else "Close"
        hints.append(f"{action} the angle at your {side} {joint} by about {round(deviation[i] / 5) * 5} degrees")
    return score, hints


def assess(label, world, reference=None):
    """Return (score 0..100 or None, list of hints) for a pose label."""
    if label in CHECKS:
        return _by_rules(CHECKS[label], world) if CHECKS[label] else (None, [])
    if reference:
        return _by_reference(reference, world)
    return None, []
