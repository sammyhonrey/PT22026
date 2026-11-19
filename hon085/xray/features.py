"""Turn raw pose landmarks into a view- and size-independent feature vector.

Input landmarks are MediaPipe world landmarks: 33 points in metres with x to
the right of the image, y pointing down and z pointing away from the camera.
"""

import numpy as np

from . import landmarks as lm

FEATURE_SETS = ("raw", "normalized", "aligned", "full")
DEFAULT_SET = "full"

_L_SH, _R_SH = lm.INDEX["left_shoulder"], lm.INDEX["right_shoulder"]
_L_HIP, _R_HIP = lm.INDEX["left_hip"], lm.INDEX["right_hip"]
_ANGLE_IDX = np.array([[lm.INDEX[n] for n in triple] for triple in lm.ANGLES.values()])
_DIST_IDX = np.array([[lm.INDEX[a], lm.INDEX[b]] for a, b in lm.DISTANCES])


def as_batch(points):
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 2:
        pts = pts[None]
    if pts.shape[1:] != (lm.COUNT, 3):
        raise ValueError(f"expected (n, {lm.COUNT}, 3) landmarks, got {pts.shape}")
    return pts


def centre_and_scale(pts):
    """Move the hip centre to the origin, flip y upwards and divide by torso length."""
    pts = as_batch(pts).copy()
    pts[..., 1] *= -1.0
    hips = (pts[:, _L_HIP] + pts[:, _R_HIP]) / 2.0
    shoulders = (pts[:, _L_SH] + pts[:, _R_SH]) / 2.0
    torso = np.linalg.norm(shoulders - hips, axis=1)
    torso = np.where(torso < 1e-6, 1.0, torso)
    return (pts - hips[:, None]) / torso[:, None, None]


def align_yaw(pts):
    """Rotate around the vertical axis so that the body's left-right line lies on x."""
    across = (pts[:, _L_SH] - pts[:, _R_SH]) + (pts[:, _L_HIP] - pts[:, _R_HIP])
    yaw = np.arctan2(across[:, 2], across[:, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    out = pts.copy()
    out[..., 0] = c[:, None] * pts[..., 0] + s[:, None] * pts[..., 2]
    out[..., 2] = -s[:, None] * pts[..., 0] + c[:, None] * pts[..., 2]
    return out


def joint_angles(pts):
    """Inner angle at each joint in degrees, shape (n, len(ANGLES))."""
    pts = as_batch(pts)
    a, b, c = (pts[:, _ANGLE_IDX[:, k]] for k in range(3))
    u, v = a - b, c - b
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def trunk_lean(pts):
    """Angle between the hip-to-shoulder line and the vertical, in degrees."""
    norm = centre_and_scale(pts)
    spine = (norm[:, _L_SH] + norm[:, _R_SH]) / 2.0
    cos = spine[:, 1] / (np.linalg.norm(spine, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def extract(points, feature_set=DEFAULT_SET):
    """Feature matrix of shape (n, d) for one pose or a batch of poses."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}")
    pts = as_batch(points)
    if feature_set == "raw":
        return pts[:, lm.BODY].reshape(len(pts), -1)
    norm = centre_and_scale(pts)
    if feature_set == "normalized":
        return norm[:, lm.BODY].reshape(len(pts), -1)
    aligned = align_yaw(norm)
    coords = aligned[:, lm.BODY].reshape(len(pts), -1)
    if feature_set == "aligned":
        return coords
    angles = joint_angles(pts) / 180.0
    lean = trunk_lean(pts)[:, None] / 180.0
    dists = np.linalg.norm(norm[:, _DIST_IDX[:, 0]] - norm[:, _DIST_IDX[:, 1]], axis=-1)
    return np.hstack([coords, angles, lean, dists])


def feature_names(feature_set=DEFAULT_SET):
    coords = [f"{lm.NAMES[i]}_{axis}" for i in lm.BODY for axis in "xyz"]
    if feature_set != "full":
        return coords
    angles = [f"angle_{name}" for name in lm.ANGLES]
    dists = [f"dist_{a}__{b}" for a, b in lm.DISTANCES]
    return coords + angles + ["trunk_lean"] + dists


def mirror(points):
    """Swap the left and right side of the body."""
    pts = as_batch(points)[:, lm.MIRROR].copy()
    pts[..., 0] *= -1.0
    return pts
