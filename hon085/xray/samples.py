"""Poses recorded by the user in the Teach tab."""

import json
import re
from collections import Counter

import numpy as np

from . import config, dataset, muscles
from . import landmarks as lm

USER_POSES_JSON = config.DATA_DIR / "user_poses.json"
_LABEL = re.compile(r"[^a-z0-9]+")
MAX_LABEL_LENGTH = 24


def clean_label(text):
    label = _LABEL.sub("_", str(text).strip().lower()).strip("_")[:MAX_LABEL_LENGTH]
    if not label or label == config.UNKNOWN:
        raise ValueError("give the pose a short name made of letters or digits")
    return label


def user_poses():
    if not USER_POSES_JSON.exists():
        return {}
    try:
        return json.loads(USER_POSES_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_user_poses(poses):
    USER_POSES_JSON.parent.mkdir(parents=True, exist_ok=True)
    USER_POSES_JSON.write_text(json.dumps(poses, indent=2), encoding="utf-8")


def add(label, worlds, visibilities, primary=(), secondary=()):
    """Append recorded poses under a label and remember which muscles it works."""
    label = clean_label(label)
    worlds = np.asarray(worlds, dtype=np.float32)
    visibilities = np.asarray(visibilities, dtype=np.float32)
    if worlds.ndim != 3 or worlds.shape[1:] != (lm.COUNT, 3) or visibilities.shape != worlds.shape[:2]:
        raise ValueError("samples must be lists of 33 landmarks with visibility")
    if not np.isfinite(worlds).all():
        raise ValueError("samples contain invalid numbers")
    keep = visibilities[:, lm.BODY].mean(axis=1) >= config.MIN_VISIBILITY
    rows = [dataset.make_row(label, "me", "user", "webcam", w, v) for w, v in zip(worlds[keep], visibilities[keep])]
    if rows:
        dataset.write_rows(config.USER_CSV, rows, append=True)
    if label not in muscles.POSES:
        poses = user_poses()
        entry = poses.setdefault(label, {"title": label.replace("_", " ").title()})
        if primary or secondary:
            entry["primary"] = [g for g in primary if g in muscles.GROUPS]
            entry["secondary"] = [g for g in secondary if g in muscles.GROUPS and g not in entry["primary"]]
        _save_user_poses(poses)
    return label, len(rows)


def counts():
    if not config.USER_CSV.exists():
        return {}
    return dict(Counter(dataset.load(config.USER_CSV).labels.tolist()))


def remove(label):
    """Delete all recorded samples of one label. Returns how many were removed."""
    label = clean_label(label)
    removed = 0
    if config.USER_CSV.exists():
        table = dataset.load(config.USER_CSV)
        keep = table.labels != label
        removed = int((~keep).sum())
        kept = table.select(keep)
        rows = [dataset.make_row(*meta, w, v) for *meta, w, v in zip(
            kept.labels, kept.subjects, kept.splits, kept.sources, kept.points, kept.visibility)]
        dataset.write_rows(config.USER_CSV, rows)
    poses = user_poses()
    if poses.pop(label, None) is not None:
        _save_user_poses(poses)
    return removed
