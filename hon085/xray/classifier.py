"""Pose classifier with a confidence threshold and per-pose joint angle references."""

import time
from pathlib import Path

import joblib
import numpy as np
import sklearn

from . import config, features, training
from . import landmarks as lm

ANGLE_NAMES = list(lm.ANGLES)
_L_KNEE, _R_KNEE = ANGLE_NAMES.index("left_knee"), ANGLE_NAMES.index("right_knee")
_L_SHOULDER, _R_SHOULDER = ANGLE_NAMES.index("left_shoulder"), ANGLE_NAMES.index("right_shoulder")
MIN_CONFIDENCE = 0.6


class ModelNotTrained(RuntimeError):
    pass


def canonical_side(points):
    """Mirror poses so that the more bent knee (then the lower arm) is on the left.

    One-sided poses such as tree or warrior then share one reference. Returns the
    poses and a boolean array telling which ones were mirrored.
    """
    pts = features.as_batch(points)
    angles = features.joint_angles(pts)
    knee_gap = angles[:, _L_KNEE] - angles[:, _R_KNEE]
    arm_gap = angles[:, _L_SHOULDER] - angles[:, _R_SHOULDER]
    flip = np.where(np.abs(knee_gap) > 10.0, knee_gap > 0, arm_gap > 10.0)
    out = pts.copy()
    if flip.any():
        out[flip] = features.mirror(pts[flip])
    return out, flip


class PoseClassifier:
    def __init__(self, algorithm=training.DEFAULT_ALGORITHM, feature_set=features.DEFAULT_SET):
        self.algorithm = algorithm
        self.feature_set = feature_set
        self.pipeline = None
        self.classes = []
        self.angle_reference = {}
        self.info = {}

    def fit(self, points, labels, mirror=True):
        points = features.as_batch(points)
        labels = np.asarray(labels)
        if len(set(labels.tolist())) < 2:
            raise ValueError("at least two different poses are needed for training")
        x_pts, y = training.augment(points, labels, mirror=mirror)
        x = features.extract(x_pts, self.feature_set)
        self.pipeline = training.build(self.algorithm).fit(x, y)
        self.classes = [str(c) for c in self.pipeline.classes_]
        self._fit_angles(points, labels)
        self.info = {
            "algorithm": self.algorithm,
            "algorithm_name": training.ALGORITHMS[self.algorithm][0],
            "feature_set": self.feature_set,
            "n_features": int(x.shape[1]),
            "n_samples": int(len(points)),
            "n_training_rows": int(len(x)),
            "per_class": {c: int((labels == c).sum()) for c in self.classes},
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sklearn": sklearn.__version__,
        }
        return self

    def _fit_angles(self, points, labels):
        canon, _ = canonical_side(points)
        angles = features.joint_angles(canon)
        self.angle_reference = {}
        for c in self.classes:
            rows = angles[labels == c]
            self.angle_reference[c] = {
                "mean": rows.mean(axis=0).round(1).tolist(),
                "std": np.maximum(rows.std(axis=0), 5.0).round(1).tolist(),
            }

    def predict_proba(self, points):
        if self.pipeline is None:
            raise ModelNotTrained("the pose classifier has not been trained yet")
        return self.pipeline.predict_proba(features.extract(points, self.feature_set))

    def predict(self, points, min_confidence=MIN_CONFIDENCE):
        """Classify one pose. A pose the model is not confident about comes back as unknown."""
        proba = self.predict_proba(points)[0]
        best = int(np.argmax(proba))
        known = proba[best] >= min_confidence
        return {
            "label": self.classes[best] if known else config.UNKNOWN,
            "closest": self.classes[best],
            "confidence": float(proba[best]),
            "known": bool(known),
            "probabilities": {c: float(p) for c, p in zip(self.classes, proba)},
        }

    def save(self, path=None):
        path = Path(path or config.CLASSIFIER_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.__dict__, path, compress=3)
        return path

    @classmethod
    def load(cls, path=None):
        path = Path(path or config.CLASSIFIER_PATH)
        if not path.exists():
            raise ModelNotTrained(f"{path.name} not found; run: python scripts/train.py")
        model = cls()
        model.__dict__.update(joblib.load(path))
        return model
