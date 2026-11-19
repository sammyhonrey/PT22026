"""Frame-by-frame coaching: stable pose label, hold time, form score, hints and active muscles."""

from collections import defaultdict

import numpy as np

from . import config, features, form, muscles
from . import landmarks as lm
from .classifier import ANGLE_NAMES

ENTER_CONFIDENCE = 0.65
EXIT_CONFIDENCE = 0.4
SWITCH_SECONDS = 0.35
SMOOTHING_SECONDS = 0.3
MAX_GAP_SECONDS = 1.0


class Coach:
    """Keeps the state of one live session."""

    def __init__(self, classifier, user_poses=None):
        self.classifier = classifier
        self.user_poses = user_poses or {}
        self.reset()

    def reset(self):
        self._smoothed = None
        self._last_time = None
        self._label = config.UNKNOWN
        self._since = 0.0
        self._candidate = None
        self._candidate_since = 0.0
        self._seconds = defaultdict(float)
        self._entries = defaultdict(int)
        self._best_hold = defaultdict(float)
        self._score_sum = defaultdict(float)
        self._score_n = defaultdict(int)
        self._frames = 0

    def update(self, world, visibility, t):
        """Process one frame. `t` is the frame time in seconds."""
        world = np.asarray(world, dtype=np.float64)
        visibility = np.asarray(visibility, dtype=np.float64)
        dt = 0.0 if self._last_time is None else t - self._last_time
        if dt < 0 or dt > MAX_GAP_SECONDS:
            self._smoothed, dt = None, 0.0
            self._set_label(config.UNKNOWN, t)
        self._last_time = t
        self._frames += 1

        if visibility[lm.BODY].mean() < config.MIN_VISIBILITY:
            self._set_label(config.UNKNOWN, t)
            self._smoothed = None
            return self._reply(status="low_visibility", angles=None)

        proba = self.classifier.predict_proba(world)[0]
        if self._smoothed is None:
            self._smoothed = proba
        else:
            alpha = 1.0 - np.exp(-dt / SMOOTHING_SECONDS)
            self._smoothed = self._smoothed + alpha * (proba - self._smoothed)
        self._choose_label(t)

        if self._label != config.UNKNOWN:
            self._seconds[self._label] += dt
            hold = t - self._since
            self._best_hold[self._label] = max(self._best_hold[self._label], hold)

        angles = features.joint_angles(world)[0]
        return self._reply(status="ok", angles=angles, world=world, t=t)

    def _choose_label(self, t):
        classes = self.classifier.classes
        best = int(np.argmax(self._smoothed))
        top, top_p = classes[best], float(self._smoothed[best])
        current_p = float(self._smoothed[classes.index(self._label)]) if self._label in classes else 0.0

        if top == self._label:
            self._candidate = None
        elif top_p >= ENTER_CONFIDENCE:
            if self._candidate != top:
                self._candidate, self._candidate_since = top, t
            if t - self._candidate_since >= SWITCH_SECONDS or self._label == config.UNKNOWN:
                self._set_label(top, t)
        else:
            self._candidate = None
        if self._label != config.UNKNOWN and self._label == top and top_p < EXIT_CONFIDENCE:
            self._set_label(config.UNKNOWN, t)
        elif self._label != config.UNKNOWN and self._label != top and current_p < EXIT_CONFIDENCE:
            self._set_label(config.UNKNOWN, t)

    def _set_label(self, label, t):
        if label == self._label:
            return
        self._label, self._since, self._candidate = label, t, None
        if label != config.UNKNOWN:
            self._entries[label] += 1

    def _reply(self, status, angles, world=None, t=0.0):
        label = self._label
        known = label != config.UNKNOWN
        score, hints = None, []
        if known and world is not None:
            score, hints = form.assess(label, world, self.classifier.angle_reference.get(label))
        if score is not None:
            self._score_sum[label] += score
            self._score_n[label] += 1
        proba = {} if self._smoothed is None else {
            c: round(float(p), 3) for c, p in zip(self.classifier.classes, self._smoothed)}
        info = muscles.describe(label, self.user_poses) if known else {}
        return {
            "status": status,
            "label": label,
            "title": info.get("title", ""),
            "cue": info.get("cue", ""),
            "known": known,
            "confidence": proba.get(label, max(proba.values(), default=0.0)),
            "probabilities": proba,
            "hold_seconds": round(t - self._since, 1) if known else 0.0,
            "repetitions": self._entries[label] if known else 0,
            "score": None if score is None else round(score),
            "hints": hints,
            "muscles": muscles.activation(label, self.user_poses) if known else {},
            "primary": info.get("primary", []),
            "secondary": info.get("secondary", []),
            "angles": {} if angles is None else {n: round(float(a)) for n, a in zip(ANGLE_NAMES, angles)},
        }

    def summary(self):
        poses = []
        for label in sorted(self._seconds, key=lambda k: -self._seconds[k]):
            n = self._score_n[label]
            poses.append({
                "label": label,
                "title": muscles.describe(label, self.user_poses)["title"],
                "seconds": round(self._seconds[label], 1),
                "times_entered": self._entries[label],
                "longest_hold": round(self._best_hold[label], 1),
                "mean_score": round(self._score_sum[label] / n) if n else None,
            })
        return {"frames": self._frames, "poses": poses}
