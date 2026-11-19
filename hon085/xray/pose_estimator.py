"""Pose landmark detection on still images with MediaPipe Pose Landmarker."""

from pathlib import Path

import numpy as np

from . import config

VARIANTS = ("lite", "full", "heavy")


class PoseEstimatorError(RuntimeError):
    pass


class PoseEstimator:
    def __init__(self, variant="heavy", min_confidence=0.5):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as tasks
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise PoseEstimatorError("mediapipe is not installed; run: pip install -r requirements.txt") from exc

        task = Path(config.TASK_DIR) / f"pose_landmarker_{variant}.task"
        if not task.exists():
            raise PoseEstimatorError(f"{task.name} is missing; run: python scripts/fetch_assets.py")
        self._mp = mp
        self.variant = variant
        options = vision.PoseLandmarkerOptions(
            base_options=tasks.BaseOptions(model_asset_path=str(task)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, image):
        """Return a dict with image and world landmarks, or None when nobody is found.

        `image` is an RGB uint8 array of shape (h, w, 3).
        """
        frame = np.ascontiguousarray(image, dtype=np.uint8)
        result = self._landmarker.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame))
        if not result.pose_world_landmarks:
            return None
        world = result.pose_world_landmarks[0]
        image_points = result.pose_landmarks[0]
        return {
            "world": np.array([[p.x, p.y, p.z] for p in world], dtype=np.float32),
            "image": np.array([[p.x, p.y, p.z] for p in image_points], dtype=np.float32),
            "visibility": np.array([p.visibility for p in world], dtype=np.float32),
        }

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
