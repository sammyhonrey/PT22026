"""FastAPI application: serves the web client and the pose recognition API."""

import io
import json
import pickle
import threading
from contextlib import asynccontextmanager

import numpy as np
import sklearn
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from . import __version__, config, features, muscles, samples, trainer, training
from . import landmarks as lm
from .classifier import ANGLE_NAMES, ModelNotTrained, PoseClassifier
from .coach import Coach
from .pose_estimator import PoseEstimator, PoseEstimatorError

MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_SIDE = 1600
MAX_SAMPLES_PER_REQUEST = 600
APP_VALIDATION_FOLDS = 5


class ModelService:
    """Owns the classifier and the still-image pose estimator."""

    def __init__(self):
        self.classifier = None
        self._estimator = None
        self._estimator_lock = threading.Lock()
        self._train_lock = threading.Lock()
        self.reload()

    def reload(self):
        """Load the saved classifier; retrain it when it was saved by another scikit-learn version."""
        try:
            model = PoseClassifier.load()
            if model.info.get("sklearn") != sklearn.__version__:
                raise ModelNotTrained("saved with a different scikit-learn version")
            self.classifier = model
        except (ModelNotTrained, pickle.UnpicklingError, AttributeError, ImportError, EOFError, ValueError):
            self.classifier = self._retrain()

    @staticmethod
    def _retrain():
        try:
            return trainer.train_and_save(validate=False)
        except (FileNotFoundError, ValueError):
            return None

    def require(self):
        if self.classifier is None:
            raise HTTPException(503, "The pose classifier is not trained yet. Run: python scripts/train.py")
        return self.classifier

    def detect(self, image):
        with self._estimator_lock:
            if self._estimator is None:
                self._estimator = PoseEstimator("full")
            return self._estimator.detect(image)

    def train(self, algorithm):
        if not self._train_lock.acquire(blocking=False):
            raise HTTPException(409, "Training is already running")
        try:
            self.classifier = trainer.train_and_save(algorithm, max_folds=APP_VALIDATION_FOLDS)
            return self.classifier
        finally:
            self._train_lock.release()

    def close(self):
        if self._estimator is not None:
            self._estimator.close()


class SampleBatch(BaseModel):
    label: str = Field(min_length=1, max_length=40)
    world: list[list[list[float]]] = Field(max_length=MAX_SAMPLES_PER_REQUEST)
    visibility: list[list[float]] = Field(max_length=MAX_SAMPLES_PER_REQUEST)
    primary: list[str] = []
    secondary: list[str] = []


class TrainRequest(BaseModel):
    algorithm: str = training.DEFAULT_ALGORITHM


def model_payload(service):
    model = service.classifier
    user = samples.user_poses()
    payload = {
        "version": __version__,
        "trained": model is not None,
        "algorithms": {key: name for key, (name, _) in training.ALGORITHMS.items()},
        "muscle_groups": [{"id": g, "title": muscles.GROUP_TITLES[g], "pattern": p} for g, p in muscles.GROUPS.items()],
        "user_samples": samples.counts(),
    }
    if model is not None:
        payload.update({
            "info": model.info,
            "classes": [{"label": c, **muscles.describe(c, user)} for c in model.classes],
            "angle_names": ANGLE_NAMES,
            "angle_reference": model.angle_reference,
        })
    if config.METRICS_PATH.exists():
        try:
            payload["evaluation"] = json.loads(config.METRICS_PATH.read_text(encoding="utf-8")).get("summary")
        except (OSError, json.JSONDecodeError):
            pass
    return payload


def create_app():
    service = ModelService()

    @asynccontextmanager
    async def lifespan(_app):
        yield
        service.close()

    app = FastAPI(title="See your Life X-ray", version=__version__, lifespan=lifespan)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": __version__, "model_trained": service.classifier is not None}

    @app.get("/api/model")
    def model():
        return model_payload(service)

    @app.post("/api/analyze")
    def analyze(file: UploadFile = File(...)):
        """Find the pose in an uploaded picture and classify it."""
        data = file.file.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "The picture is larger than 12 MB")
        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, "The file is not a picture this app can read")
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        try:
            found = service.detect(np.asarray(image))
        except PoseEstimatorError as exc:
            raise HTTPException(503, str(exc))
        if found is None:
            return {"found": False, "width": image.width, "height": image.height}

        classifier = service.require()
        result = classifier.predict(found["world"])
        user = samples.user_poses()
        angles = features.joint_angles(found["world"])[0]
        return {
            "found": True,
            "width": image.width,
            "height": image.height,
            "landmarks": found["image"][:, :2].round(4).tolist(),
            "visibility": found["visibility"].round(3).tolist(),
            **result,
            **(muscles.describe(result["label"], user) if result["known"] else {}),
            "muscles": muscles.activation(result["label"], user),
            "angles": {name: round(float(a)) for name, a in zip(ANGLE_NAMES, angles)},
        }

    @app.get("/api/samples")
    def list_samples():
        return {"counts": samples.counts(), "poses": samples.user_poses()}

    @app.post("/api/samples")
    def add_samples(batch: SampleBatch):
        try:
            label, stored = samples.add(batch.label, batch.world, batch.visibility, batch.primary, batch.secondary)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return {"label": label, "stored": stored, "counts": samples.counts()}

    @app.delete("/api/samples/{label}")
    def delete_samples(label: str):
        try:
            removed = samples.remove(label)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return {"removed": removed, "counts": samples.counts()}

    @app.post("/api/train")
    def train(request: TrainRequest):
        if request.algorithm not in training.ALGORITHMS:
            raise HTTPException(422, f"Unknown algorithm {request.algorithm!r}")
        try:
            service.train(request.algorithm)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(422, str(exc))
        return model_payload(service)

    @app.websocket("/ws/coach")
    async def coach_socket(socket: WebSocket):
        """Live coaching: the client sends landmarks for each frame and gets the coach's reply."""
        await socket.accept()
        coach = None
        try:
            while True:
                message = await socket.receive_json()
                if message.get("type") == "summary":
                    await socket.send_json({"type": "summary", **(coach.summary() if coach else {"frames": 0, "poses": []})})
                    continue
                if message.get("type") == "reset":
                    coach = None
                    continue
                if service.classifier is None:
                    await socket.send_json({"type": "coach", "status": "no_model"})
                    continue
                if coach is None or coach.classifier is not service.classifier:
                    coach = Coach(service.classifier, samples.user_poses())
                try:
                    world = np.asarray(message["world"], dtype=np.float64)
                    visibility = np.asarray(message["visibility"], dtype=np.float64)
                    if world.shape != (lm.COUNT, 3) or visibility.shape != (lm.COUNT,):
                        raise ValueError("wrong landmark shape")
                    reply = coach.update(world, visibility, float(message["t"]) / 1000.0)
                except (KeyError, TypeError, ValueError):
                    await socket.send_json({"type": "error", "detail": "bad frame"})
                    continue
                await socket.send_json({"type": "coach", **reply})
        except WebSocketDisconnect:
            pass

    @app.get("/")
    def index():
        return FileResponse(config.WEB_DIR / "index.html")

    app.mount("/", StaticFiles(directory=config.WEB_DIR), name="web")
    return app


app = create_app()
