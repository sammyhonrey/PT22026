import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from xray import config, samples
from xray import landmarks as lm


@pytest.fixture
def client(tmp_path, monkeypatch, toy_data):
    """Server with its data and model folders redirected to a temporary place."""
    monkeypatch.setattr(config, "LANDMARK_CSV", tmp_path / "pose_landmarks.csv")
    monkeypatch.setattr(config, "USER_CSV", tmp_path / "user_samples.csv")
    monkeypatch.setattr(config, "CLASSIFIER_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr(samples, "USER_POSES_JSON", tmp_path / "user_poses.json")
    from xray import server

    with TestClient(server.create_app()) as test_client:
        yield test_client


def post_samples(client, label, points):
    body = {"label": label, "world": points.tolist(), "visibility": np.ones((len(points), lm.COUNT)).tolist()}
    return client.post("/api/samples", json=body)


def test_health_and_untrained_model(client):
    assert client.get("/api/health").json() == {"status": "ok", "version": "2.0.0", "model_trained": False}
    model = client.get("/api/model").json()
    assert not model["trained"] and len(model["muscle_groups"]) >= 10


def test_teach_train_and_coach(client, toy_data):
    points, labels = toy_data
    for label in ("standing", "arms_up"):
        reply = post_samples(client, label, points[labels == label])
        assert reply.status_code == 200 and reply.json()["stored"] == 40

    trained = client.post("/api/train", json={"algorithm": "logreg"})
    assert trained.status_code == 200
    assert [c["label"] for c in trained.json()["classes"]] == ["arms_up", "standing"]

    pose = points[labels == "arms_up"][0]
    with client.websocket_connect("/ws/coach") as socket:
        for i in range(5):
            socket.send_json({"t": i * 100, "world": pose.tolist(), "visibility": [1.0] * lm.COUNT})
            reply = socket.receive_json()
        assert reply["label"] == "arms_up"
        socket.send_json({"type": "summary"})
        assert socket.receive_json()["poses"][0]["label"] == "arms_up"
        socket.send_json({"t": 600, "world": [[0, 0, 0]], "visibility": [1.0]})
        assert socket.receive_json()["type"] == "error"


def test_training_needs_enough_samples(client, toy_data):
    points, labels = toy_data
    post_samples(client, "standing", points[labels == "standing"])
    post_samples(client, "arms_up", points[labels == "arms_up"][:5])
    reply = client.post("/api/train", json={"algorithm": "logreg"})
    assert reply.status_code == 422 and "too few samples" in reply.json()["detail"]


def test_bad_requests_are_rejected(client, toy_data):
    points, _ = toy_data
    assert post_samples(client, "???", points[:3]).status_code == 422
    assert client.post("/api/samples", json={"label": "x", "world": [[[0, 0, 0]]], "visibility": [[1]]}).status_code == 422
    assert client.post("/api/train", json={"algorithm": "magic"}).status_code == 422
    assert client.post("/api/analyze", files={"file": ("notes.txt", b"hello", "text/plain")}).status_code == 400


def test_delete_samples(client, toy_data):
    points, labels = toy_data
    post_samples(client, "standing", points[labels == "standing"])
    assert client.delete("/api/samples/standing").json() == {"removed": 40, "counts": {}}


def test_picture_without_a_person(client):
    if not any(config.TASK_DIR.glob("pose_landmarker_full.task")):
        pytest.skip("pose model not downloaded")
    buffer = io.BytesIO()
    Image.new("RGB", (320, 240), (40, 90, 60)).save(buffer, "PNG")
    reply = client.post("/api/analyze", files={"file": ("blank.png", buffer.getvalue(), "image/png")})
    assert reply.status_code == 200 and reply.json()["found"] is False


def test_web_client_is_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "See your Life X-ray" in page.text
