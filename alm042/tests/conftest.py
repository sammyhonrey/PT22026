"""Shared fixtures for the VoiceForge test suite."""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from voiceforge.app_core import VoiceForgeApp  # noqa: E402
from voiceforge.checkpoint import get_model_config  # noqa: E402
from voiceforge.jobs import JobManager  # noqa: E402
from voiceforge.library import Library  # noqa: E402
from voiceforge.model import AcousticModel  # noqa: E402
from voiceforge.text import CHAR_TO_ID  # noqa: E402
from voiceforge.web_app import create_server  # noqa: E402


@pytest.fixture
def tiny_checkpoint(tmp_path):
    """A very small custom-model checkpoint so tests stay fast."""

    model = AcousticModel(
        vocab_size=len(CHAR_TO_ID),
        n_mels=80,
        embedding_dim=8,
        hidden_dim=8,
        attention_dim=8,
    )

    # Predict roughly 20 Mel frames instead of a random tiny length.
    with torch.no_grad():
        model.duration_projection[2].bias.fill_(3.0)

    path = tmp_path / "tiny.pt"

    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "loss": 1.0,
            "model_config": get_model_config(model),
        },
        path,
    )

    return path


@pytest.fixture
def app(tmp_path, tiny_checkpoint):
    return VoiceForgeApp(
        checkpoint_path=tiny_checkpoint,
        device="cpu",
        output_dir=tmp_path / "app_outputs",
        griffin_lim_iterations=2,
        max_mel_frames=32,
        engine="voiceforge",
    )


@pytest.fixture
def server(app, tmp_path):
    library = Library(
        db_path=tmp_path / "library.db",
        audio_dir=tmp_path / "library",
    )
    jobs = JobManager(app, job_root=tmp_path / "jobs", library=library)

    httpd = create_server(
        app,
        host="127.0.0.1",
        port=0,
        jobs=jobs,
        library=library,
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{httpd.server_address[1]}"

    try:
        yield url
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)



@pytest.fixture
def wav_file(tmp_path):
    """A small valid WAV for library tests."""

    import numpy as np
    import soundfile as sf

    path = tmp_path / "source.wav"
    sample_rate = 22050
    time = np.arange(sample_rate) / sample_rate
    data = (0.2 * np.sin(2 * np.pi * 220 * time)).astype("float32")
    sf.write(str(path), data, sample_rate, subtype="PCM_16")

    return path

def wait_for_job(url: str, job_id: str, timeout: float = 120.0) -> dict:
    """Poll ``GET /api/jobs/{id}`` until it reaches a terminal status."""

    import time

    deadline = time.time() + timeout

    while time.time() < deadline:
        with http_get(f"{url}/api/jobs/{job_id}") as response:
            payload = json.loads(response.read())

        job = payload["job"]

        if job["status"] in ("completed", "failed"):
            return job

        time.sleep(0.1)

    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def http_get(url: str):
    """GET without letting urllib normalise the path."""

    request = urllib.request.Request(url, method="GET")
    return urllib.request.urlopen(request, timeout=15)


def http_post_json(url: str, payload):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=240)


def error_payload(error: urllib.error.HTTPError) -> dict:
    return json.loads(error.read())
