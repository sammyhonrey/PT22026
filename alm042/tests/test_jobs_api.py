"""Tests for jobs, ZIP archives and TXT uploads."""

from __future__ import annotations

import base64
import json
import threading
import urllib.error
import zipfile

import io

import pytest

from conftest import error_payload, http_get, http_post_json, wait_for_job
from voiceforge.app_core import AppError
from voiceforge.jobs import JobManager, JobQueueFull


def submit(server, payload):
    return http_post_json(server + "/api/jobs", payload)


def upload(server, files):
    return http_post_json(server + "/api/uploads", {"files": files})


def b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


# ----------------------------------------------------------------------
# Jobs
# ----------------------------------------------------------------------
def test_long_text_job_completes(server):
    payload = {
        "type": "long_text",
        "engine": "voiceforge",
        "items": [{"name": "Long", "text": "Hello there. " * 30}],
    }

    with submit(server, payload) as response:
        created = json.loads(response.read())

    assert response.status == 202
    assert created["ok"] is True
    assert created["job"]["status"] in ("queued", "running")

    job = wait_for_job(server, created["job"]["id"])

    assert job["status"] == "completed"
    assert job["completed"] == 1
    assert job["failed"] == 0

    item = job["items"][0]
    assert item["status"] == "completed"
    assert item["filename"].endswith(".wav")
    assert item["chunks_total"] > 1
    assert item["chunks_completed"] == item["chunks_total"]
    assert "/audio/" in item["download_url"]


def test_batch_job_with_two_items(server):
    payload = {
        "type": "batch",
        "engine": "voiceforge",
        "items": [
            {"name": "First", "text": "Hello from the first item."},
            {"name": "Second", "text": "And hello from the second."},
        ],
    }

    with submit(server, payload) as response:
        job_id = json.loads(response.read())["job"]["id"]

    job = wait_for_job(server, job_id)

    assert job["status"] == "completed"
    assert job["total"] == 2
    assert job["completed"] == 2


def test_item_audio_is_downloadable(server):
    payload = {
        "type": "batch",
        "engine": "voiceforge",
        "items": [{"name": "One", "text": "Hello world."}],
    }

    with submit(server, payload) as response:
        job_id = json.loads(response.read())["job"]["id"]

    job = wait_for_job(server, job_id)

    with http_get(server + job["items"][0]["download_url"]) as response:
        audio = response.read()

    assert response.status == 200
    assert response.headers["Content-Type"] == "audio/wav"
    assert audio[:4] == b"RIFF"


def test_partial_failure_keeps_successful_items(server):
    payload = {
        "type": "batch",
        "engine": "voiceforge",
        "items": [
            {"name": "Good", "text": "This item is fine."},
            {"name": "Bad", "text": "This item has digits 123."},
        ],
    }

    with submit(server, payload) as response:
        job_id = json.loads(response.read())["job"]["id"]

    job = wait_for_job(server, job_id)

    assert job["status"] == "completed"
    assert job["completed"] == 1
    assert job["failed"] == 1
    assert [item["status"] for item in job["items"]] == ["completed", "failed"]
    assert job["items"][1]["error"]


def test_all_items_failing_marks_job_failed(server):
    payload = {
        "type": "long_text",
        "engine": "voiceforge",
        "items": [{"text": "Digits 123 are not supported."}],
    }

    with submit(server, payload) as response:
        job_id = json.loads(response.read())["job"]["id"]

    job = wait_for_job(server, job_id)

    assert job["status"] == "failed"
    assert job["completed"] == 0
    assert job["failed"] == 1
    assert job["download_url"] is None


def test_download_without_results_returns_409(server):
    payload = {
        "type": "long_text",
        "engine": "voiceforge",
        "items": [{"text": "Bad 123 text."}],
    }

    with submit(server, payload) as response:
        job_id = json.loads(response.read())["job"]["id"]

    wait_for_job(server, job_id)

    with pytest.raises(urllib.error.HTTPError) as error:
        http_get(f"{server}/api/jobs/{job_id}/download")

    payload = error_payload(error.value)

    assert error.value.code == 409
    assert payload["code"] == "no_results"


def test_unknown_job_returns_404(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        http_get(server + "/api/jobs/doesnotexist")

    payload = error_payload(error.value)

    assert error.value.code == 404
    assert payload["code"] == "job_not_found"


@pytest.mark.parametrize(
    "payload,code",
    [
        ({"type": "nope", "items": [{"text": "Hi."}]}, "invalid_job_type"),
        ({"type": "batch", "items": []}, "invalid_job"),
        ({"type": "batch", "items": [{"text": "   "}]}, "empty_text"),
        (
            {"type": "batch", "items": [{"text": "a" * 20001}]},
            "text_too_long",
        ),
        (
            {"type": "batch", "items": [{"text": "Hi."}], "engine": "nope"},
            "invalid_engine",
        ),
        (
            {
                "type": "batch",
                "engine": "kokoro",
                "voice": "nope",
                "items": [{"text": "Hi."}],
            },
            "invalid_voice",
        ),
        (
            {
                "type": "batch",
                "engine": "kokoro",
                "speed": 5,
                "items": [{"text": "Hi."}],
            },
            "invalid_speed",
        ),
        (
            {
                "type": "batch",
                "engine": "voiceforge",
                "speed": 1.2,
                "items": [{"text": "Hi."}],
            },
            "speed_not_supported",
        ),
    ],
)
def test_invalid_job_submissions_are_rejected(server, payload, code):
    with pytest.raises(urllib.error.HTTPError) as error:
        submit(server, payload)

    body = error_payload(error.value)

    assert error.value.code == 400
    assert body["code"] == code


def test_too_many_items_are_rejected(server):
    payload = {
        "type": "batch",
        "engine": "voiceforge",
        "items": [{"text": f"Item {index}."} for index in range(21)],
    }

    with pytest.raises(urllib.error.HTTPError) as error:
        submit(server, payload)

    assert error_payload(error.value)["code"] == "too_many_items"


def test_queue_limit_rejects_extra_jobs(tmp_path):
    class BlockingApp:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        def resolve_engine(self, engine):
            return "voiceforge"

        def synthesize(self, *args, **kwargs):
            self.started.set()
            self.release.wait(5)
            raise AppError("blocked")

    app = BlockingApp()
    manager = JobManager(app, job_root=tmp_path / "jobs", max_jobs=2)

    payload = {"type": "batch", "items": [{"text": "Hello."}]}

    try:
        manager.submit(payload)
        assert app.started.wait(2), "worker did not start"

        manager.submit(payload)  # one running + one queued = limit

        with pytest.raises(JobQueueFull):
            manager.submit(payload)
    finally:
        app.release.set()
        manager.stop()


def test_stop_fails_queued_jobs(tmp_path):
    class BlockingApp:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        def resolve_engine(self, engine):
            return "voiceforge"

        def synthesize(self, *args, **kwargs):
            self.started.set()
            self.release.wait(5)
            raise AppError("blocked")

    app = BlockingApp()
    manager = JobManager(app, job_root=tmp_path / "jobs", max_jobs=3)

    payload = {"type": "batch", "items": [{"text": "Hello."}]}

    first = manager.submit(payload)
    assert app.started.wait(2)

    second = manager.submit(payload)
    third = manager.submit(payload)

    app.release.set()
    manager.stop()

    assert second.status in ("failed", "completed")
    assert third.status in ("failed", "completed")


# ----------------------------------------------------------------------
# ZIP
# ----------------------------------------------------------------------
def test_zip_contains_audio_and_manifest(server):
    payload = {
        "type": "batch",
        "engine": "voiceforge",
        "items": [
            {"name": "Alpha", "text": "First item text."},
            {"name": "Beta", "text": "Second item text."},
        ],
    }

    with submit(server, payload) as response:
        job_id = json.loads(response.read())["job"]["id"]

    wait_for_job(server, job_id)

    with http_get(f"{server}/api/jobs/{job_id}/download") as response:
        archive = response.read()

    assert response.status == 200
    assert response.headers["Content-Type"] == "application/zip"

    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        names = bundle.namelist()
        manifest = json.loads(bundle.read("manifest.json"))

    assert sum(name.endswith(".wav") for name in names) == 2
    assert "manifest.json" in names
    assert manifest["completed"] == 2
    assert manifest["failed"] == 0
    assert len(manifest["items"]) == 2


def test_zip_manifest_records_failures(server):
    payload = {
        "type": "batch",
        "engine": "voiceforge",
        "items": [
            {"name": "Ok", "text": "Fine text."},
            {"name": "Bad", "text": "Digits 42 here."},
        ],
    }

    with submit(server, payload) as response:
        job_id = json.loads(response.read())["job"]["id"]

    wait_for_job(server, job_id)

    with http_get(f"{server}/api/jobs/{job_id}/download") as response:
        archive = response.read()

    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))

    assert manifest["completed"] == 1
    assert manifest["failed"] == 1
    assert manifest["items"][1]["status"] == "failed"
    assert manifest["items"][1]["error"]


def test_zip_entry_names_are_safe(server):
    payload = {
        "type": "batch",
        "engine": "voiceforge",
        "items": [
            {"name": "../../etc/passwd", "text": "Hello there."},
            {"name": "..\\..\\windows", "text": "Second text."},
        ],
    }

    with submit(server, payload) as response:
        job_id = json.loads(response.read())["job"]["id"]

    wait_for_job(server, job_id)

    with http_get(f"{server}/api/jobs/{job_id}/download") as response:
        archive = response.read()

    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        names = bundle.namelist()

    for name in names:
        assert ".." not in name
        assert not name.startswith("/")
        assert "\\" not in name


# ----------------------------------------------------------------------
# Uploads
# ----------------------------------------------------------------------
def test_upload_accepts_txt_and_strips_bom(server):
    content = "\ufeffHello from a file."

    with upload(
        server,
        [{"name": "notes.txt", "data_base64": b64(content)}],
    ) as response:
        payload = json.loads(response.read())

    assert response.status == 200
    assert payload["count"] == 1
    assert payload["files"][0]["name"] == "notes.txt"
    assert payload["files"][0]["text"] == "Hello from a file."
    assert payload["files"][0]["characters"] == len("Hello from a file.")


def test_upload_rejects_non_txt(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        upload(server, [{"name": "notes.md", "data_base64": b64("Hello")}])

    payload = error_payload(error.value)

    assert error.value.code == 400
    assert payload["code"] == "unsupported_file"


def test_upload_rejects_invalid_base64(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        upload(server, [{"name": "a.txt", "data_base64": "!!!notbase64!!!"}])

    assert error_payload(error.value)["code"] == "invalid_upload"


def test_upload_rejects_invalid_utf8(server):
    raw = b"\xff\xfe\x00\x41"
    encoded = base64.b64encode(raw).decode("ascii")

    with pytest.raises(urllib.error.HTTPError) as error:
        upload(server, [{"name": "bad.txt", "data_base64": encoded}])

    assert error_payload(error.value)["code"] == "invalid_encoding"


def test_upload_rejects_too_many_files(server):
    files = [
        {"name": f"f{index}.txt", "data_base64": b64("Hello.")}
        for index in range(21)
    ]

    with pytest.raises(urllib.error.HTTPError) as error:
        upload(server, files)

    assert error_payload(error.value)["code"] == "upload_too_large"


def test_upload_rejects_oversized_file(server):
    big = "a" * (1024 * 1024 + 10)

    with pytest.raises(urllib.error.HTTPError) as error:
        upload(server, [{"name": "big.txt", "data_base64": b64(big)}])

    assert error_payload(error.value)["code"] == "upload_too_large"


def test_upload_sanitizes_paths(server):
    with upload(
        server,
        [{"name": "../../secret.txt", "data_base64": b64("Hello.")}],
    ) as response:
        payload = json.loads(response.read())

    assert payload["files"][0]["name"] == "secret.txt"


def test_upload_rejects_empty_content(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        upload(server, [{"name": "empty.txt", "data_base64": ""}])

    assert error_payload(error.value)["code"] == "invalid_upload"


# ----------------------------------------------------------------------
# Backwards compatibility
# ----------------------------------------------------------------------
def test_legacy_generate_still_works(server):
    with http_post_json(
        server + "/generate",
        {"text": "Hello world.", "engine": "voiceforge"},
    ) as response:
        payload = json.loads(response.read())

    assert response.status == 200
    assert payload["ok"] is True
