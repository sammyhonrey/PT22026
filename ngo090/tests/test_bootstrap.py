"""The startup fetch: what it downloads, from where, and what it refuses.

Every test here talks to a throwaway HTTP server on localhost, so the real
Hugging Face and GitHub links are never touched -- but the code path (URL
shaping, hash check, fallback) is the one that runs in a deployment.
"""
from __future__ import annotations

import hashlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))             # bootstrap is an app/ module, importable here
import bootstrap as bs  # noqa: E402


class _Handler(BaseHTTPRequestHandler):
    files: dict[str, bytes] = {}
    hits: list[str] = []
    ranges: list[str | None] = []
    truncate: set[str] = set()          # paths whose first response is cut in half
    ignore_range: bool = False          # a host that does not implement Range at all

    def do_GET(self):                                    # noqa: N802 (http.server API)
        _Handler.hits.append(self.path)
        _Handler.ranges.append(self.headers.get("Range"))
        body = _Handler.files.get(self.path)
        if body is None:
            self.send_error(404)
            return
        wanted = self.headers.get("Range")
        if wanted and wanted.startswith("bytes=") and not _Handler.ignore_range:
            start = int(wanted.split("=", 1)[1].split("-", 1)[0])
            rest = body[start:]
            self.send_response(206)
            self.send_header("Content-Length", str(len(rest)))
            self.end_headers()
            self.wfile.write(rest)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.path in _Handler.truncate:                # a CDN hanging up mid-stream
            _Handler.truncate.discard(self.path)
            self.wfile.write(body[:len(body) // 2])
            self.close_connection = True
            return
        self.wfile.write(body)

    def log_message(self, *args):                         # keep the test output clean
        pass


@pytest.fixture
def server():
    """A dict-like {path: bytes} served on localhost, plus the base URL."""
    _Handler.files, _Handler.hits, _Handler.ranges, _Handler.truncate = {}, [], [], set()
    _Handler.ignore_range = False
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield _Handler, f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()


def _artifact(path: str, payload: bytes) -> bs.Artifact:
    return bs.Artifact(path, len(payload), hashlib.sha256(payload).hexdigest())


def test_missing_reports_only_absent_or_wrong_sized_files(tmp_path):
    good, bad = b"a" * 32, b"b" * 32
    one, two = _artifact("models/t/one.joblib", good), _artifact("models/t/two.joblib", bad)
    (tmp_path / "models" / "t").mkdir(parents=True)
    (tmp_path / "models" / "t" / "one.joblib").write_bytes(good)
    (tmp_path / "models" / "t" / "two.joblib").write_bytes(b"c" * 31)   # truncated

    assert bs.missing(tmp_path, (one, two)) == [two]


def test_urls_follow_each_hosts_convention():
    artifact = _artifact("models/tag/xgboost.joblib", b"x")
    kept = bs.Source("HF", "https://huggingface.co/spaces/u/s/resolve/main", keep_dirs=True)
    flat = bs.Source("GH", "https://github.com/u/r/releases/download/artifacts-v1", keep_dirs=False)

    assert kept.url_for(artifact) == "https://huggingface.co/spaces/u/s/resolve/main/models/tag/xgboost.joblib"
    assert flat.url_for(artifact) == "https://github.com/u/r/releases/download/artifacts-v1/xgboost.joblib"
    trailing = bs.Source("HF", "https://host/base/").url_for(artifact)
    assert "//" not in trailing.replace("https://", "")          # a trailing slash is not doubled


def test_ensure_downloads_every_missing_file_from_the_first_source(tmp_path, server):
    handler, base = server
    payloads = {"models/t/x.joblib": b"xgboost-bytes" * 10, "results/t/base.npz": b"npz" * 50}
    artifacts = [_artifact(path, body) for path, body in payloads.items()]
    handler.files.update({f"/{path}": body for path, body in payloads.items()})

    notes = bs.ensure(tmp_path, artifacts, (bs.Source("Hugging Face", base),), pause=0)

    assert len(notes) == 2 and all("Hugging Face" in note for note in notes)
    for path, body in payloads.items():
        assert (tmp_path / path).read_bytes() == body
    assert not list(tmp_path.rglob("*.part"))            # nothing half-written is left behind
    assert "2 trained files" in bs.summary(notes)


def test_a_corrupt_download_is_rejected_and_the_next_source_is_used(tmp_path, server):
    handler, base = server
    payload = b"the real model" * 40
    artifact = _artifact("models/t/xgboost.joblib", payload)
    # First host answers 200 with the wrong bytes: same length, different content.
    handler.files["/models/t/xgboost.joblib"] = b"X" * len(payload)

    sources = (bs.Source("Hugging Face", base),                    # keeps the folders
               bs.Source("GitHub release", f"{base}/flat", keep_dirs=False))
    handler.files["/flat/xgboost.joblib"] = payload                # the fallback is honest

    notes = bs.ensure(tmp_path, [artifact], sources, pause=0)

    assert (tmp_path / artifact.path).read_bytes() == payload
    assert notes == ["xgboost.joblib from GitHub release"]
    assert any(path.startswith("/models/") for path in handler.hits)   # it did try HF first


def test_a_dropped_stream_resumes_where_it_stopped(tmp_path, server):
    handler, base = server
    payload = bytes(range(256)) * 200                     # 51,200 bytes, easy to compare
    artifact = _artifact("models/t/xgboost.joblib", payload)
    handler.files["/models/t/xgboost.joblib"] = payload
    handler.truncate.add("/models/t/xgboost.joblib")      # first response is cut in half

    notes = bs.ensure(tmp_path, [artifact], (bs.Source("Hugging Face", base),), pause=0)

    assert (tmp_path / artifact.path).read_bytes() == payload
    assert notes == ["xgboost.joblib from Hugging Face"]
    assert len(handler.hits) == 2                          # one cut, one resume -- not a re-download
    assert handler.ranges[0] is None
    assert handler.ranges[1] == f"bytes={len(payload) // 2}-"


def test_a_server_that_ignores_range_restarts_the_file_cleanly(tmp_path, server):
    handler, base = server
    payload = b"payload-" * 400
    artifact = _artifact("models/t/x.joblib", payload)
    handler.files["/models/t/x.joblib"] = payload
    handler.truncate.add("/models/t/x.joblib")
    handler.ignore_range = True                            # always answers 200 with the whole file

    notes = bs.ensure(tmp_path, [artifact], (bs.Source("Hugging Face", base),), pause=0)

    assert (tmp_path / artifact.path).read_bytes() == payload   # not payload + half of payload
    assert len(notes) == 1


def test_every_source_failing_raises_with_both_links_in_the_message(tmp_path, server):
    _, base = server                                     # server serves nothing: 404 everywhere
    artifact = _artifact("models/t/xgboost.joblib", b"payload")
    sources = (bs.Source("Hugging Face", base), bs.Source("GitHub release", f"{base}/flat", keep_dirs=False))

    with pytest.raises(bs.ArtifactsUnavailable) as info:
        bs.ensure(tmp_path, [artifact], sources, pause=0)

    message = str(info.value)
    assert "Hugging Face" in message and "GitHub release" in message
    assert "train" in message.lower()                    # tells the reader how to rebuild locally
    assert not (tmp_path / artifact.path).exists()


def test_nothing_is_requested_when_the_files_are_already_there(tmp_path, server):
    handler, base = server
    payload = b"already here"
    artifact = _artifact("models/t/x.joblib", payload)
    (tmp_path / "models" / "t").mkdir(parents=True)
    (tmp_path / "models" / "t" / "x.joblib").write_bytes(payload)

    assert bs.ensure(tmp_path, [artifact], (bs.Source("Hugging Face", base),), pause=0) == []
    assert handler.hits == []                            # a warm start touches the network zero times


def test_the_environment_variable_is_tried_before_the_built_in_sources(tmp_path, server, monkeypatch):
    handler, base = server
    payload = b"override"
    artifact = _artifact("models/t/x.joblib", payload)
    handler.files["/models/t/x.joblib"] = payload
    monkeypatch.setenv("FRAUDINTEL_ARTIFACTS_URL", base)

    notes = bs.ensure(tmp_path, [artifact], (), pause=0)   # no built-in source offered at all

    assert notes == ["x.joblib from FRAUDINTEL_ARTIFACTS_URL"]


def test_the_manifest_covers_exactly_the_run_the_app_loads():
    from src.artifacts import PREFERRED_RUN
    from src.scoring import MODEL_NAMES

    expected = {f"models/{PREFERRED_RUN}/{name}.joblib" for name in (*MODEL_NAMES, "stacker")}
    expected.add(f"results/{PREFERRED_RUN}/baseline_scores.npz")
    assert {a.path for a in bs.ARTIFACTS} == expected
