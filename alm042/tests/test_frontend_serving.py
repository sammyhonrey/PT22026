"""Tests for the separated frontend: static serving, MIME types, safety."""

from __future__ import annotations

import re

import pytest

from conftest import http_get


def test_index_is_served(server):
    with http_get(server + "/") as response:
        body = response.read().decode("utf-8")

    assert response.status == 200
    assert response.headers["Content-Type"].startswith("text/html")
    assert "VoiceForge" in body
    assert 'id="page-root"' in body
    assert 'type="module" src="/js/app.js"' in body


def test_stylesheets_are_served(server):
    for name in ("style.css", "layout.css", "components.css"):
        with http_get(f"{server}/css/{name}") as response:
            body = response.read().decode("utf-8")

        assert response.status == 200, name
        assert response.headers["Content-Type"].startswith("text/css"), name
        assert "{" in body, name


def test_javascript_modules_are_served(server):
    for path in ("/js/app.js", "/js/api.js", "/js/pages/studio.js"):
        with http_get(server + path) as response:
            body = response.read().decode("utf-8")

        assert response.status == 200, path
        assert response.headers["Content-Type"].startswith("text/javascript"), path
        assert "export" in body or "import" in body, path


def test_favicon_is_served(server):
    with http_get(server + "/assets/favicon.svg") as response:
        body = response.read().decode("utf-8")

    assert response.status == 200
    assert response.headers["Content-Type"] == "image/svg+xml"
    assert "<svg" in body


def test_every_asset_referenced_by_index_is_served(server):
    with http_get(server + "/") as response:
        html = response.read().decode("utf-8")

    paths = set(re.findall(r'(?:href|src)="(/[^"]+)"', html))
    paths = {path for path in paths if path.startswith(("/css/", "/js/", "/assets/"))}

    assert paths, "index.html should reference static assets"

    for path in sorted(paths):
        with http_get(server + path) as response:
            assert response.status == 200, path


@pytest.mark.parametrize(
    "path",
    [
        "/css/../app.py",
        "/js/../../app.py",
        "/css/%2e%2e/app.py",
        "/assets/../index.html",
        "/../alm042/app.py",
    ],
)
def test_path_traversal_is_rejected(server, path):
    with pytest.raises(Exception) as error:
        http_get(server + path)

    assert getattr(error.value, "code", 404) == 404


def test_unknown_asset_returns_404(server):
    with pytest.raises(Exception) as error:
        http_get(server + "/css/missing.css")

    assert getattr(error.value, "code", 404) == 404


def test_non_static_paths_are_not_served(server):
    # Project files outside frontend/ must never be reachable.
    for path in ("/voiceforge/app_core.py", "/app.py", "/package.json"):
        with pytest.raises(Exception) as error:
            http_get(server + path)

        assert getattr(error.value, "code", 404) == 404


def test_health_reports_limits_and_engine_metadata(server):
    import json

    with http_get(server + "/health") as response:
        payload = json.loads(response.read())

    assert response.status == 200
    assert payload["ok"] is True
    assert payload["limits"]["max_text_length"] == 400

    engines = payload["engines"]
    assert set(engines["labels"]) == {"kokoro", "voiceforge"}
    assert len(engines["kokoro"]["voices"]) >= 1
    assert "female" in engines["kokoro"]["voice_labels"]["af_heart"]
    assert engines["default"] in {"kokoro", "voiceforge"}
