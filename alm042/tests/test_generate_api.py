"""API tests: generation, validation, engine switching and download."""

from __future__ import annotations

import json
import urllib.error

import pytest

from conftest import error_payload, http_get, http_post_json


def post_generate(server, payload):
    return http_post_json(server + "/generate", payload)


def test_custom_engine_generation(server):
    with post_generate(server, {"text": "Hello world.", "engine": "voiceforge"}) as response:
        payload = json.loads(response.read())

    assert response.status == 200
    assert payload["ok"] is True
    assert payload["engine"] == "voiceforge"
    assert payload["sample_rate"] == 22050
    assert payload["duration_seconds"] > 0
    assert payload["audio_url"].startswith("/audio/")
    assert payload["filename"].endswith(".wav")


def test_default_engine_is_used_when_omitted(server):
    with post_generate(server, {"text": "Hello world."}) as response:
        payload = json.loads(response.read())

    assert response.status == 200
    assert payload["engine"] == "voiceforge"  # fixture is configured this way


def test_generated_audio_is_downloadable(server):
    with post_generate(server, {"text": "Hello world.", "engine": "voiceforge"}) as response:
        payload = json.loads(response.read())

    with http_get(server + payload["audio_url"]) as response:
        audio = response.read()

    assert response.status == 200
    assert response.headers["Content-Type"] == "audio/wav"
    assert "inline" in response.headers["Content-Disposition"]
    assert audio[:4] == b"RIFF"
    assert len(audio) > 1000


def test_generation_returns_unique_files(server):
    names = set()

    for _ in range(2):
        with post_generate(server, {"text": "Hello world.", "engine": "voiceforge"}) as response:
            names.add(json.loads(response.read())["filename"])

    assert len(names) == 2


@pytest.mark.parametrize(
    "text",
    ["", "   ", "\n\t "],
)
def test_empty_text_is_rejected(server, text):
    with pytest.raises(urllib.error.HTTPError) as error:
        post_generate(server, {"text": text, "engine": "voiceforge"})

    payload = error_payload(error.value)

    assert error.value.code == 400
    assert payload["ok"] is False
    assert "enter some" in payload["error"].lower()


def test_too_long_text_is_rejected(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        post_generate(server, {"text": "a" * 401, "engine": "voiceforge"})

    payload = error_payload(error.value)

    assert error.value.code == 400
    assert "too long" in payload["error"].lower()


def test_digits_are_rejected_by_the_custom_model(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        post_generate(server, {"text": "Room 204", "engine": "voiceforge"})

    payload = error_payload(error.value)

    assert error.value.code == 400
    assert "unsupported characters" in payload["error"].lower()


def test_unknown_engine_is_rejected(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        post_generate(server, {"text": "Hello.", "engine": "nope"})

    payload = error_payload(error.value)

    assert error.value.code == 400
    assert "unknown speech engine" in payload["error"].lower()


def test_non_json_body_is_rejected(server):
    import urllib.request

    request = urllib.request.Request(
        server + "/generate",
        data=b"not json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=15)

    assert error.value.code == 400


def test_missing_audio_returns_404(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        http_get(server + "/audio/does-not-exist.wav")

    assert error.value.code == 404


def test_audio_traversal_is_rejected(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        http_get(server + "/audio/../app.py")

    assert error.value.code == 404
