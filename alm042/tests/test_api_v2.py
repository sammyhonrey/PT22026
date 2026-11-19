"""Phase 2 API tests: /api/engines, /api/generate, speed and errors."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from conftest import error_payload, http_get, http_post_json


def test_api_engines_lists_both_engines(server):
    with http_get(server + "/api/engines") as response:
        payload = json.loads(response.read())

    assert response.status == 200
    assert payload["ok"] is True
    assert payload["limits"]["max_text_length"] == 400
    assert set(payload["engines"]) == {"kokoro", "voiceforge"}
    assert payload["default"] in {"kokoro", "voiceforge"}


def test_api_engines_kokoro_capabilities(server):
    with http_get(server + "/api/engines") as response:
        payload = json.loads(response.read())

    kokoro = payload["engines"]["kokoro"]

    assert kokoro["id"] == "kokoro"
    assert kokoro["name"]
    assert isinstance(kokoro["installed"], bool)
    assert isinstance(kokoro["loaded"], bool)
    assert kokoro["language"] == "en-US"
    assert kokoro["sample_rate"] == 24000
    assert kokoro["speed"] == {
        "supported": True,
        "min": 0.5,
        "max": 2.0,
        "default": 1.0,
    }
    assert len(kokoro["voices"]) >= 1
    assert {"id", "name", "default"} <= set(kokoro["voices"][0])
    assert "speed-control" in kokoro["capabilities"]


def test_api_engines_voiceforge_does_not_claim_speed(server):
    with http_get(server + "/api/engines") as response:
        payload = json.loads(response.read())

    voiceforge = payload["engines"]["voiceforge"]

    assert voiceforge["speed"]["supported"] is False
    assert voiceforge["speed"]["min"] is None
    assert voiceforge["speed"]["max"] is None
    assert "speed-control" not in voiceforge["capabilities"]
    assert voiceforge["sample_rate"] == 22050


def test_api_generate_with_custom_engine(server):
    with http_post_json(
        server + "/api/generate",
        {"text": "Hello world.", "engine": "voiceforge"},
    ) as response:
        payload = json.loads(response.read())

    assert response.status == 200
    assert payload["ok"] is True
    assert payload["engine"] == "voiceforge"
    assert payload["voice"] is None
    assert payload["speed"] is None
    assert payload["sample_rate"] == 22050
    assert payload["text"] == "Hello world."
    assert payload["audio_url"].startswith("/audio/")


def test_api_generate_audio_downloads(server):
    with http_post_json(
        server + "/api/generate",
        {"text": "Hello world.", "engine": "voiceforge"},
    ) as response:
        payload = json.loads(response.read())

    with http_get(server + payload["audio_url"]) as response:
        audio = response.read()

    assert response.status == 200
    assert response.headers["Content-Type"] == "audio/wav"
    assert audio[:4] == b"RIFF"


@pytest.mark.parametrize("speed", [0.4, 0.49, 2.01, 3, -1])
def test_speed_out_of_range_is_rejected(server, speed):
    with pytest.raises(urllib.error.HTTPError) as error:
        http_post_json(
            server + "/api/generate",
            {"text": "Hello.", "engine": "kokoro", "speed": speed},
        )

    payload = error_payload(error.value)

    assert error.value.code == 400
    assert payload["code"] == "invalid_speed"
    assert "between 0.5 and 2.0" in payload["error"]


@pytest.mark.parametrize("speed", ["fast", [], {}, True])
def test_speed_with_wrong_type_is_rejected(server, speed):
    with pytest.raises(urllib.error.HTTPError) as error:
        http_post_json(
            server + "/api/generate",
            {"text": "Hello.", "engine": "kokoro", "speed": speed},
        )

    payload = error_payload(error.value)

    assert error.value.code == 400
    assert payload["code"] == "invalid_speed"


def test_speed_is_rejected_for_the_custom_engine(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        http_post_json(
            server + "/api/generate",
            {"text": "Hello.", "engine": "voiceforge", "speed": 1.5},
        )

    payload = error_payload(error.value)

    assert error.value.code == 400
    assert payload["code"] == "speed_not_supported"
    assert "kokoro" in payload["error"].lower()


def test_invalid_engine_has_a_code(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        http_post_json(server + "/api/generate", {"text": "Hello.", "engine": "nope"})

    payload = error_payload(error.value)

    assert error.value.code == 400
    assert payload["code"] == "invalid_engine"


def test_invalid_voice_has_a_code(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        http_post_json(
            server + "/api/generate",
            {"text": "Hello.", "engine": "kokoro", "voice": "not_a_voice"},
        )

    payload = error_payload(error.value)

    assert error.value.code == 400
    assert payload["code"] == "invalid_voice"


def test_empty_text_has_a_code(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        http_post_json(server + "/api/generate", {"text": "  ", "engine": "voiceforge"})

    payload = error_payload(error.value)

    assert error.value.code == 400
    assert payload["code"] == "empty_text"


def test_malformed_json_has_a_code(server):
    request = urllib.request.Request(
        server + "/api/generate",
        data=b"{not json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=30)

    payload = json.loads(error.value.read())

    assert error.value.code == 400
    assert payload["code"] == "bad_request"


def test_missing_audio_has_a_code(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        http_get(server + "/audio/missing.wav")

    payload = json.loads(error.value.read())

    assert error.value.code == 404
    assert payload["code"] == "audio_not_found"


def test_old_generate_endpoint_still_works(server):
    with http_post_json(
        server + "/generate",
        {"text": "Hello world.", "engine": "voiceforge"},
    ) as response:
        payload = json.loads(response.read())

    assert response.status == 200
    assert payload["ok"] is True
    assert payload["engine"] == "voiceforge"


def test_unexpected_error_returns_json_500(server, app, monkeypatch):
    secret = "C:\\secret\\internal\\path.py"

    def explode(*args, **kwargs):
        raise RuntimeError(f"boom at {secret}")

    monkeypatch.setattr(app, "synthesize", explode)

    with pytest.raises(urllib.error.HTTPError) as error:
        http_post_json(server + "/api/generate", {"text": "Hello."})

    raw = error.value.read().decode("utf-8")
    payload = json.loads(raw)

    assert error.value.code == 500
    assert payload["ok"] is False
    assert payload["code"] == "internal_error"
    assert "internal server" in payload["error"].lower() or "could not complete" in payload["error"].lower()
    # Nothing internal may leak to the client.
    assert "secret" not in raw
    assert "traceback" not in raw.lower()
    assert ".py" not in raw
