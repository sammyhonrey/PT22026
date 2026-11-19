"""Rate-limiting integration test (T9.2 acceptance)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_repeated_login_attempts_are_throttled():
    payload = {"email": "nobody@example.com", "password": "whatever"}
    statuses = [client.post("/api/v1/auth/login", json=payload).status_code for _ in range(11)]

    # The first 10 are normal auth attempts; the 11th is rate-limited.
    assert 429 not in statuses[:10]
    assert statuses[10] == 429
