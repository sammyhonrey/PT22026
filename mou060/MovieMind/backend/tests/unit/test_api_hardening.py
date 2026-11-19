"""API hardening tests: CORS (T9.1) and body size limit (T9.3)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_cors_allowed_origin():
    r = client.get("/api/v1/health", headers={"Origin": "http://localhost:5173"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_disallowed_origin_rejected():
    r = client.get("/api/v1/health", headers={"Origin": "http://evil.example.com"})
    assert r.status_code == 200
    assert "access-control-allow-origin" not in r.headers


def test_cors_preflight():
    r = client.options(
        "/api/v1/auth/login",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-csrf-token",
        },
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "POST" in r.headers["access-control-allow-methods"]
    assert "x-csrf-token" in r.headers["access-control-allow-headers"].lower()


def test_body_limit_rejects_oversized_body():
    oversized = "x" * (1024 * 1024 + 10)  # > 1 MiB default
    r = client.post(
        "/api/v1/auth/register",
        content=oversized,
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 413


def test_normal_body_accepted():
    r = client.post(
        "/api/v1/auth/register",
        json={"email": "small@example.com", "password": "password123"},
    )
    # 201 (created) or 409/422 depending on state; just confirm it is not 413.
    assert r.status_code != 413
