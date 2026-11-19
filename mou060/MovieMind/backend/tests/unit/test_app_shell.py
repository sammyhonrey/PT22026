"""App-level tests for T2.2/T2.3/T2.4 acceptance criteria.

These tests use the FastAPI TestClient. The database check inside /ready is
exercised as-is: with no PostgreSQL running, /ready must still return a
controlled 503 (not a crash) — that is part of the contract.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_service_identity_root():
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["app"] == "MovieMind API"
    assert "env" in body


def test_health_liveness_no_secrets():
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    text = r.text.lower()
    for forbidden in ("secret", "password", "c:\\", "/home/"):
        assert forbidden not in text


def test_ready_reports_checks_and_is_controlled():
    r = client.get("/api/v1/ready")
    body = r.json()
    assert "checks" in body
    assert "database" in body["checks"]
    assert "model" in body["checks"]
    # Model is now really loaded (Phase 3); DB status depends on connectivity.
    assert body["checks"]["model"]["status"] == "loaded"
    assert body["checks"]["database"]["status"] in ("ok", "unavailable")
    # Controlled: 200 when both ready, 503 otherwise; never a 500 crash.
    assert r.status_code in (200, 503)


def test_request_id_header_present():
    r = client.get("/api/v1/health")
    assert r.headers.get("X-Request-ID")


def test_cors_preflight_allows_put_requests():
    r = client.options(
        "/api/v1/preferences/me",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "PUT",
            "Access-Control-Request-Headers": "authorization, content-type",
        },
    )
    assert r.status_code == 200
    assert "PUT" in r.headers.get("access-control-allow-methods", "")
    assert "authorization" in r.headers.get("access-control-allow-headers", "").lower()


def test_unknown_route_error_shape():
    r = client.get("/api/v1/does-not-exist")
    assert r.status_code == 404
    body = r.json()
    assert set(body.keys()) == {"detail", "code"}
    assert body["code"] == "http_error"


def test_openapi_generated_in_non_production():
    r = client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    assert "/api/v1/health" in schema["paths"]
    assert "/api/v1/ready" in schema["paths"]
