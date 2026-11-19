"""Authentication flow integration tests (T6.3/T6.4 acceptance criteria).

These run against the real (dev) database and commit real users, so every test
uses a unique email to avoid cross-test collisions.
"""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _email(prefix: str = "auth") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}@example.com"


def _register(email: str | None = None, password: str = "password123", **extra):
    return client.post(
        "/api/v1/auth/register",
        json={"email": email or _email(), "password": password, **extra},
    )


def _login(email: str, password: str = "password123"):
    return client.post("/api/v1/auth/login", json={"email": email, "password": password})


def _auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_register_login_me_flow():
    email = _email()
    r = _register(email)
    assert r.status_code == 201
    body = r.json()
    assert body["access_token"]
    assert body["token_type"] == "bearer"
    assert body["user"]["email"] == email
    assert "password" not in body["user"]

    token = body["access_token"]
    me = client.get("/api/v1/users/me", headers=_auth_header(token))
    assert me.status_code == 200
    assert me.json()["email"] == email

    login = _login(email)
    assert login.status_code == 200
    assert login.json()["access_token"]


def test_duplicate_email_conflict():
    email = _email("dup")
    assert _register(email).status_code == 201
    assert _register(email).status_code == 409


def test_login_unknown_email_and_wrong_password_same_message():
    email = _email()
    _register(email)
    wrong_pw = _login(email, "not-the-password")
    unknown = _login(_email("nobody"), "whatever123")
    assert wrong_pw.status_code == unknown.status_code == 401
    assert wrong_pw.json()["detail"] == unknown.json()["detail"]


def test_me_requires_auth():
    assert client.get("/api/v1/users/me").status_code == 401
    assert client.get("/api/v1/users/me", headers=_auth_header("garbage")).status_code == 401


def test_email_normalized_to_lowercase():
    email = _email("MixedCase")
    _register(email.upper())
    login = _login(email.lower())
    assert login.status_code == 200


def test_patch_me_updates_display_name():
    token = _register().json()["access_token"]
    r = client.patch(
        "/api/v1/users/me",
        json={"display_name": "Alice"},
        headers=_auth_header(token),
    )
    assert r.status_code == 200
    assert r.json()["display_name"] == "Alice"


def test_refresh_and_logout_rotate_and_revoke_session():
    registered = _register()
    assert registered.status_code == 201
    csrf_token = registered.json()["csrf_token"]

    refreshed = client.post(
        "/api/v1/auth/refresh",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert refreshed.status_code == 200
    assert refreshed.json()["csrf_token"] != csrf_token

    logout = client.post(
        "/api/v1/auth/logout",
        headers={"X-CSRF-Token": refreshed.json()["csrf_token"]},
    )
    assert logout.status_code == 204
    assert client.post("/api/v1/auth/refresh").status_code == 401


def test_short_password_rejected():
    r = _register(_email("short"), "short")
    assert r.status_code == 422
