"""Password hashing and JWT access tokens.

- Passwords are hashed with Argon2id (``pwdlib`` + argon2-cffi) — never stored
  in plaintext, never reversible.
- Access tokens are short-lived, HS256-signed JWTs with ``sub`` = user UUID,
  ``type`` = "access", ``iat`` and ``exp``.   Refresh tokens are opaque random values stored only as hashes by the refresh
  session repository.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import jwt
from pwdlib import PasswordHash

from app.core.config import get_settings

_password_hasher = PasswordHash.recommended()


def hash_password(password: str) -> str:
    """Return an Argon2id hash of ``password``."""
    return _password_hasher.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    """Return True if ``password`` matches ``hashed`` (never raises)."""
    try:
        return _password_hasher.verify(password, hashed)
    except Exception:  # noqa: BLE001 — any hash error is treated as a mismatch
        return False


def create_access_token(user_id: uuid.UUID, expires_delta: timedelta | None = None) -> str:
    """Issue a signed access token for ``user_id``."""
    settings = get_settings()
    now = datetime.now(UTC)
    expires = now + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": expires,
        "type": "access",
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Decode and verify an access token; raises ``jwt`` errors on failure."""
    settings = get_settings()
    return jwt.decode(
        token,
        settings.SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM],
        options={"require": ["sub", "exp", "iat", "type"]},
    )
