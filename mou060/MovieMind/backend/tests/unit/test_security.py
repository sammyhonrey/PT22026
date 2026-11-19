"""Unit tests for password hashing and JWT (T6.1/T6.2 acceptance criteria)."""

from __future__ import annotations

import uuid
from datetime import timedelta

import jwt
import pytest

from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


class TestPasswordHashing:
    def test_hash_and_verify(self):
        hashed = hash_password("correct horse battery staple")
        assert hashed != "correct horse battery staple"
        assert verify_password("correct horse battery staple", hashed)

    def test_wrong_password_rejected(self):
        hashed = hash_password("secret-password-123")
        assert not verify_password("wrong", hashed)

    def test_hashes_are_unique_per_call(self):
        assert hash_password("same") != hash_password("same")


class TestAccessToken:
    def test_roundtrip(self):
        user_id = uuid.uuid4()
        token = create_access_token(user_id)
        payload = decode_access_token(token)
        assert payload["sub"] == str(user_id)
        assert payload["type"] == "access"

    def test_expiry(self):
        user_id = uuid.uuid4()
        token = create_access_token(user_id, expires_delta=timedelta(seconds=-1))
        with pytest.raises(jwt.ExpiredSignatureError):
            decode_access_token(token)

    def test_tampered_token_rejected(self):
        token = create_access_token(uuid.uuid4())
        with pytest.raises(jwt.InvalidTokenError):
            decode_access_token(token + "x")

    def test_missing_claims_rejected(self):
        # A token without the required "type"/"sub" claims fails validation.
        import jwt as _jwt

        from app.core.config import get_settings

        settings = get_settings()
        bare = _jwt.encode({"foo": "bar"}, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
        with pytest.raises(_jwt.MissingRequiredClaimError):
            decode_access_token(bare)
