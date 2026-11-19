"""Shared FastAPI dependencies."""

from __future__ import annotations

import uuid
from typing import Annotated

import jwt
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import TooManyRequestsApiError, UnauthorizedApiError
from app.core.rate_limit import SlidingWindowRateLimiter
from app.core.security import decode_access_token
from app.db.models.user import User
from app.db.session import get_db
from app.repositories import users as user_repo

# Request-scoped DB session, injected via Annotated to satisfy linters.
DbSession = Annotated[Session, Depends(get_db)]

_bearer = HTTPBearer(auto_error=False)
Credentials = Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)]


def get_current_user(db: DbSession, credentials: Credentials) -> User:
    """Resolve the authenticated user strictly from the bearer token.

    Never trusts a client-supplied user ID — identity comes only from the
    validated JWT ``sub`` claim.
    """
    if credentials is None:
        raise UnauthorizedApiError("Not authenticated")

    token = credentials.credentials
    try:
        payload = decode_access_token(token)
    except jwt.ExpiredSignatureError:
        raise UnauthorizedApiError("Token expired") from None
    except jwt.InvalidTokenError:
        raise UnauthorizedApiError("Invalid token") from None

    if payload.get("type") != "access":
        raise UnauthorizedApiError("Invalid token type")

    raw_sub = payload.get("sub")
    if not raw_sub:
        raise UnauthorizedApiError("Invalid token")
    try:
        user_id = uuid.UUID(raw_sub)
    except (ValueError, TypeError):
        raise UnauthorizedApiError("Invalid token") from None

    user = user_repo.get_by_id(db, user_id)
    if user is None or not user.is_active:
        raise UnauthorizedApiError("User not found or inactive")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


# --- Rate limiting (T9.2) ----------------------------------------------------
# In-process sliding-window limiters keyed by client IP. Limits are documented
# in the Settings defaults (requests per minute).
_settings = get_settings()
_login_limiter = SlidingWindowRateLimiter(_settings.RATE_LIMIT_LOGIN, window_seconds=60)
_register_limiter = SlidingWindowRateLimiter(_settings.RATE_LIMIT_REGISTER, window_seconds=60)
_demo_limiter = SlidingWindowRateLimiter(_settings.RATE_LIMIT_DEMO, window_seconds=60)
_ALL_LIMITERS = (_login_limiter, _register_limiter, _demo_limiter)


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def rate_limit_login(request: Request) -> None:
    if not _login_limiter.allow(f"login:{_client_key(request)}"):
        raise TooManyRequestsApiError("Too many login attempts. Please try again later.")


def rate_limit_register(request: Request) -> None:
    if not _register_limiter.allow(f"register:{_client_key(request)}"):
        raise TooManyRequestsApiError("Too many account registrations. Please try again later.")


def rate_limit_demo(request: Request) -> None:
    if not _demo_limiter.allow(f"demo:{_client_key(request)}"):
        raise TooManyRequestsApiError("Demo rate limit exceeded. Please try again later.")


def reset_rate_limiters() -> None:
    """Clear all in-process rate-limit state (used by the test suite)."""
    for limiter in _ALL_LIMITERS:
        limiter.reset()
