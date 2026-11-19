"""Authentication endpoints (register, login, refresh, logout)."""

from __future__ import annotations

import hashlib
import hmac
from datetime import timedelta

from fastapi import APIRouter, Depends, Header, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.api.dependencies import DbSession, rate_limit_login, rate_limit_register
from app.core.config import get_settings
from app.core.errors import ConflictApiError, ForbiddenApiError, UnauthorizedApiError
from app.core.security import create_access_token, hash_password, verify_password
from app.db.models.user import User
from app.repositories import refresh_sessions as refresh_session_repo
from app.repositories import users as user_repo
from app.schemas.auth import (
    CsrfTokenResponse,
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    RegisterResponse,
    TokenResponse,
)
from app.schemas.user import UserOut

router = APIRouter(prefix="/auth", tags=["auth"])

_REFRESH_COOKIE = "moviemind_refresh"
_CSRF_COOKIE = "moviemind_csrf"


def _cookie_path() -> str:
    prefix = get_settings().API_V1_PREFIX.rstrip("/")
    return f"{prefix or ''}/auth"


def _csrf_token(refresh_token: str) -> str:
    settings = get_settings()
    return hmac.new(
        settings.SECRET_KEY.encode("utf-8"),
        refresh_token.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _set_auth_cookies(response: Response, refresh_token: str) -> str:
    settings = get_settings()
    csrf_token = _csrf_token(refresh_token)
    cookie_options = {
        "path": _cookie_path(),
        "secure": settings.refresh_cookie_secure,
        "samesite": settings.REFRESH_COOKIE_SAME_SITE,
    }
    response.set_cookie(
        _REFRESH_COOKIE,
        refresh_token,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        httponly=True,
        **cookie_options,
    )
    response.set_cookie(
        _CSRF_COOKIE,
        csrf_token,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        httponly=False,
        **cookie_options,
    )
    response.headers["Cache-Control"] = "no-store"
    return csrf_token


def _clear_auth_cookies(response: Response) -> None:
    settings = get_settings()
    cookie_options = {
        "path": _cookie_path(),
        "secure": settings.refresh_cookie_secure,
        "samesite": settings.REFRESH_COOKIE_SAME_SITE,
    }
    response.delete_cookie(_REFRESH_COOKIE, httponly=True, **cookie_options)
    response.delete_cookie(_CSRF_COOKIE, httponly=False, **cookie_options)


def _issue_session(user: User, db: Session, response: Response) -> TokenResponse:
    settings = get_settings()
    refresh_token = refresh_session_repo.create(
        db, user, timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    )
    csrf_token = _set_auth_cookies(response, refresh_token)
    return TokenResponse(
        access_token=create_access_token(user.id),
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        csrf_token=csrf_token,
    )


def _csrf_is_valid(request: Request, refresh_token: str, header_token: str | None) -> bool:
    expected = _csrf_token(refresh_token)
    cookie_token = request.cookies.get(_CSRF_COOKIE)
    return bool(
        cookie_token
        and header_token
        and hmac.compare_digest(cookie_token, expected)
        and hmac.compare_digest(header_token, expected)
    )


def _invalid_refresh_response() -> JSONResponse:
    response = JSONResponse(
        status_code=401,
        content={"detail": "Refresh token is invalid or expired", "code": "unauthorized"},
        headers={"WWW-Authenticate": "Bearer", "Cache-Control": "no-store"},
    )
    _clear_auth_cookies(response)
    return response


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=201,
    dependencies=[Depends(rate_limit_register)],
    summary="Create account",
)
def register(payload: RegisterRequest, db: DbSession, response: Response) -> RegisterResponse:
    email = payload.email.lower().strip()

    if user_repo.get_by_email(db, email) is not None:
        raise ConflictApiError("Email already registered")
    if payload.username and user_repo.get_by_username(db, payload.username) is not None:
        raise ConflictApiError("Username already taken")

    user = user_repo.create(
        db,
        email=email,
        password_hash=hash_password(payload.password),
        username=payload.username,
        display_name=payload.display_name,
    )

    token = _issue_session(user, db, response)
    return RegisterResponse(**token.model_dump(), user=UserOut.model_validate(user))


@router.post(
    "/login",
    response_model=LoginResponse,
    dependencies=[Depends(rate_limit_login)],
    summary="Authenticate and issue token",
)
def login(payload: LoginRequest, db: DbSession, response: Response) -> LoginResponse:
    email = payload.email.lower().strip()
    user = user_repo.get_by_email(db, email)

    # Same generic message whether the email is unknown or the password is
    # wrong, to avoid account-enumeration leaks.
    if (
        user is None
        or not user.is_active
        or not verify_password(payload.password, user.password_hash)
    ):
        raise UnauthorizedApiError("Invalid email or password")

    token = _issue_session(user, db, response)
    return LoginResponse(**token.model_dump(), user=UserOut.model_validate(user))


@router.get("/csrf", response_model=CsrfTokenResponse, summary="Get the CSRF token for this session")
def get_csrf_token(request: Request, response: Response) -> CsrfTokenResponse:
    refresh_token = request.cookies.get(_REFRESH_COOKIE)
    if not refresh_token:
        response.delete_cookie(
            _CSRF_COOKIE,
            path=_cookie_path(),
            secure=get_settings().refresh_cookie_secure,
            samesite=get_settings().REFRESH_COOKIE_SAME_SITE,
        )
        response.headers["Cache-Control"] = "no-store"
        return CsrfTokenResponse(csrf_token=None)
    csrf_token = _csrf_token(refresh_token)
    response.set_cookie(
        _CSRF_COOKIE,
        csrf_token,
        max_age=get_settings().REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        httponly=False,
        path=_cookie_path(),
        secure=get_settings().refresh_cookie_secure,
        samesite=get_settings().REFRESH_COOKIE_SAME_SITE,
    )
    response.headers["Cache-Control"] = "no-store"
    return CsrfTokenResponse(csrf_token=csrf_token)


@router.post("/refresh", response_model=TokenResponse, summary="Rotate the refresh token")
def refresh(
    request: Request,
    response: Response,
    db: DbSession,
    csrf_header: str | None = Header(default=None, alias="X-CSRF-Token"),
) -> TokenResponse | JSONResponse:
    refresh_token = request.cookies.get(_REFRESH_COOKIE)
    if not refresh_token:
        return _invalid_refresh_response()
    if not _csrf_is_valid(request, refresh_token, csrf_header):
        raise ForbiddenApiError("CSRF validation failed")

    rotated = refresh_session_repo.rotate(
        db,
        refresh_token,
        timedelta(days=get_settings().REFRESH_TOKEN_EXPIRE_DAYS),
    )
    if rotated is None:
        return _invalid_refresh_response()
    user, next_refresh_token = rotated
    csrf_token = _set_auth_cookies(response, next_refresh_token)
    response.headers["Cache-Control"] = "no-store"
    return TokenResponse(
        access_token=create_access_token(user.id),
        expires_in=get_settings().ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        csrf_token=csrf_token,
    )


@router.post("/logout", status_code=204, summary="Revoke this refresh-token family")
def logout(
    request: Request,
    response: Response,
    db: DbSession,
    csrf_header: str | None = Header(default=None, alias="X-CSRF-Token"),
) -> Response:
    refresh_token = request.cookies.get(_REFRESH_COOKIE)
    if refresh_token:
        if not _csrf_is_valid(request, refresh_token, csrf_header):
            raise ForbiddenApiError("CSRF validation failed")
        refresh_session_repo.revoke(db, refresh_token)
    _clear_auth_cookies(response)
    response.status_code = 204
    response.headers["Cache-Control"] = "no-store"
    return response
