"""Application error model and global exception handlers.

Design goals:
- Every error response has the same JSON shape: ``{"detail": ..., "code": ...}``.
- Unhandled exceptions return a generic message to the client; the real
  traceback goes to the log, never to the response body.
- Database-unavailable and model-unavailable failures are distinct states so
  ``/api/v1/ready`` and monitoring can tell them apart.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger

logger = get_logger("errors")

# starlette renamed this constant; resolve it robustly across versions.
HTTP_422 = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)


class ApiError(Exception):
    """Base class for errors that map to a controlled HTTP response."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"
    log_level: int = logging.ERROR

    def __init__(
        self, detail: str, *, code: str | None = None, headers: dict[str, str] | None = None
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.headers = headers
        if code is not None:
            self.code = code

    def to_response(self) -> JSONResponse:
        return JSONResponse(
            status_code=self.status_code,
            content={"detail": self.detail, "code": self.code},
            headers=self.headers,
        )


class ValidationApiError(ApiError):
    status_code = HTTP_422
    code = "validation_error"
    log_level = logging.INFO


class UnauthorizedApiError(ApiError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "unauthorized"
    log_level = logging.INFO

    def __init__(self, detail: str = "Not authenticated") -> None:
        super().__init__(detail, headers={"WWW-Authenticate": "Bearer"})


class ForbiddenApiError(ApiError):
    status_code = status.HTTP_403_FORBIDDEN
    code = "forbidden"
    log_level = logging.INFO


class NotFoundApiError(ApiError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"
    log_level = logging.INFO


class ConflictApiError(ApiError):
    status_code = status.HTTP_409_CONFLICT
    code = "conflict"
    log_level = logging.INFO


class NotImplementedApiError(ApiError):
    status_code = status.HTTP_501_NOT_IMPLEMENTED
    code = "not_implemented"
    log_level = logging.INFO


class DatabaseUnavailableError(ApiError):
    """The database is unreachable or otherwise unusable."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "database_unavailable"

    def __init__(self, detail: str = "Database is temporarily unavailable") -> None:
        super().__init__(detail)


class ModelUnavailableError(ApiError):
    """Model artifacts are missing, incompatible, or failed to load."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "model_unavailable"

    def __init__(self, detail: str = "Recommendation model is not available") -> None:
        super().__init__(detail)


class TooManyRequestsApiError(ApiError):
    """The client has exceeded an in-process rate limit."""

    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    code = "rate_limited"
    log_level = logging.INFO

    def __init__(self, detail: str = "Too many requests. Please slow down.") -> None:
        super().__init__(detail, headers={"Retry-After": "60"})


def _error_body(detail: str, code: str) -> dict[str, Any]:
    return {"detail": detail, "code": code}


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all global handlers to ``app``."""

    @app.exception_handler(ApiError)
    async def handle_api_error(_: Request, exc: ApiError) -> JSONResponse:
        logger.log(exc.log_level, "api_error code=%s detail=%s", exc.code, exc.detail)
        return exc.to_response()

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        # Framework-raised HTTP errors keep the same body shape.
        logger.info("http_error status=%s detail=%s", exc.status_code, exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(str(exc.detail), "http_error"),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        logger.info("validation_error count=%s", len(exc.errors()))
        return JSONResponse(
            status_code=HTTP_422,
            content=_error_body("Request validation failed", "validation_error"),
        )

    @app.exception_handler(SQLAlchemyError)
    async def handle_db_error(_: Request, exc: SQLAlchemyError) -> JSONResponse:
        # Log internals; return a generic message.
        logger.exception("database_error type=%s", type(exc).__name__)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_error_body("Database is temporarily unavailable", "database_unavailable"),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error type=%s", type(exc).__name__)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_body("Internal server error", "internal_error"),
        )
