"""Application entry point.

Responsibilities:
- Configure logging before anything else.
- Build the FastAPI app with explicit CORS origins (no wildcards).
- Attach a request-correlation middleware (request id + access log).
- Register global exception handlers and the /api/v1 router.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.body_limit import RequestBodyLimitMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.services import model_service

setup_logging()
logger = get_logger("main")

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("startup app=%s env=%s", settings.APP_NAME, settings.APP_ENV)
    # Load the model once at startup; on failure readiness reports it honestly.
    model_service.load_model_once()
    yield
    logger.info("shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version="0.1.0",
        lifespan=lifespan,
        # Do not leak internals via docs in production.
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
    )

    # --- CORS: explicit origins only -----------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "Accept",
            "Origin",
            "X-Requested-With",
            "X-CSRF-Token",
        ],
        max_age=600,
    )

    # --- Request body size limit (413 on oversized bodies) --------------------
    app.add_middleware(RequestBodyLimitMiddleware, max_body_size=settings.MAX_REQUEST_BODY_BYTES)

    # --- Trusted hosts (empty = disabled) -------------------------------------
    # In production behind a reverse proxy, run uvicorn with ``--proxy-headers``
    # so ``X-Forwarded-Proto`` (HTTPS) is honored, and set ``ALLOWED_HOSTS``.
    if settings.allowed_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

    # --- Request correlation + access log -------------------------------------
    @app.middleware("http")
    async def correlate_requests(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        # Method/path/status/duration only — never headers, bodies, or tokens.
        logger.info(
            "request method=%s path=%s status=%s duration_ms=%.1f request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )
        return response

    register_exception_handlers(app)

    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    @app.get("/", include_in_schema=False)
    def service_identity() -> dict[str, str]:
        """Basic service identity for humans and health checks."""
        return {"app": settings.APP_NAME, "env": settings.APP_ENV, "docs": "/docs"}

    return app


app = create_app()
