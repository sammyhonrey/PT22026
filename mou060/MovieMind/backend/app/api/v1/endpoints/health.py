"""Operational endpoints: service identity, liveness, readiness, model info."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response, status
from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import get_engine
from app.services import model_service

router = APIRouter()


@router.get("/health", summary="Liveness check")
def liveness() -> dict[str, Any]:
    """Process is up. Reports no secrets, no paths, no infrastructure details."""
    settings = get_settings()
    return {"status": "ok", "app": settings.APP_NAME, "env": settings.APP_ENV}


@router.get("/ready", summary="Readiness: database and model state")
def readiness(response: Response) -> dict[str, Any]:
    """Ready only when every required dependency is usable.

    Database and model failures are reported separately so operators can
    distinguish the two states. Failures never expose secrets, file paths,
    or stack traces.
    """
    checks: dict[str, Any] = {}
    ready = True

    # --- Database -------------------------------------------------------------
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = {"status": "ok"}
    except Exception as exc:  # noqa: BLE001 — any failure means not-ready
        ready = False
        checks["database"] = {"status": "unavailable", "reason": type(exc).__name__}

    # --- Model ----------------------------------------------------------------
    model_status = model_service.get_model_status()
    checks["model"] = model_status
    if not model_status.get("loaded"):
        ready = False

    response.status_code = status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ready" if ready else "not_ready", "checks": checks}
