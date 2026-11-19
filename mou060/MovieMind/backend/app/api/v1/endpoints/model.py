"""Model diagnostics endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.services import model_service

router = APIRouter()


@router.get("/model/info", summary="Model version, type, and load status")
def model_info() -> dict[str, Any]:
    """Safe public summary of the loaded recommendation model.

    Reports version, type, status, and supported features. Never discloses
    file paths, secrets, or architecture internals.
    """
    return model_service.get_model_info()
