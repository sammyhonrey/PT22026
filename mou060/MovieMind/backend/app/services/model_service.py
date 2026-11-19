"""Model service: load the NCF bundle once and expose status/recommender.

The model is loaded exactly once per process (thread-safe, idempotent) and
shared by every request. On any artifact failure the error is recorded and
surfaced through the readiness endpoint and ``/model/info`` — the app never
silently substitutes fake recommendations.
"""

from __future__ import annotations

import threading
from typing import Any

from app.core.config import get_settings
from app.core.errors import ModelUnavailableError
from app.core.logging import get_logger
from app.ml.model_loader import LoadedModel, ModelArtifactError, load_model
from app.ml.recommenders.ncf_recommender import NCFRecommender

logger = get_logger("model")

_state: dict[str, Any] = {"loaded": None, "error": None}
_lock = threading.Lock()


def load_model_once() -> LoadedModel | None:
    """Idempotently load the model from the configured artifact directory.

    Returns the loaded model, or ``None`` if loading failed (the reason is
    available via :func:`get_model_status`).
    """
    if _state["loaded"] is not None or _state["error"] is not None:
        return _state["loaded"]

    with _lock:
        # Re-check under the lock (another thread may have loaded first).
        if _state["loaded"] is not None or _state["error"] is not None:
            return _state["loaded"]
        try:
            settings = get_settings()
            loaded = load_model(settings.MODEL_ARTIFACT_DIR, settings.MODEL_DEVICE)
            _state["loaded"] = loaded
            logger.info(
                "model loaded version=%s users=%d movies=%d device=%s",
                loaded.version,
                len(loaded.user_mapping),
                len(loaded.movie_mapping),
                loaded.device,
            )
        except ModelArtifactError as exc:
            _state["error"] = str(exc)
            logger.error("model load failed: %s", exc)

    return _state["loaded"]


def get_model_status() -> dict[str, Any]:
    """Status summary for the readiness endpoint."""
    load_model_once()
    loaded: LoadedModel | None = _state["loaded"]
    if loaded is not None:
        return {
            "status": "loaded",
            "loaded": True,
            "version": loaded.version,
            "device": loaded.device,
        }
    if _state["error"]:
        return {
            "status": "unavailable",
            "loaded": False,
            "version": None,
            "reason": _state["error"],
        }
    return {"status": "not_configured", "loaded": False, "version": None}


def get_model_info() -> dict[str, Any]:
    """Richer, public-safe summary for ``/model/info`` (no secrets/paths)."""
    status = get_model_status()
    info: dict[str, Any] = {
        "model_type": "ncf",
        "status": status["status"],
        "loaded": status["loaded"],
        "version": status.get("version"),
        "supported_features": ["rating_prediction", "movielens_demo"],
    }
    loaded: LoadedModel | None = _state["loaded"]
    if loaded is not None:
        config = loaded.config
        info.update(
            {
                "dataset": config.get("dataset"),
                "rating_scale": {
                    "min": config.get("rating_min"),
                    "max": config.get("rating_max"),
                },
                "num_users": config.get("num_users"),
                "num_movies": config.get("num_movies"),
                "device": loaded.device,
            }
        )
    else:
        info["reason"] = status.get("reason")
    return info


def get_ncf_recommender() -> NCFRecommender:
    """Return the shared NCF recommender, or raise a controlled 503.

    MovieMind account IDs must never be passed to the returned recommender —
    it only understands raw MovieLens IDs.
    """
    loaded = load_model_once()
    if loaded is None:
        raise ModelUnavailableError(_state["error"] or "Recommendation model is not available")
    return NCFRecommender(loaded)
