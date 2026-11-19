"""Model artifact smoke test (T3.4 acceptance criteria).

Deterministic, end-to-end proof that the *real* artifact bundle loads and
produces finite predictions in the valid rating range — exercised both through
the service and through the public ``/model/info`` + readiness endpoints.

Skipped (not failed) with a clear reason when ``backend/artifacts`` is absent.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import model_service
from app.services.model_service import get_ncf_recommender

ARTIFACT_DIR = Path("artifacts")
HAS_ARTIFACTS = ARTIFACT_DIR.is_dir()

pytestmark = pytest.mark.skipif(
    not HAS_ARTIFACTS, reason="real artifact bundle not present in backend/artifacts"
)

client = TestClient(app)


def test_service_loads_real_bundle():
    status = model_service.get_model_status()
    assert status["status"] == "loaded"
    assert status["loaded"] is True
    assert status["version"] == "1.0.0"


def test_known_inference_is_finite_in_range():
    rec = get_ncf_recommender()
    scores = rec.predict_batch([(1, 1), (2, 3), (5, 10), (42, 250)])
    assert len(scores) == 4
    for s in scores:
        assert math.isfinite(s)
        assert 0.5 <= s <= 5.0


def test_all_mapped_predictions_within_rating_range():
    """The linear head is unbounded, so every prediction must be clamped to the
    advertised [0.5, 5.0] rating scale (regression guard for the clamp)."""
    rec = get_ncf_recommender()
    user_id = rec.known_user_ids()[0]
    scores = rec.score_candidates(user_id, rec.known_movie_ids())
    values = [s for _, s in scores]
    assert len(values) == len(rec.known_movie_ids())
    for s in values:
        assert math.isfinite(s)
        assert 0.5 <= s <= 5.0


def test_model_info_endpoint():
    r = client.get("/api/v1/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_type"] == "ncf"
    assert body["status"] == "loaded"
    assert body["loaded"] is True
    assert body["version"] == "1.0.0"
    assert body["rating_scale"]["min"] == 0.5
    assert body["rating_scale"]["max"] == 5.0
    # Safe summary: no filesystem paths or secrets.
    text = r.text.lower()
    for forbidden in ("c:\\", "/home/", "password", "secret"):
        assert forbidden not in text


def test_readiness_reports_model_loaded():
    r = client.get("/api/v1/ready")
    body = r.json()
    assert body["checks"]["model"]["status"] == "loaded"
    # Overall readiness also requires the database; assert structure only.
    assert body["checks"]["database"]["status"] in ("ok", "unavailable")
