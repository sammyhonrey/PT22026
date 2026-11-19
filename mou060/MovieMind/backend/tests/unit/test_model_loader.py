"""Tests for app.ml.model_loader (T3.1 acceptance criteria)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ml.model_loader import ModelArtifactError, load_model

ARTIFACT_DIR = Path("artifacts")

pytestmark = pytest.mark.skipif(
    not ARTIFACT_DIR.is_dir(), reason="real artifacts not present"
)


def test_loads_real_bundle():
    loaded = load_model(ARTIFACT_DIR)
    assert loaded.version == "1.0.0"
    assert loaded.device == "cpu"
    assert len(loaded.user_mapping) == loaded.config["num_users"]
    assert len(loaded.movie_mapping) == loaded.config["num_movies"]
    assert not loaded.model.training  # eval mode


def test_missing_directory_raises():
    with pytest.raises(ModelArtifactError, match="not found"):
        load_model("does-not-exist")


def test_missing_file_raises(tmp_path):
    # Copy a valid bundle minus one required file.
    (tmp_path / "model_config.json").write_text(
        json.dumps(
            {
                "num_users": 2,
                "num_movies": 2,
                "embedding_dim": 4,
                "mlp_hidden_dims": [8],
                "dropout": 0.1,
                "model_version": "test",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ModelArtifactError, match="user mapping"):
        load_model(tmp_path)


def test_out_of_range_mapping_raises(tmp_path):
    (tmp_path / "model_config.json").write_text(
        json.dumps(
            {
                "num_users": 2,
                "num_movies": 2,
                "embedding_dim": 4,
                "mlp_hidden_dims": [8],
                "dropout": 0.1,
                "model_version": "test",
            }
        ),
        encoding="utf-8",
    )
    # Movie index 2 is out of range for num_movies=2 (valid: 0, 1).
    (tmp_path / "user_mapping.json").write_text('{"1": 0, "2": 1}', encoding="utf-8")
    (tmp_path / "movie_mapping.json").write_text('{"1": 0, "2": 2}', encoding="utf-8")
    (tmp_path / "moviemind_ncf.pt").write_bytes(b"not a real checkpoint")
    with pytest.raises(ModelArtifactError, match="out of range"):
        load_model(tmp_path)


def test_invalid_config_type_raises(tmp_path):
    (tmp_path / "model_config.json").write_text('{"num_users": "many"}', encoding="utf-8")
    with pytest.raises(ModelArtifactError):
        load_model(tmp_path)
