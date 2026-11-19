"""Release identity must follow content and reject partial experiment replacement."""

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest

from src.artifacts import (ArtifactReleaseError, RUNTIME_FILES, capture_digests,
                           publish_manifest, read_release, verified_bytes)
from src.features import FEATURE_COLUMNS
from src.forecasting import ModelArtifactError, load_model_bundle


@pytest.fixture
def local_experiment(tmp_path):
    for name in RUNTIME_FILES:
        (tmp_path / name).write_bytes(b"fixture-A")
    (tmp_path / "training_metadata.json").write_text(json.dumps({"dataset_id": "xky_2025"}))
    return tmp_path


def test_local_snapshot_detects_content_change_with_identical_stat_metadata(local_experiment):
    before = read_release(local_experiment, "xky_2025")
    path = local_experiment / "model_bundle.joblib"
    stat = path.stat()
    path.write_bytes(b"fixture-B")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    after = read_release(local_experiment, "xky_2025")
    assert path.stat().st_size == stat.st_size
    assert path.stat().st_mtime_ns == stat.st_mtime_ns
    assert after.release_id != before.release_id


def test_publish_and_verify_same_release(local_experiment):
    release = publish_manifest(local_experiment, "xky_2025", capture_digests(local_experiment))
    assert release.managed
    assert read_release(local_experiment, "xky_2025", require_manifest=True) == release
    assert not list(local_experiment.glob("*.tmp"))


@pytest.mark.parametrize("filename", RUNTIME_FILES)
def test_manifest_blocks_partial_replacement(local_experiment, filename):
    publish_manifest(local_experiment, "xky_2025", capture_digests(local_experiment))
    (local_experiment / filename).write_bytes(b"replacement")
    with pytest.raises(ArtifactReleaseError, match=filename.replace(".", r"\.")):
        read_release(local_experiment, "xky_2025")


def test_manifest_rejects_other_household_and_malformed_json(local_experiment):
    publish_manifest(local_experiment, "xky_2025", capture_digests(local_experiment))
    with pytest.raises(ArtifactReleaseError, match="dataset identity"):
        read_release(local_experiment, "uci_household")
    (local_experiment / "release_manifest.json").write_text("{invalid")
    with pytest.raises(ArtifactReleaseError, match="invalid"):
        read_release(local_experiment, "xky_2025")


def test_required_manifest_and_missing_runtime_files_are_actionable(local_experiment):
    with pytest.raises(ArtifactReleaseError, match="--write-manifests"):
        read_release(local_experiment, "xky_2025", require_manifest=True)
    expected = capture_digests(local_experiment)
    (local_experiment / "test_metrics.csv").unlink()
    with pytest.raises(ArtifactReleaseError, match="missing"):
        publish_manifest(local_experiment, "xky_2025", expected)


def test_changed_bytes_are_rejected_before_model_deserialization(tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump({"models": {}, "selected_model": "Seasonal Naive", "peak_threshold_kwh": 1.0,
                 "feature_columns": FEATURE_COLUMNS, "dataset_id": "xky_2025"}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_model_bundle(path, "xky_2025", digest)["dataset_id"] == "xky_2025"
    path.write_bytes(path.read_bytes() + b"changed")
    with patch("src.forecasting.joblib.load") as deserialize:
        with pytest.raises(ModelArtifactError, match="changed while loading"):
            load_model_bundle(path, "xky_2025", digest)
        deserialize.assert_not_called()
    with pytest.raises(ArtifactReleaseError, match="changed while loading"):
        verified_bytes(path, digest)


def test_publication_rejects_new_model_with_old_prediction_evidence(tmp_path, monkeypatch):
    """A matching score CSV alone must not certify a replaced fitted model."""
    from dataclasses import replace
    from scripts import verify_release
    from src.datasets import DATASETS

    index = pd.date_range("2026-01-01", periods=193, freq="h")
    processed = tmp_path / "hourly.csv"
    pd.DataFrame({"timestamp": index, "energy_kwh": 1.25}).to_csv(processed, index=False)
    origin = index[168]
    groups = []
    for model in ["Seasonal Naive", "Ridge", "HistGradientBoosting"]:
        groups.append(pd.DataFrame({"forecast_origin": origin, "target_timestamp": index[169:],
                                    "horizon": range(1, 25), "model": model,
                                    "actual_kwh": 1.25, "predicted_kwh": 1.25}))
    pd.concat(groups).to_csv(tmp_path / "test_predictions.csv", index=False)

    class TestOnlyConstant:
        def __init__(self, value):
            self.value = value

        def predict(self, frame):
            return np.full(len(frame), self.value)

    bundle = {"models": {"Ridge": TestOnlyConstant(1.25), "HistGradientBoosting": TestOnlyConstant(1.25)}}
    spec = replace(DATASETS["xky_2025"], artifact_dir=tmp_path, processed_path=processed)
    monkeypatch.setitem(verify_release.DATASETS, "xky_2025", spec)
    monkeypatch.setattr(verify_release, "load_model_bundle", lambda *args: bundle)
    verify_release.verify_fitted_predictions("xky_2025")
    bundle["models"]["HistGradientBoosting"].value = 2.0
    with pytest.raises(ArtifactReleaseError, match="do not match the fitted HistGradientBoosting"):
        verify_release.verify_fitted_predictions("xky_2025")
