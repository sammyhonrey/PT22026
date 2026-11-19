"""Tests for configuration loading and the error registry."""

from __future__ import annotations

from mmsa.config import load_config
from mmsa.errors import (
    ERROR_REGISTRY,
    MMSAError,
    MMSAWarning,
    AudioTooShortError,
    InferenceError,
)


def test_config_loads(project_root):
    cfg = load_config(project_root / "config", device="cpu")
    assert cfg.device == "cpu"
    assert cfg.models.asr_id == "openai/whisper-small.en"
    assert cfg.models.acoustic_id == "superb/wav2vec2-base-superb-er"
    assert cfg.models.text_id == "bhadresh-savani/distilbert-base-uncased-emotion"


def test_fusion_config_values(app_config):
    fusion = app_config.fusion
    assert fusion.mode in {"credal", "weighted"}
    assert 0.0 < fusion.gamma <= 2.0
    assert 0.0 < fusion.delta <= 2.0
    assert fusion.conflict_aligned < fusion.conflict_mild
    assert set(fusion.priors_acoustic) == {"happy", "sad", "angry", "fear", "disgust", "neutral"}
    assert set(fusion.priors_text) == set(fusion.priors_acoustic)


def test_stress_config_values(app_config):
    stress = app_config.stress
    assert stress.threshold_low < stress.threshold_high
    assert stress.weights["negative_valence"] > 0
    assert stress.weights["positive_emotion"] < 0  # protective factor
    assert stress.instability_features == ("f0_iqr", "rms_std", "pause_ratio", "zcr_std")


def test_error_registry_is_complete():
    assert len(ERROR_REGISTRY) == 12
    assert list(ERROR_REGISTRY) == [f"MMSA-E{n}" for n in range(101, 113)]


def test_error_carries_code_and_hint():
    err = AudioTooShortError(0.4, 1.0)
    assert err.code == "MMSA-E104"
    assert "0.4" in err.user_message
    assert err.hint
    assert err.to_dict()["severity"] == "error"
    assert "MMSA-E104" in str(err)


def test_warning_severity_differs():
    assert MMSAWarning.severity == "warning"
    assert MMSAError.severity == "error"


def test_inference_error_mentions_stage():
    err = InferenceError("acoustic")
    assert "acoustic" in err.user_message


def test_device_resolution_returns_valid_string():
    cfg = load_config(device="auto")
    assert cfg.device in {"cpu", "cuda", "mps"}
