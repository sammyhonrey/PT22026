"""Tests for the stress index."""

from __future__ import annotations

import pytest

from mmsa.schemas import DimensionalResult
from mmsa.stress import compute_stress, robust_z, sigmoid

BALANCED = {"happy": 0.1, "sad": 0.2, "angry": 0.2, "fear": 0.1, "disgust": 0.05, "neutral": 0.35}
THREATENING = {"happy": 0.02, "sad": 0.15, "angry": 0.45, "fear": 0.30, "disgust": 0.03, "neutral": 0.05}
PLEASANT = {"happy": 0.70, "sad": 0.02, "angry": 0.02, "fear": 0.01, "disgust": 0.05, "neutral": 0.20}

REFERENCE = {
    "f0_iqr": {"median": 40.0, "iqr": 25.0},
    "rms_std": {"median": 6.0, "iqr": 3.0},
    "pause_ratio": {"median": 0.25, "iqr": 0.15},
    "zcr_std": {"median": 0.04, "iqr": 0.02},
}

CALM_FEATURES = {"f0_iqr": 40.0, "rms_std": 6.0, "pause_ratio": 0.25, "zcr_std": 0.04}
TENSE_FEATURES = {"f0_iqr": 110.0, "rms_std": 18.0, "pause_ratio": 0.05, "zcr_std": 0.14}


def test_sigmoid_bounds():
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert 0.0 < sigmoid(-50.0) < 0.01
    assert sigmoid(20.0) > 0.999
    # At +50 the logistic saturates to exactly 1.0 in float64; that is correct
    # behaviour, not a clamp, so the assertion records it rather than dodging it.
    assert sigmoid(50.0) == pytest.approx(1.0)
    assert sigmoid(-1.0) < sigmoid(1.0)


def test_robust_z_clips_and_handles_degenerate_iqr():
    assert robust_z(65.0, 40.0, 25.0) == pytest.approx(1.0)
    assert robust_z(1e9, 40.0, 25.0) == 3.0
    assert robust_z(65.0, 40.0, 0.0) == 0.0


def test_contributors_reconstruct_the_logit(app_config):
    stress_cfg = app_config.stress
    out = compute_stress(
        BALANCED,
        cfg=stress_cfg,
        features=CALM_FEATURES,
        asr_confidence=0.9,
        lexical_negative=0.2,
        prosody_reference=REFERENCE,
    )
    reconstructed = stress_cfg.bias + sum(out.contributors.values())
    assert out.features["logit"] == pytest.approx(reconstructed, abs=1e-3)


def test_threat_emotions_raise_the_score(app_config):
    cfg = app_config.stress
    calm = compute_stress(
        PLEASANT, cfg=cfg, features=CALM_FEATURES, asr_confidence=0.95,
        lexical_negative=0.0, prosody_reference=REFERENCE,
    )
    tense = compute_stress(
        THREATENING, cfg=cfg, features=TENSE_FEATURES, asr_confidence=0.95,
        lexical_negative=0.8, prosody_reference=REFERENCE,
    )
    assert tense.score > calm.score
    assert tense.band in {"Medium", "High"}
    assert calm.band in {"Low", "Medium"}


def test_positive_emotion_is_protective(app_config):
    cfg = app_config.stress
    with_happy = compute_stress(
        PLEASANT, cfg=cfg, features=CALM_FEATURES, asr_confidence=0.9,
        lexical_negative=0.0, prosody_reference=REFERENCE,
    )
    assert with_happy.contributors["positive_emotion"] < 0.0


def test_monotone_in_negative_lexicon(app_config):
    cfg = app_config.stress
    low = compute_stress(
        BALANCED, cfg=cfg, features=CALM_FEATURES, asr_confidence=0.9,
        lexical_negative=0.0, prosody_reference=REFERENCE,
    )
    high = compute_stress(
        BALANCED, cfg=cfg, features=CALM_FEATURES, asr_confidence=0.9,
        lexical_negative=1.0, prosody_reference=REFERENCE,
    )
    assert high.score > low.score


def test_asr_uncertainty_contributes(app_config):
    cfg = app_config.stress
    sure = compute_stress(
        BALANCED, cfg=cfg, features=CALM_FEATURES, asr_confidence=1.0,
        lexical_negative=0.0, prosody_reference=REFERENCE,
    )
    unsure = compute_stress(
        BALANCED, cfg=cfg, features=CALM_FEATURES, asr_confidence=0.0,
        lexical_negative=0.0, prosody_reference=REFERENCE,
    )
    assert unsure.score > sure.score
    assert sure.contributors["asr_uncertainty"] == pytest.approx(0.0)


def test_extreme_inputs_land_in_extreme_bands(app_config):
    cfg = app_config.stress
    maxed = compute_stress(
        THREATENING, cfg=cfg, features=TENSE_FEATURES, asr_confidence=0.0,
        lexical_negative=1.0, prosody_reference=REFERENCE,
    )
    floor = compute_stress(
        PLEASANT, cfg=cfg, features=CALM_FEATURES, asr_confidence=1.0,
        lexical_negative=0.0, prosody_reference=REFERENCE,
    )
    assert maxed.band == "High"
    assert floor.band == "Low"


def test_borderline_qualifier_appears_near_threshold(app_config):
    """The qualifier is a property of the banding rule, so test it directly.

    Sweeping inputs until a score happens to land within five points of a
    threshold tests the input distribution, not the rule. Moving the threshold
    to the score instead makes the window exact and deterministic.
    """
    from dataclasses import replace

    cfg = app_config.stress
    base = compute_stress(
        BALANCED, cfg=cfg, features=CALM_FEATURES, asr_confidence=0.9,
        lexical_negative=0.2, prosody_reference=REFERENCE,
    )
    shifted = replace(cfg, threshold_low=min(0.99, base.score + 0.01))
    out = compute_stress(
        BALANCED, cfg=shifted, features=CALM_FEATURES, asr_confidence=0.9,
        lexical_negative=0.2, prosody_reference=REFERENCE,
    )

    assert out.band == "Low"
    assert "borderline" in out.band_label


def test_dimensional_head_overrides_the_proxy(app_config):
    cfg = app_config.stress
    proxy_run = compute_stress(
        BALANCED, cfg=cfg, features=CALM_FEATURES, asr_confidence=0.9,
        lexical_negative=0.2, prosody_reference=REFERENCE,
    )
    dim_run = compute_stress(
        BALANCED,
        cfg=cfg,
        features=CALM_FEATURES,
        asr_confidence=0.9,
        lexical_negative=0.2,
        dimensional=DimensionalResult(arousal=0.95, dominance=0.2, valence=0.05, model_id="test"),
        prosody_reference=REFERENCE,
    )
    assert proxy_run.features["affect_source"] == 0.0
    assert dim_run.features["affect_source"] == 1.0
    assert dim_run.score > proxy_run.score  # high arousal, near-zero valence


def test_missing_reference_yields_zero_instability(app_config):
    cfg = app_config.stress
    out = compute_stress(
        BALANCED, cfg=cfg, features=CALM_FEATURES, asr_confidence=0.9,
        lexical_negative=0.0, prosody_reference=None,
    )
    assert out.features["prosody_instability"] == pytest.approx(0.0)


def test_score_always_in_unit_interval(app_config):
    cfg = app_config.stress
    for probs in (BALANCED, THREATENING, PLEASANT):
        out = compute_stress(
            probs, cfg=cfg, features=TENSE_FEATURES, asr_confidence=0.5,
            lexical_negative=0.5, prosody_reference=REFERENCE,
        )
        assert 0.0 <= out.score <= 1.0
        assert out.band in {"Low", "Medium", "High"}
