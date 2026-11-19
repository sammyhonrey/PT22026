"""Tests for the fusion engine, using synthetic distributions only (no models)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mmsa.config import FusionConfig, load_config
from mmsa.fusion import (
    agreement_level,
    fuse,
    jensen_shannon,
    normalised_entropy,
)
from mmsa.schemas import BranchResult

ACOUSTIC = ["ang", "hap", "neu", "sad"]
TEXT = ["joy", "love", "anger", "sadness", "fear", "surprise"]


def make_branch(ontology, modality, probs, confidence=0.8, coverage_override=None):
    labels = ACOUSTIC if modality == "acoustic" else TEXT
    mapped, coverage = ontology.map_probs(modality, labels, probs)
    if coverage_override is not None:
        total = sum(mapped.values()) or 1.0
        mapped = {k: v * (coverage_override / total) for k, v in mapped.items()}
        coverage = coverage_override
    top = max(mapped.items(), key=lambda kv: kv[1])[0]
    return BranchResult(
        modality=modality,
        raw_labels=tuple(labels),
        raw_probs=tuple(float(p) for p in probs),
        mapped_probs=mapped,
        coverage=float(coverage),
        confidence=float(confidence),
        weight=0.0,
        per_class_weight={},
        top_label=top,
        latency_ms=1.0,
    )


@pytest.fixture(scope="module")
def cfg(project_root) -> FusionConfig:
    return load_config(project_root / "config", device="cpu").fusion


# Information-theoretic helpers


def test_js_divergence_is_zero_for_identical_distributions():
    p = np.array([0.7, 0.2, 0.1])
    assert jensen_shannon(p, p) == pytest.approx(0.0, abs=1e-12)


def test_js_divergence_is_bounded_by_ln2():
    p = np.array([1.0, 0.0, 0.0, 0.0])
    q = np.array([0.0, 1.0, 0.0, 0.0])
    assert jensen_shannon(p, q) == pytest.approx(math.log(2.0), abs=1e-9)


def test_js_divergence_is_symmetric():
    p = np.array([0.9, 0.05, 0.05])
    q = np.array([0.1, 0.1, 0.8])
    assert jensen_shannon(p, q) == pytest.approx(jensen_shannon(q, p))


def test_normalised_entropy_extremes():
    assert normalised_entropy(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert normalised_entropy(np.array([0.25, 0.25, 0.25, 0.25])) == pytest.approx(1.0)


def test_agreement_bands(cfg):
    assert agreement_level(0.0, cfg) == "aligned"
    assert agreement_level(0.25, cfg) == "mild_conflict"
    assert agreement_level(0.8, cfg) == "strong_conflict"
    assert agreement_level(None, cfg) == "indeterminate"


# Fusion behaviour


def test_agreement_produces_confident_fused_output(ontology, cfg):
    audio = make_branch(ontology, "acoustic", [0.05, 0.05, 0.05, 0.85], confidence=0.9)
    text = make_branch(ontology, "text", [0.0, 0.0, 0.0, 0.9, 0.05, 0.05], confidence=0.9)
    out = fuse(audio, text, cfg)

    assert out.agreement == "aligned"
    assert out.conflict_index is not None and out.conflict_index < 0.15
    assert out.top_label == "sad"
    assert out.probs["sad"] > 0.8


def test_contradiction_is_detected_and_reported(ontology, cfg):
    audio = make_branch(ontology, "acoustic", [0.90, 0.02, 0.04, 0.04], confidence=0.9)
    text = make_branch(ontology, "text", [0.90, 0.02, 0.02, 0.02, 0.02, 0.02], confidence=0.9)
    out = fuse(audio, text, cfg)

    assert out.agreement == "strong_conflict"
    assert out.conflict_index is not None and out.conflict_index > 0.4
    # The fused answer must sit between the two opinions, not jump to either.
    assert out.probs["angry"] < 0.90
    assert out.probs["happy"] < 0.90


def test_unreachable_class_gets_no_acoustic_mass(ontology, cfg):
    audio = make_branch(ontology, "acoustic", [0.33, 0.33, 0.34, 0.0], confidence=0.9)
    text = make_branch(ontology, "text", [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], confidence=0.9)
    out = fuse(audio, text, cfg)

    text_fear = text.mapped_probs["fear"]
    assert text_fear == pytest.approx(1.0)
    assert out.probs["fear"] < text_fear  # diluted, never amplified
    assert out.probs["fear"] > 0.0


def test_fused_distribution_is_proper(ontology, cfg):
    audio = make_branch(ontology, "acoustic", [0.25, 0.25, 0.25, 0.25])
    text = make_branch(ontology, "text", [0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
    out = fuse(audio, text, cfg)
    assert sum(out.probs.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(v >= 0.0 for v in out.probs.values())
    assert set(out.probs) == {"happy", "sad", "angry", "fear", "disgust", "neutral"}


def test_alpha_effective_responds_to_branch_confidence(ontology, cfg):
    audio = make_branch(ontology, "acoustic", [0.25, 0.25, 0.25, 0.25], confidence=0.9)
    text_confident = make_branch(ontology, "text", [0.2, 0.2, 0.2, 0.2, 0.1, 0.1], confidence=0.95)
    text_unsure = make_branch(ontology, "text", [0.2, 0.2, 0.2, 0.2, 0.1, 0.1], confidence=0.05)

    alpha_confident = fuse(audio, text_confident, cfg).alpha_effective
    alpha_unsure = fuse(audio, text_unsure, cfg).alpha_effective

    assert alpha_confident < alpha_unsure
    assert 0.0 < alpha_confident < 1.0


def test_modes_differ_and_are_a_real_ablation(ontology, project_root):
    base = load_config(project_root / "config", device="cpu").fusion
    credal_cfg = FusionConfig(**{**base.__dict__, "mode": "credal"})
    weighted_cfg = FusionConfig(**{**base.__dict__, "mode": "weighted"})

    audio = make_branch(ontology, "acoustic", [0.7, 0.1, 0.1, 0.1], confidence=0.95)
    text = make_branch(ontology, "text", [0.05, 0.05, 0.05, 0.05, 0.75, 0.05], confidence=0.35)

    credal = fuse(audio, text, credal_cfg).probs
    weighted = fuse(audio, text, weighted_cfg).probs

    divergence = sum(abs(credal[k] - weighted[k]) for k in credal)
    assert divergence > 0.02, "credal and weighted modes must not be equivalent"


def test_indeterminate_conflict_when_a_branch_is_blind(ontology, cfg):
    audio = make_branch(ontology, "acoustic", [0.3, 0.3, 0.4, 0.0], coverage_override=0.30)
    text = make_branch(ontology, "text", [0.9, 0.02, 0.02, 0.02, 0.02, 0.02])
    out = fuse(audio, text, cfg)

    assert out.conflict_index is None
    assert out.agreement == "indeterminate"


def test_degenerate_inputs_fall_back_instead_of_crashing(ontology, cfg):
    zero = {label: 0.0 for label in ("happy", "sad", "angry", "fear", "disgust", "neutral")}

    def empty(modality):
        return BranchResult(
            modality=modality,
            raw_labels=(),
            raw_probs=(),
            mapped_probs=zero,
            coverage=0.0,
            confidence=0.0,
            weight=0.0,
            per_class_weight={},
            top_label="neutral",
            latency_ms=0.0,
        )

    out = fuse(empty("acoustic"), empty("text"), cfg)
    assert sum(out.probs.values()) == pytest.approx(1.0)
    assert out.ignorance == pytest.approx(1.0)
    assert out.agreement == "indeterminate"


def test_ignorance_tracks_coverage(ontology, cfg):
    full_audio = make_branch(ontology, "acoustic", [0.25, 0.25, 0.25, 0.25])
    full_text = make_branch(ontology, "text", [0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
    blind_audio = make_branch(ontology, "acoustic", [0.25, 0.25, 0.25, 0.25], coverage_override=0.5)

    low = fuse(full_audio, full_text, cfg).ignorance
    high = fuse(blind_audio, full_text, cfg).ignorance
    assert high > low


def test_per_class_weights_are_stored_for_both_branches(ontology, cfg):
    audio = make_branch(ontology, "acoustic", [0.25, 0.25, 0.25, 0.25])
    text = make_branch(ontology, "text", [0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
    out = fuse(audio, text, cfg)
    assert set(out.per_class_weights) == {"acoustic", "text"}
    assert all(v > 0 for v in out.per_class_weights["acoustic"].values())


def test_unknown_mode_raises(ontology, project_root):
    base = load_config(project_root / "config", device="cpu").fusion
    bad = FusionConfig(**{**base.__dict__, "mode": "quantum"})
    audio = make_branch(ontology, "acoustic", [0.25, 0.25, 0.25, 0.25])
    text = make_branch(ontology, "text", [0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
    with pytest.raises(ValueError):
        fuse(audio, text, bad)
