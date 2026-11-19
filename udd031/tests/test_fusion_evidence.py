"""The transcript evidence gate must change fusion, and only in the right direction."""

from __future__ import annotations

from mmsa.config import FusionConfig
from mmsa.fusion import fuse
from mmsa.schemas import BranchResult

CANON = ("happy", "sad", "angry", "fear", "disgust", "neutral")


def _branch(modality: str, top: str, strength: float = 0.85, coverage: float = 1.0) -> BranchResult:
    rest = (1.0 - strength) / (len(CANON) - 1)
    probs = {c: (strength if c == top else rest) for c in CANON}
    return BranchResult(
        modality=modality,
        raw_labels=CANON,
        raw_probs=tuple(probs[c] for c in CANON),
        mapped_probs=probs,
        coverage=coverage,
        confidence=1.0,
        weight=0.0,
        per_class_weight={},
        top_label=top,
        latency_ms=0.0,
    )


def test_zero_evidence_lets_the_acoustic_branch_decide():
    # The realistic shape of the defect: a certain text branch against a hedged
    # acoustic one, both on a transcript that mentions nothing.
    acoustic = _branch("acoustic", "happy", strength=0.55)
    text = _branch("text", "angry", strength=0.95)
    cfg = FusionConfig()

    without = fuse(acoustic, text, cfg, text_evidence=1.0)
    with_gate = fuse(acoustic, text, cfg, text_evidence=0.0)

    assert without.top_label == "angry"  # the defect: a silent transcript wins
    assert with_gate.top_label == "happy"  # the fix: the audio decides


def test_gate_is_off_by_default_for_backwards_compatibility():
    acoustic = _branch("acoustic", "sad")
    text = _branch("text", "happy")
    cfg = FusionConfig()
    assert fuse(acoustic, text, cfg).probs == fuse(acoustic, text, cfg, text_evidence=1.0).probs


def test_evidence_scales_text_weight_monotonically():
    acoustic = _branch("acoustic", "neutral", strength=0.5)
    text = _branch("text", "angry", strength=0.6)
    cfg = FusionConfig()
    weights = [
        fuse(acoustic, text, cfg, text_evidence=e).per_class_weights["text"]["angry"] for e in (0.0, 0.25, 0.5, 1.0)
    ]
    assert weights == sorted(weights)


def test_conflict_is_withheld_when_the_transcript_has_no_evidence():
    acoustic = _branch("acoustic", "happy")
    text = _branch("text", "angry")
    cfg = FusionConfig()
    silent = fuse(acoustic, text, cfg, text_evidence=0.0)
    mentioned = fuse(acoustic, text, cfg, text_evidence=0.8)
    assert silent.conflict_index is None
    assert silent.agreement == "indeterminate"
    assert mentioned.conflict_index is not None


def test_floor_stops_the_text_branch_from_vanishing_entirely():
    acoustic = _branch("acoustic", "happy")
    text = _branch("text", "angry")
    cfg = FusionConfig(text_evidence_floor=0.02)
    out = fuse(acoustic, text, cfg, text_evidence=0.0)
    assert out.per_class_weights["text"]["angry"] > 0.0
