"""Per-class reliability fusion of the two branches.

r = rho * coverage^gamma * confidence^delta, then weighted opinion pooling, with
the mass a branch cannot express left as ignorance instead of being rescaled.
This is not Dempster-Shafer; the difference matters in high-conflict cases and is
discussed in the report. A transcript with no lexical evidence drops to
``text_evidence.floor`` of its reliability and may not report a conflict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .config import FusionConfig
from .schemas import Agreement, BranchResult

__all__ = [
    "FusionOutput",
    "fuse",
    "jensen_shannon",
    "normalised_entropy",
    "agreement_level",
]

_EPS = 1e-12


@dataclass(frozen=True)
class FusionOutput:
    """Result of reconciling two branch opinions."""

    probs: Mapping[str, float]
    per_class_weights: Mapping[str, Mapping[str, float]]
    alpha_effective: float
    conflict_index: float | None
    agreement: Agreement
    ignorance: float
    entropy: float

    @property
    def top_label(self) -> str:
        """Most probable canonical emotion."""
        return max(self.probs.items(), key=lambda kv: kv[1])[0]

    @property
    def confidence(self) -> float:
        """Probability assigned to :attr:`top_label`."""
        return float(self.probs[self.top_label])


# Information-theoretic helpers


def _as_array(mapping: Mapping[str, float], labels: tuple[str, ...]) -> np.ndarray:
    return np.asarray([float(mapping.get(label, 0.0)) for label in labels], dtype=np.float64)


def normalised_entropy(probs: Mapping[str, float] | np.ndarray) -> float:
    """Shannon entropy scaled to ``[0, 1]`` (0 = certain, 1 = uniform)."""
    p = np.asarray(list(probs.values()) if isinstance(probs, Mapping) else probs, dtype=np.float64)
    p = p[p > 0.0]
    if p.size <= 1:
        return 0.0
    p = p / p.sum()
    h = float(-(p * np.log(p)).sum())
    return float(min(1.0, h / math.log(p.size)))


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0.0
    return float((p[mask] * np.log(p[mask] / np.maximum(q[mask], _EPS))).sum())


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats (symmetric, bounded by ``ln 2``)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return max(0.0, 0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def agreement_level(conflict: float | None, cfg: FusionConfig) -> Agreement:
    """Map a normalised conflict index onto the three-band vocabulary."""
    if conflict is None:
        return "indeterminate"
    if conflict <= cfg.conflict_aligned:
        return "aligned"
    if conflict <= cfg.conflict_mild:
        return "mild_conflict"
    return "strong_conflict"


# Reliability


def _reliability(
    branch: BranchResult,
    cfg: FusionConfig,
    labels: tuple[str, ...],
) -> np.ndarray:
    """Per-class reliability vector ``r_m(c)``."""
    prior = cfg.prior(branch.modality)
    rho = np.asarray([float(prior.get(label, 1.0)) for label in labels], dtype=np.float64)
    coverage_term = float(max(branch.coverage, 0.0)) ** cfg.gamma
    confidence_term = float(max(branch.confidence, 0.0)) ** cfg.delta
    return rho * coverage_term * confidence_term


# Fusion


def fuse(
    acoustic: BranchResult,
    text: BranchResult,
    cfg: FusionConfig,
    *,
    text_evidence: float | None = None,
) -> FusionOutput:
    """Fuse two branch results into one distribution.

    Returns probabilities, entropy, ignorance and the conflict index. ``text_evidence``
    scales the text branch's reliability; below ``withhold_below`` the conflict index
    is reported as unknown rather than as a number.
    """
    labels = tuple(acoustic.mapped_probs.keys())
    if labels != tuple(text.mapped_probs.keys()):
        raise ValueError("branch probability dictionaries must share the same key order")

    p_a = _as_array(acoustic.mapped_probs, labels)
    p_t = _as_array(text.mapped_probs, labels)

    evidence = 1.0 if text_evidence is None else float(max(0.0, min(1.0, text_evidence)))
    evidence_gain = max(float(cfg.text_evidence_floor), evidence)

    if p_a.sum() <= 0.0 and p_t.sum() <= 0.0:
        # Both branches empty: fall back to the config prior rather than crash.
        uniform = 1.0 / len(labels)
        probs = {label: uniform for label in labels}
        return FusionOutput(
            probs=probs,
            per_class_weights={"acoustic": dict(probs), "text": dict(probs)},
            alpha_effective=0.5,
            conflict_index=None,
            agreement="indeterminate",
            ignorance=1.0,
            entropy=1.0,
        )

    mode = (cfg.mode or "credal").lower()

    if mode == "weighted":
        alpha = float(min(max(cfg.alpha, 0.0), 1.0))
        # The baseline deliberately ignores coverage and per-class priors: that is
        # what makes it a fair ablation target.
        combined = alpha * p_a + (1.0 - alpha) * p_t
        r_a = np.full(len(labels), alpha)
        r_t = np.full(len(labels), 1.0 - alpha)
        alpha_effective = alpha
    elif mode == "credal":
        r_a = _reliability(acoustic, cfg, labels) + cfg.epsilon
        r_t = (_reliability(text, cfg, labels) * evidence_gain) + cfg.epsilon
        combined = r_a * p_a + r_t * p_t
        denom = float(r_a.sum() + r_t.sum())
        alpha_effective = float(r_a.sum() / denom) if denom > 0 else 0.5
    else:
        raise ValueError(f"unknown fusion mode {cfg.mode!r}; expected 'credal' or 'weighted'")

    total = float(combined.sum())
    if total <= _EPS:
        # Every class was structurally unreachable; use the per-class weights as a
        # prior-shaped fallback so the UI still has something honest to show.
        fallback = r_a + r_t
        total = float(fallback.sum())
        combined = fallback if total > 0 else np.ones(len(labels))
        total = float(combined.sum())

    fused = combined / total
    probs = {label: float(fused[i]) for i, label in enumerate(labels)}

    # -- conflict ---------------------------------------------------------- #
    if evidence < float(cfg.text_evidence_withhold):
        # The transcript says nothing about how the speaker feels, so a disagreement
        # between the channels is not a disagreement about affect at all. Withhold the
        # number rather than report a conflict the text never raised.
        conflict_raw: float | None = None
        conflict_norm: float | None = None
    elif acoustic.coverage < cfg.ignorance_threshold or text.coverage < cfg.ignorance_threshold:
        # A branch that cannot express half the target space has no standing to
        # disagree about a class. Report indeterminacy instead of a fake number.
        conflict_raw: float | None = None
        conflict_norm: float | None = None
    else:
        q_a = p_a / max(acoustic.coverage, _EPS)
        q_t = p_t / max(text.coverage, _EPS)
        conflict_raw = jensen_shannon(q_a, q_t)
        conflict_norm = float(min(1.0, conflict_raw / math.log(2.0)))

    # -- ignorance --------------------------------------------------------- #
    w_bar_a = float(r_a.mean())
    w_bar_t = float(r_t.mean())
    denom_w = w_bar_a + w_bar_t
    ignorance = (
        (w_bar_a * (1.0 - acoustic.coverage) + w_bar_t * (1.0 - text.coverage)) / denom_w
        if denom_w > 0
        else 0.0
    )

    return FusionOutput(
        probs=probs,
        per_class_weights={
            "acoustic": {label: float(r_a[i]) for i, label in enumerate(labels)},
            "text": {label: float(r_t[i]) for i, label in enumerate(labels)},
        },
        alpha_effective=float(alpha_effective),
        conflict_index=conflict_norm,
        agreement=agreement_level(conflict_norm, cfg),
        ignorance=float(max(0.0, min(1.0, ignorance))),
        entropy=normalised_entropy(fused),
    )
