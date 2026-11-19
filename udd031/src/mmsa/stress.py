"""Transparent stress composite; deliberately not a classifier.

    z     = b + sum_i w_i * x_i
    score = sigmoid(z)

Contributors plus the bias equal z exactly, which the tests assert. No corpus
labels stress levels, so only construct validity is claimed.
"""

from __future__ import annotations

import math
from typing import Mapping

from .config import StressConfig
from .schemas import DimensionalResult, StressResult

__all__ = ["compute_stress", "sigmoid", "robust_z"]

_EPS = 1e-9


def sigmoid(z: float) -> float:
    """Numerically safe logistic function."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-min(z, 60.0)))
    e = math.exp(max(z, -60.0))
    return e / (1.0 + e)


def robust_z(value: float, median: float, iqr: float) -> float:
    """Robust standard score against (median, IQR) reference statistics."""
    if not math.isfinite(value):
        return 0.0
    if iqr is None or iqr <= _EPS:
        return 0.0
    return float(max(-3.0, min(3.0, (value - median) / iqr)))


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _proxy_affect(fused_probs: Mapping[str, float], cfg: StressConfig) -> tuple[float, float]:
    """Valence in [-1, 1] and arousal in [0, 1] from the fused emotion vector."""
    valence = sum(cfg.proxy_valence.get(k, 0.0) * v for k, v in fused_probs.items())
    arousal_sum = sum(cfg.proxy_arousal.get(k, 0.0) * v for k, v in fused_probs.items())
    return _clip(valence, -1.0, 1.0), _clip(0.5 + 0.5 * arousal_sum, 0.0, 1.0)


def _prosody_instability(
    features: Mapping[str, float],
    cfg: StressConfig,
    reference: Mapping[str, Mapping[str, float]] | None,
) -> float:
    """Mean robust z-score of the configured instability features, in [-1, 1]."""
    if not reference:
        return 0.0
    zs: list[float] = []
    for name in cfg.instability_features:
        if name not in features:
            continue
        stats = reference.get(name) or {}
        z = robust_z(
            float(features[name]),
            float(stats.get("median", 0.0)),
            float(stats.get("iqr", 0.0)),
        )
        zs.append(z / 3.0)  # map the [-3, 3] clip onto [-1, 1]
    if not zs:
        return 0.0
    return _clip(sum(zs) / len(zs), -1.0, 1.0)


def compute_stress(
    fused_probs: Mapping[str, float],
    *,
    cfg: StressConfig,
    features: Mapping[str, float],
    asr_confidence: float,
    lexical_negative: float,
    dimensional: DimensionalResult | None = None,
    prosody_reference: Mapping[str, Mapping[str, float]] | None = None,
) -> StressResult:
    """Score the seven-term composite and return it with the decomposition.

    ``contributors`` are signed logit-space terms that sum with ``bias`` to ``z``.
    """
    if dimensional is not None:
        # audeering head outputs roughly [0, 1]; map both onto the same scales as
        # the proxy so the weights remain interpretable and interchangeable.
        valence = _clip(2.0 * dimensional.valence - 1.0, -1.0, 1.0)
        arousal = _clip(dimensional.arousal, 0.0, 1.0)
        source = "dimensional"
    else:
        valence, arousal = _proxy_affect(fused_probs, cfg)
        source = "proxy"

    p_angry = float(fused_probs.get("angry", 0.0))
    p_fear = float(fused_probs.get("fear", 0.0))
    p_happy = float(fused_probs.get("happy", 0.0))

    instability = _prosody_instability(features, cfg, prosody_reference)

    x: dict[str, float] = {
        "negative_valence": _clip(-valence, -1.0, 1.0),
        "arousal": arousal,
        "threat_emotion": _clip(p_angry + p_fear, 0.0, 1.0),
        "positive_emotion": _clip(p_happy, 0.0, 1.0),
        "prosody_instability": instability,
        "lexical_negative": _clip(lexical_negative, 0.0, 1.0),
        "asr_uncertainty": _clip(1.0 - asr_confidence, 0.0, 1.0),
    }

    contributors: dict[str, float] = {}
    z = cfg.bias
    for name, value in x.items():
        weight = float(cfg.weights.get(name, 0.0))
        term = weight * value
        contributors[name] = round(term, 6)
        z += term

    score = sigmoid(z)

    if score < cfg.threshold_low:
        band: str = "Low"
    elif score > cfg.threshold_high:
        band = "High"
    else:
        band = "Medium"

    band_label = _band_label(score, band, cfg)

    features_out: dict[str, float] = {
        "valence": round(valence, 4),
        "arousal": round(arousal, 4),
        "prosody_instability": round(instability, 4),
        "lexical_negative": round(lexical_negative, 4),
        "asr_confidence": round(asr_confidence, 4),
        "affect_source": 1.0 if source == "dimensional" else 0.0,
        "logit": round(z, 4),
    }
    for name, value in x.items():
        features_out[f"x_{name}"] = round(value, 4)

    return StressResult(
        score=round(float(score), 6),
        band=band,  # type: ignore[arg-type]
        band_label=band_label,
        contributors=contributors,
        features=features_out,
    )


def _band_label(score: float, band: str, cfg: StressConfig) -> str:
    """Add a borderline qualifier when the score sits near a threshold."""
    band_width = cfg.borderline_band
    near_low = abs(score - cfg.threshold_low) < band_width
    near_high = abs(score - cfg.threshold_high) < band_width

    if band == "Medium" and near_low:
        return "Medium (borderline low)"
    if band == "Medium" and near_high:
        return "Medium (borderline high)"
    if band == "Low" and near_low:
        return "Low (borderline medium)"
    if band == "High" and near_high:
        return "High (borderline medium)"
    return band
