"""Deterministic natural-language interpretation.

The interface explains its verdict in prose. An LLM is deliberately *not* used at
runtime: the app must work offline, cost nothing per request, and produce the same
sentence for the same input so that a figure in the report can be reproduced.

Instead this is a small template engine. It composes three sentence slots --
verdict, evidence, caveat -- from the actual numbers, and varies phrasing by a seed
derived from the request id so that repeated analyses do not read like a stuck
record while remaining fully reproducible.
"""

from __future__ import annotations

import random
from typing import Mapping, Sequence

from .schemas import Agreement, BranchResult, StressResult

__all__ = ["build_narrative", "summarise"]

_EMOTION_PHRASE = {
    "happy": "positive and animated",
    "sad": "low and subdued",
    "angry": "tense and activated",
    "fear": "apprehensive and activated",
    "disgust": "averse",
    "neutral": "even and unmarked",
}

_STRESS_PHRASE = {
    "Low": "little sign of strain",
    "Medium": "a moderate strain signal",
    "High": "a strong strain signal",
}

_VERDICT_TEMPLATES: Sequence[str] = (
    "Fused reading: {emotion} at {conf:.0%} confidence, with {stress_phrase} (stress {score:.2f}, {band}).",
    "Across both channels the state reads as {emotion} ({conf:.0%}), and the stress index sits at {score:.2f} - {band_lower}.",
    "The combined estimate is {emotion} with {conf:.0%} confidence; stress is {band_lower} at {score:.2f}.",
)

_ALIGNED_TEMPLATES: Sequence[str] = (
    "Both channels agree: the voice reads {audio} and the words read {text}, so the fused distribution stays close to each branch.",
    "The two modalities are consistent - {audio} acoustically, {text} textually - which is why the fused vector is sharply peaked.",
    "Agreement is high (conflict index {conflict:.2f}): vocal tone and verbal content point the same way.",
)

_MILD_TEMPLATES: Sequence[str] = (
    "There is mild disagreement: the voice leans {audio} while the words lean {text}, so the fused estimate is pulled between them (conflict {conflict:.2f}).",
    "The branches only partly align - {audio} from prosody, {text} from content - producing a moderately soft fused distribution.",
)

_STRONG_TEMPLATES: Sequence[str] = (
    "The channels conflict sharply (index {conflict:.2f}): the words read as {text}, but the vocal tone carries {audio}. "
    "The fused reading is {emotion}, and its confidence is deliberately lower than either branch's, because the evidence is not consistent.",
    "Strong cross-modal conflict (index {conflict:.2f}). Verbatim content sounds {text}, whereas prosody indicates {audio}; "
    "the system reports {emotion} while discounting both branches.",
)

_INDETERMINATE_TEMPLATES: Sequence[str] = (
    "A conflict index is not reported: the acoustic branch can only express {n_ac} of {n_total} target emotions, "
    "so it has no standing to disagree about the rest. Coverage is carried into the fusion as ignorance instead.",
)

_CAVEAT_LOW_COVERAGE = (
    "Acoustic coverage is {cov:.0%} - this head was trained on a four-class inventory, so fear and disgust are "
    "left as ignorance rather than guessed."
)

_CAVEAT_BORDERLINE = "The stress score sits within {band:.2f} of a band threshold, so the band is qualified as borderline."

_CAVEAT_LOW_ASR = "Transcript confidence is only {asr:.2f}; the text branch was down-weighted accordingly."

_CAVEAT_IGNORANCE = "Residual ignorance in the fused estimate is {ign:.0%}."


def _fmt(template: str, **kwargs: object) -> str:
    try:
        return template.format(**kwargs)
    except (KeyError, ValueError):  # pragma: no cover - template authoring guard
        return template


def build_narrative(
    *,
    request_id: str,
    fused_probs: Mapping[str, float],
    acoustic: BranchResult,
    text: BranchResult,
    stress: StressResult,
    agreement: Agreement,
    conflict_index: float | None,
    ignorance: float,
    asr_confidence: float,
    n_canonical: int = 6,
) -> tuple[str, str]:
    """Compose the long interpretive note and its one-line summary.

    Returns
    -------
    (long, short)
        ``long`` is two to four sentences for the callout box; ``short`` is a
        single clause suitable for the result header.
    """
    rng = random.Random(abs(hash((request_id, round(stress.score, 3)))) % (2**32))

    ordered = sorted(fused_probs.items(), key=lambda kv: kv[1], reverse=True)
    emotion, confidence = ordered[0][0], float(ordered[0][1])
    runner_up, runner_p = (ordered[1][0], float(ordered[1][1])) if len(ordered) > 1 else (emotion, 0.0)

    verdict = _fmt(
        rng.choice(_VERDICT_TEMPLATES),
        emotion=emotion,
        conf=confidence,
        score=stress.score,
        band=stress.band,
        band_lower=stress.band.lower(),
        stress_phrase=_STRESS_PHRASE.get(stress.band, "an unclear strain signal"),
    )

    audio_top = acoustic.top_label
    text_top = text.top_label
    n_ac = int(round(acoustic.coverage * n_canonical)) or 4

    if agreement == "indeterminate":
        evidence = _fmt(
            rng.choice(_INDETERMINATE_TEMPLATES),
            n_ac=n_ac,
            n_total=n_canonical,
        )
        # Substituting a concrete count is clearer than a percentage of classes
        # the branch cannot even name.
        evidence = (
            f"A conflict index is not reported: the acoustic branch was trained on a four-class inventory "
            f"({', '.join(sorted(acoustic.raw_labels))}), so it has no standing to disagree about "
            f"{n_canonical - n_ac} of the {n_canonical} target emotions. Its unmapped mass is carried into the "
            f"fusion as ignorance instead."
        )
    elif agreement == "aligned":
        evidence = _fmt(
            rng.choice(_ALIGNED_TEMPLATES),
            audio=audio_top,
            text=text_top,
            conflict=conflict_index or 0.0,
        )
    elif agreement == "mild_conflict":
        evidence = _fmt(
            rng.choice(_MILD_TEMPLATES),
            audio=audio_top,
            text=text_top,
            conflict=conflict_index or 0.0,
        )
    else:
        evidence = _fmt(
            rng.choice(_STRONG_TEMPLATES),
            audio=audio_top,
            text=text_top,
            emotion=emotion,
            conflict=conflict_index or 0.0,
        )

    caveats: list[str] = []
    if acoustic.coverage < 0.99:
        caveats.append(_CAVEAT_LOW_COVERAGE.format(cov=acoustic.coverage))
    if "borderline" in stress.band_label:
        caveats.append(_CAVEAT_BORDERLINE.format(band=0.05))
    if asr_confidence < 0.5:
        caveats.append(_CAVEAT_LOW_ASR.format(asr=asr_confidence))
    if ignorance > 0.25:
        caveats.append(_CAVEAT_IGNORANCE.format(ign=ignorance))

    parts = [verdict, evidence]
    if caveats:
        parts.append(" ".join(caveats[:2]))

    long_text = " ".join(parts)
    short_text = (
        f"{emotion.capitalize()} ({confidence:.0%}) · {stress.band_label} stress {stress.score:.2f}"
        f" · {_agreement_word(agreement)}"
    )
    return long_text, short_text


def _agreement_word(agreement: Agreement) -> str:
    return {
        "aligned": "channels aligned",
        "mild_conflict": "mild conflict",
        "strong_conflict": "strong conflict",
        "indeterminate": "conflict indeterminate",
    }.get(agreement, agreement)


def summarise(result_text: str, limit: int = 180) -> str:
    """Trim a narrative to a card-sized preview."""
    text = " ".join(result_text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
