"""Lexical evidence that a transcript really carries affect.

The text branch is confident on any string, including carrier sentences with no
affect in them, so fusion needs a measure of what the words actually say: affect
terms, self-reference, emphasis. Score is coarse and auditable by design.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

__all__ = ["EvidenceResult", "lexical_evidence", "SELF_REFERENCE"]

SELF_REFERENCE: frozenset[str] = frozenset(
    {"i", "i'm", "im", "i've", "ive", "i'd", "me", "my", "mine", "myself", "we", "we're", "our"}
)

INTENSIFIERS: frozenset[str] = frozenset(
    {"very", "really", "so", "extremely", "terribly", "totally", "completely", "quite", "too", "awfully"}
)

_WORD = re.compile(r"[a-z']+")


@dataclass(frozen=True)
class EvidenceResult:
    """A transcript's lexical evidence for carrying affect."""

    score: float
    self_reference: bool
    intensifier: bool
    emphasised: bool
    affect_hits: int
    matched_terms: tuple[str, ...]

    @property
    def is_silent(self) -> bool:
        """True when the transcript says nothing about how the speaker feels."""
        return self.score <= 0.0


def _lexicon_keys(lexicon: Mapping[str, object] | None, fallback: Sequence[str] | None) -> tuple[str, ...]:
    """Harvest affect terms from ``config/lexicon.yaml`` whatever its nesting.

    The pipeline hands over ``{category: [term, ...]}``; a richer lexicon might nest
    ``{category: {"weight": w, "terms": [...]}}``. Both are handled, because the gate
    must not silently become deaf if the lexicon file gains structure.
    """
    keys: list[str] = []
    if lexicon:
        for key, value in lexicon.items():
            if isinstance(value, Mapping):
                for term in value.get("terms") or ():  # type: ignore[union-attr]
                    keys.append(str(term).lower())
                continue
            if isinstance(value, (list, tuple, set)):
                keys.extend(str(term).lower() for term in value)  # type: ignore[union-attr]
                continue
            keys.append(str(key).lower())
    if not keys and fallback:
        keys.extend(str(t).lower() for t in fallback)
    return tuple(dict.fromkeys(keys))


def lexical_evidence(
    text: str,
    lexicon: Mapping[str, object] | None = None,
    *,
    fallback_terms: Sequence[str] | None = None,
    hits_for_full_credit: int = 3,
) -> EvidenceResult:
    """Score how much a transcript says about the speaker's state.

    Affect vocabulary, self-reference and emphasis, weighted 0.6 / 0.3 / 0.1, with a
    small bonus for intensifiers. Accepts either lexicon layout used in config.
    """
    lowered = (text or "").lower()
    tokens = _WORD.findall(lowered)
    if not tokens:
        return EvidenceResult(0.0, False, False, False, 0, ())

    keys = _lexicon_keys(lexicon, fallback_terms)
    matched: list[str] = []
    for key in keys:
        if " " in key:
            if key in lowered and key not in matched:
                matched.append(key)
        elif key in tokens and key not in matched:
            matched.append(key)

    self_reference = any(t in SELF_REFERENCE for t in tokens)
    intensifier = any(t in INTENSIFIERS for t in tokens)
    emphasised = "!" in (text or "")

    vocabulary = min(1.0, len(matched) / float(max(1, hits_for_full_credit)))
    score = 0.6 * vocabulary + 0.3 * float(self_reference) + 0.1 * float(emphasised)
    if intensifier and (vocabulary > 0.0 or self_reference):
        score += 0.05
    return EvidenceResult(
        score=float(max(0.0, min(1.0, score))),
        self_reference=self_reference,
        intensifier=intensifier,
        emphasised=emphasised,
        affect_hits=len(matched),
        matched_terms=tuple(matched),
    )
