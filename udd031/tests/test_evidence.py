"""Tests for the transcript evidence gate (``mmsa.evidence``)."""

from __future__ import annotations

import pytest

from mmsa.evidence import lexical_evidence

LEXICON = {
    "negative": ["stress", "stressed", "anxious", "exhausted"],
    "positive": ["happy", "calm"],
}


def test_neutral_carrier_sentence_has_no_evidence():
    """The RAVDESS carrier sentence must score zero: it says nothing about affect."""
    result = lexical_evidence("Kids are talking by the door", LEXICON)
    assert result.score == 0.0
    assert result.affect_hits == 0
    assert not result.self_reference
    assert result.is_silent


def test_affect_vocabulary_alone_raises_evidence():
    result = lexical_evidence("feeling anxious again", LEXICON)
    assert result.affect_hits >= 1
    assert result.score >= 0.15


def test_self_reference_and_vocabulary_score_higher_than_vocabulary_alone():
    plain = lexical_evidence("stressed", LEXICON)
    personal = lexical_evidence("I am really stressed", LEXICON)
    assert personal.score > plain.score
    assert personal.intensifier


def test_emphasis_adds_evidence_but_cannot_saturate_alone():
    shouted = lexical_evidence("EXAMS!!!", LEXICON)
    assert shouted.emphasised
    assert 0.0 < shouted.score <= 0.1


def test_score_is_bounded_and_empty_text_is_silent():
    for text in ("", "   ", "!!!", "I feel exhausted and anxious and stressed beyond belief"):
        result = lexical_evidence(text, LEXICON)
        assert 0.0 <= result.score <= 1.0
    assert lexical_evidence("", LEXICON).is_silent


def test_lexicon_given_as_nested_mapping_is_understood():
    nested = {"negative": {"weight": 1.0, "terms": ["overwhelmed"]}}
    result = lexical_evidence("I feel overwhelmed", nested)
    assert result.matched_terms == ("overwhelmed",)


def test_fallback_terms_used_when_lexicon_is_empty():
    result = lexical_evidence("I am stressed", None, fallback_terms=["stressed"])
    assert result.affect_hits == 1


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I'm fine.", True),
        ("Dogs are sitting by the door", False),
    ],
)
def test_self_reference_detection(text, expected):
    assert lexical_evidence(text, LEXICON).self_reference is expected
