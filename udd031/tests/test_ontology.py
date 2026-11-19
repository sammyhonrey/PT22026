"""Tests for the label-ontology adapter."""

from __future__ import annotations

import numpy as np
import pytest

from mmsa.ontology import Ontology, OntologyError
from mmsa.schemas import CANONICAL_LABELS


def test_canonical_order_is_fixed(ontology):
    assert ontology.canonical == CANONICAL_LABELS
    assert ontology.canonical == ("happy", "sad", "angry", "fear", "disgust", "neutral")


def test_native_vocabularies(ontology):
    assert ontology.native_labels("acoustic") == ("ang", "hap", "neu", "sad")
    assert ontology.native_labels("text") == (
        "joy",
        "love",
        "anger",
        "sadness",
        "fear",
        "surprise",
    )


def test_all_rows_sum_to_at_most_one(ontology):
    for branch in ("acoustic", "text"):
        matrix = ontology.matrix(branch)
        row_sums = matrix.sum(axis=1)
        assert np.all(row_sums <= 1.0 + 1e-9), f"{branch} has an over-claiming row"


def test_acoustic_mapping_is_identity(ontology):
    mapped, coverage = ontology.map_probs("acoustic", ["ang", "hap", "neu", "sad"], [1.0, 0.0, 0.0, 0.0])
    assert mapped["angry"] == pytest.approx(1.0)
    assert coverage == pytest.approx(1.0, abs=1e-9)
    assert mapped["fear"] == pytest.approx(0.0)
    assert mapped["disgust"] == pytest.approx(0.0)


def test_love_splits_between_happy_and_neutral(ontology):
    mapped, coverage = ontology.map_probs(
        "text",
        ["joy", "love", "anger", "sadness", "fear", "surprise"],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    )
    assert mapped["happy"] == pytest.approx(0.65)
    assert mapped["neutral"] == pytest.approx(0.35)
    assert coverage == pytest.approx(1.0)


def test_surprise_is_ambiguous(ontology):
    mapped, _ = ontology.map_probs(
        "text",
        ["joy", "love", "anger", "sadness", "fear", "surprise"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    )
    assert mapped["fear"] == pytest.approx(0.35)
    assert mapped["happy"] == pytest.approx(0.30)
    assert mapped["neutral"] == pytest.approx(0.35)
    assert sum(mapped.values()) == pytest.approx(1.0)


def test_coverage_of_target_space(ontology):
    # The acoustic head can express four of six canonical classes.
    assert ontology.coverage_of_target_space("acoustic") == pytest.approx(4 / 6)
    assert ontology.coverage_of_target_space("text") == pytest.approx(5 / 6)


def test_acoustic_cannot_reach_fear(ontology):
    assert not ontology.is_supported("acoustic", "fear")
    assert not ontology.is_supported("acoustic", "disgust")
    assert ontology.is_supported("acoustic", "angry")


def test_unknown_label_raises(ontology):
    with pytest.raises(OntologyError):
        ontology.map_probs("acoustic", ["joy", "hap", "neu", "sad"], [1.0, 0.0, 0.0, 0.0])


def test_reordered_labels_are_realigned(ontology):
    # Same set, different order: the adapter must reorder rather than mis-map.
    mapped, _ = ontology.map_probs("acoustic", ["hap", "ang", "neu", "sad"], [1.0, 0.0, 0.0, 0.0])
    assert mapped["happy"] == pytest.approx(1.0)
    assert mapped["angry"] == pytest.approx(0.0)


def test_all_zero_distribution_raises(ontology):
    with pytest.raises(OntologyError):
        ontology.map_probs("acoustic", ["ang", "hap", "neu", "sad"], [0.0, 0.0, 0.0, 0.0])


def test_renormalise_restores_unit_sum(ontology):
    mapped, coverage = ontology.map_probs("acoustic", ["ang", "hap", "neu", "sad"], [0.1, 0.2, 0.3, 0.4])
    proper = Ontology.renormalise(mapped)
    assert sum(proper.values()) == pytest.approx(1.0)
    assert coverage == pytest.approx(1.0)


def test_row_over_one_is_rejected():
    with pytest.raises(OntologyError):
        Ontology(
            canonical=["happy", "sad"],
            matrices={"acoustic": {"hap": {"happy": 1.2}}},
        )
