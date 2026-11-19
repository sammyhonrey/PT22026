"""Regression tests for the ASR confidence estimator.

The estimator has no model dependency beyond tensor arithmetic, so it is tested
here with a hand-built ``generate`` output. The case that matters is the forced
decoder prefix Whisper prepends: those tokens appear in ``sequences`` but not in
``scores``, and aligning the two from the start of the sequence reports a
probability near zero for a transcript the decoder was actually certain about.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from mmsa.asr import TransformersWhisperBackend  # noqa: E402


class _Generated:
    """Minimal stand-in for ``generate(return_dict_in_generate=True)``."""

    def __init__(self, sequences: torch.Tensor, scores: tuple[torch.Tensor, ...]) -> None:
        self.sequences = sequences
        self.scores = scores


def _output(prefix: tuple[int, ...], tokens: tuple[int, ...], vocab: int = 8) -> _Generated:
    """Build an output whose scored tokens are ``tokens`` behind ``prefix``."""
    logits = torch.zeros(1, vocab)
    logits[0, 2] = 6.0  # almost all mass on token 2
    scores = tuple(logits.clone() for _ in tokens)
    sequences = torch.tensor([[*prefix, *tokens]], dtype=torch.long)
    return _Generated(sequences, scores)


def test_forced_prefix_does_not_destroy_confidence():
    # Two forced decoder tokens, eight scored tokens, all scored token id 2.
    generated = _output(prefix=(50_257, 50_362), tokens=(2,) * 8)
    confidence = TransformersWhisperBackend._token_confidence(generated, model=None)

    expected = math.exp(6.0) / (math.exp(6.0) + 7.0)
    assert confidence == pytest.approx(expected, rel=1e-4)
    assert confidence > 0.98


def test_higher_entropy_yields_lower_confidence():
    confident = _output(prefix=(1,), tokens=(2,) * 6)
    flat = torch.zeros(1, 8)
    flat[0, 2] = 0.1
    uncertain = _Generated(confident.sequences, tuple(flat.clone() for _ in range(6)))

    assert TransformersWhisperBackend._token_confidence(confident, model=None) > (
        TransformersWhisperBackend._token_confidence(uncertain, model=None)
    )


def test_missing_scores_falls_back_to_neutral_value():
    generated = _Generated(torch.tensor([[1, 2, 3]]), ())
    assert TransformersWhisperBackend._token_confidence(generated, model=None) == 0.5
