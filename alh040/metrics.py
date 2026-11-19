"""
metrics.py - Scores used to evaluate grammar correction.

GLEU (Wu et al., 2016 - the "Google BLEU" variant)
--------------------------------------------------
Compare the words of the AI's sentence with a human correction:
  - count every 1-, 2-, 3- and 4-word sequence (n-gram) in both,
  - precision = matching n-grams / n-grams in the AI's sentence,
  - recall    = matching n-grams / n-grams in the human sentence,
  - GLEU      = min(precision, recall), between 0 and 1 (1 = identical).
With several human corrections we use the one that gives the best score.

Note: the official JFLEG benchmark uses a slightly different GLEU (Napoles et
al., 2015) that also penalises copying mistakes from the source sentence, so
our numbers are not directly comparable with published leaderboards - but they
are a fair way to compare "before" vs "after" correction in this project.
"""

from collections import Counter

from feedback import tokenize


def _ngrams(tokens: list, max_n: int = 4) -> Counter:
    grams = Counter()
    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            grams[tuple(tokens[i:i + n])] += 1
    return grams


def _gleu_counts(hypothesis: str, reference: str) -> tuple:
    hyp = _ngrams([t.lower() for t in tokenize(hypothesis)])
    ref = _ngrams([t.lower() for t in tokenize(reference)])
    matches = sum((hyp & ref).values())
    return matches, max(sum(hyp.values()), sum(ref.values()))


def sentence_gleu(hypothesis: str, references: list) -> float:
    """GLEU of one sentence against its best-matching reference."""
    best = 0.0
    for ref in references:
        matches, total = _gleu_counts(hypothesis, ref)
        if total:
            best = max(best, matches / total)
    return best


def corpus_gleu(hypotheses: list, references_list: list) -> float:
    """GLEU over a whole dataset: total matches / total n-grams (best reference per sentence)."""
    total_matches, total_ngrams = 0, 0
    for hyp, refs in zip(hypotheses, references_list):
        best = max((_gleu_counts(hyp, ref) for ref in refs),
                   key=lambda mt: mt[0] / mt[1] if mt[1] else 0.0)
        total_matches += best[0]
        total_ngrams += best[1]
    return total_matches / total_ngrams if total_ngrams else 0.0


def exact_match(hypothesis: str, references: list) -> bool:
    """True if the sentence equals one of the references (ignoring case and spacing)."""
    hyp = [t.lower() for t in tokenize(hypothesis)]
    return any(hyp == [t.lower() for t in tokenize(ref)] for ref in references)
