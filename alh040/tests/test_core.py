"""
Unit tests for the parts of GrammarCoach that do not need the AI model.

Run:   python -m pytest tests/
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feedback import correctness_score, diff_html, find_edits, tokenize  # noqa: E402
from metrics import corpus_gleu, exact_match, sentence_gleu              # noqa: E402
import progress as prog                                                   # noqa: E402
import quiz                                                               # noqa: E402


def categories(original, corrected):
    return [e.category for e in find_edits(original, corrected)]


# --- feedback.py ---------------------------------------------------------
def test_tokenize_splits_punctuation():
    assert tokenize("I don't know.") == ["I", "don't", "know", "."]


def test_no_mistakes():
    assert find_edits("This is fine.", "This is fine.") == []
    assert correctness_score("This is fine.", "This is fine.") == 100


def test_verb_tense():
    assert categories("Yesterday I go home.", "Yesterday I went home.") == ["Verb tense"]
    assert categories("I buyed it.", "I bought it.") == ["Verb tense"]


def test_subject_verb_agreement():
    assert categories("He go to work.", "He goes to work.") == ["Subject-verb agreement"]
    assert categories("They is happy.", "They are happy.") == ["Subject-verb agreement"]


def test_articles_and_plurals():
    assert categories("I ate a apple.", "I ate an apple.") == ["Articles (a / an / the)"]
    assert categories("I have two book.", "I have two books.") == ["Singular / plural"]


def test_capitalization_spelling_confused_words():
    assert categories("i like it.", "I like it.") == ["Capitalization"]
    assert categories("See you tomorow.", "See you tomorrow.") == ["Spelling"]
    assert categories("Their going home.", "They're going home.") == ["Commonly confused words"]


def test_word_order_is_detected():
    assert categories("I like really music.", "I really like music.") == ["Word order"]


def test_diff_html_marks_changes():
    html = diff_html("I goes home.", "I go home.")
    assert '<del class="gc-del">goes</del>' in html
    assert '<ins class="gc-ins">go</ins>' in html


# --- metrics.py ----------------------------------------------------------
def test_gleu_identical_and_different():
    assert sentence_gleu("I went home.", ["I went home."]) == 1.0
    assert sentence_gleu("cats", ["I went home."]) == 0.0
    assert corpus_gleu(["I went home."], [["I went home."]]) == 1.0


def test_exact_match_ignores_case():
    assert exact_match("i went home .", ["I went home."])


# --- quiz.py -------------------------------------------------------------
def test_detokenize():
    assert quiz.detokenize("I do n't know , really .") == "I don't know, really."


def test_grade_perfect_and_unchanged():
    question = quiz.BUILT_IN_QUESTIONS[0]
    perfect = quiz.grade_answer(question, question["corrections"][0])
    assert perfect["score"] == 100 and perfect["remaining"] == []
    unchanged = quiz.grade_answer(question, question["sentence"])
    assert unchanged["unchanged"] and unchanged["score"] < 100


# --- progress.py ---------------------------------------------------------
def test_progress_counts_mistakes():
    p = prog.new_progress()
    p = prog.record_text(p, find_edits("He go home.", "He goes home."))
    p = prog.record_quiz(p, 80, [])
    assert p["mistakes"] == {"Subject-verb agreement": 1}
    assert p["texts_checked"] == 1 and p["quiz_scores"] == [80]
    assert "Subject-verb agreement" in prog.summary_markdown(p)
