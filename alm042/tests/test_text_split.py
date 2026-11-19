"""Tests for the sentence-aware text splitter."""

from __future__ import annotations

import pytest

from voiceforge.text_split import MAX_LONG_TEXT_CHARS, split_text


def test_splits_at_sentence_boundaries():
    text = "First sentence. Second sentence! Third sentence?"

    chunks = split_text(text, 30)

    assert chunks == ["First sentence.", "Second sentence!", "Third sentence?"]


def test_punctuation_is_preserved():
    text = "Wait... what? Yes; really! Fine."

    chunks = split_text(text, 100)

    assert chunks == [text]


def test_packs_short_sentences_together():
    text = "One. Two. Three."

    chunks = split_text(text, 10)

    assert chunks == ["One. Two.", "Three."]

def test_respects_maximum_length():
    text = " ".join(["Sentence number %d is here." % index for index in range(40)])

    chunks = split_text(text, 60)

    assert chunks
    assert all(len(chunk) <= 60 for chunk in chunks)


def test_never_drops_words_or_changes_order():
    text = (
        "The first paragraph has two sentences. Here is the second one.\n\n"
        "A new paragraph starts here, with more words. And it ends."
    )

    chunks = split_text(text, 40)

    assert " ".join(chunks).split() == text.split()


def test_no_empty_chunks():
    text = "A. B. C. D. E."

    chunks = split_text(text, 25)

    assert all(chunk.strip() for chunk in chunks)


def test_empty_and_whitespace_text_returns_empty_list():
    assert split_text("", 50) == []
    assert split_text("   \n\t ", 50) == []


def test_paragraphs_and_line_breaks_are_respected():
    text = "Line one.\nLine two.\n\nNew paragraph here."

    chunks = split_text(text, 200)

    assert " ".join(chunks).split() == text.split()


def test_abbreviations_do_not_end_sentences():
    text = "Dr. Smith met Prof. Jones at 3 p.m. They talked for a while."

    chunks = split_text(text, 200)

    assert chunks[0].startswith("Dr. Smith")
    assert chunks == [text]


def test_initials_do_not_end_sentences():
    text = "J. R. R. Tolkien wrote the book. It is long."

    chunks = split_text(text, 200)

    assert chunks == [text]


def test_decimals_do_not_end_sentences():
    text = "The value is 3.14159 and it is useful."

    chunks = split_text(text, 200)

    assert chunks == [text]


def test_long_sentence_falls_back_to_whitespace():
    sentence = "word " * 40 + "end."
    text = sentence.strip()

    chunks = split_text(text, 50)

    assert len(chunks) > 1
    assert all(len(chunk) <= 50 for chunk in chunks)
    assert " ".join(chunks).split() == text.split()


def test_single_giant_word_is_cut_not_dropped():
    word = "a" * 130

    chunks = split_text(word, 40)

    assert all(len(chunk) <= 40 for chunk in chunks)
    assert "".join(chunks) == word


def test_unicode_text_is_not_corrupted():
    text = "Café naïve résumé — “quoted” text. Second sentence with 日本語."

    chunks = split_text(text, 30)

    assert "".join(chunks).replace(" ", "") != ""
    for chunk in chunks:
        assert chunk == chunk.strip()
        assert len(chunk) <= 30

    # No broken surrogate pairs or replacement characters.
    assert "\ufffd" not in "".join(chunks)


def test_non_string_input_is_rejected():
    with pytest.raises(TypeError):
        split_text(None, 50)


def test_invalid_max_chars_is_rejected():
    with pytest.raises(ValueError):
        split_text("Hello.", 5)


def test_maximum_long_text_constant_is_sane():
    assert MAX_LONG_TEXT_CHARS >= 10000
