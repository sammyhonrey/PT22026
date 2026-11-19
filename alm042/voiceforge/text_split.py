"""
Sentence-aware text splitting for long-form synthesis.

Engines cannot safely consume an arbitrarily long string, so long text is
split into chunks at sentence boundaries and synthesised chunk by chunk.

The splitter never drops text: joining the chunks with a single space
reproduces the original word sequence.
"""

from __future__ import annotations

import re

#: Longest text accepted for one long-text/batch item.
MAX_LONG_TEXT_CHARS = 20000

#: Shortest chunk size the splitter accepts (below this progress is silly).
MIN_CHUNK_CHARS = 10

#: Common abbreviations that must not end a sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "eg", "ie", "fig", "figs", "no", "vol", "approx", "inc", "ltd",
    "co", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
    "sept", "oct", "nov", "dec", "e.g", "i.e", "a.m", "p.m", "u.s",
}

_TOKEN_BEFORE_DOT = re.compile(r"([A-Za-z][A-Za-z.]*)$")


def _is_abbreviation(text: str, token_start: int, dot_index: int) -> bool:
    """True when the dot at ``dot_index`` closes an abbreviation or initial."""

    token = text[token_start:dot_index].strip()

    match = _TOKEN_BEFORE_DOT.search(token)

    if not match:
        return False

    word = match.group(1).lower().rstrip(".")

    if not word:
        return False

    # Single initials such as "J. R. R. Tolkien".
    if len(word) == 1:
        return True

    return word in _ABBREVIATIONS


def _split_paragraph(paragraph: str) -> list[str]:
    """Split one paragraph into sentences, keeping punctuation."""

    sentences: list[str] = []
    start = 0
    index = 0
    length = len(paragraph)

    while index < length:
        character = paragraph[index]

        if character == "\n":
            segment = paragraph[start:index].strip()

            if segment:
                sentences.append(segment)

            start = index + 1
            index += 1
            continue

        if character in ".!?\u2026":
            end = index

            # Consume runs such as "?!" or "..."
            while end + 1 < length and paragraph[end + 1] in ".!?\u2026":
                end += 1

            boundary_end = end + 1
            following = paragraph[boundary_end:boundary_end + 1]

            previous_is_digit = index > 0 and paragraph[index - 1].isdigit()
            next_is_digit = following.isdigit()

            # "3.14" is a decimal number, not a sentence break.
            is_decimal = character == "." and previous_is_digit and next_is_digit

            is_abbreviation = character == "." and _is_abbreviation(
                paragraph, start, index
            )

            # A boundary needs whitespace or end-of-string after it.
            at_boundary = boundary_end >= length or following in " \t\n"

            if at_boundary and not is_decimal and not is_abbreviation:
                segment = paragraph[start:boundary_end].strip()

                if segment:
                    sentences.append(segment)

                start = boundary_end

                while start < length and paragraph[start] in " \t":
                    start += 1

                index = start
                continue

            index = boundary_end
            continue

        index += 1

    tail = paragraph[start:].strip()

    if tail:
        sentences.append(tail)

    return sentences


def _wrap_long_segment(segment: str, max_chars: int) -> list[str]:
    """
    Split a segment that is too long, preferring spaces.

    A single word longer than ``max_chars`` is cut into pieces so the limit
    is never exceeded and no characters are lost.
    """

    if len(segment) <= max_chars:
        return [segment]

    pieces: list[str] = []
    current = ""

    for word in segment.split():

        while len(word) > max_chars:
            if current:
                pieces.append(current)
                current = ""

            pieces.append(word[:max_chars])
            word = word[max_chars:]

        if not word:
            continue

        if not current:
            current = word
        elif len(current) + 1 + len(word) <= max_chars:
            current = f"{current} {word}"
        else:
            pieces.append(current)
            current = word

    if current:
        pieces.append(current)

    return pieces or [segment[:max_chars]]


def split_text(text: object, max_chars: int) -> list[str]:
    """
    Split ``text`` into chunks of at most ``max_chars`` characters.

    Sentences are kept whole whenever they fit, paragraphs and line breaks
    are respected, and a sentence that is still too long is broken at
    whitespace. Returns ``[]`` for empty or whitespace-only input.
    """

    if not isinstance(text, str):
        raise TypeError("Text must be a string")

    if not isinstance(max_chars, int) or max_chars < MIN_CHUNK_CHARS:
        raise ValueError(
            f"max_chars must be an integer of at least {MIN_CHUNK_CHARS}"
        )

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    if not normalized:
        return []

    paragraphs = re.split(r"\n\s*\n", normalized)

    sentences: list[str] = []

    for paragraph in paragraphs:
        sentences.extend(_split_paragraph(paragraph))

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        for piece in _wrap_long_segment(sentence, max_chars):
            if not current:
                current = piece
            elif len(current) + 1 + len(piece) <= max_chars:
                current = f"{current} {piece}"
            else:
                chunks.append(current)
                current = piece

    if current:
        chunks.append(current)

    return [chunk for chunk in chunks if chunk.strip()]
