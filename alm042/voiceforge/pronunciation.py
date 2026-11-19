"""
Custom pronunciation rules applied before synthesis.

Kokoro phonemises whatever text it receives through misaki/espeak, so the
reliable integration point is the text itself: a matched word or phrase is
replaced with the user's phonetic respelling, and the engine then converts
that respelling to phonemes as usual.

This is deliberately a single-pass substitution:
- no recursive replacement (a replacement is never scanned again),
- longest match wins,
- whole-word / whole-phrase matches only,
- punctuation around a match is preserved.
"""

from __future__ import annotations

import re

#: Safety cap on the number of rules applied in one pass.
MAX_RULES = 500


def apply_dictionary(
    text: str,
    entries: list[dict] | None,
) -> tuple[str, int]:
    """
    Replace dictionary words in ``text``.

    Returns ``(new_text, replacements)``. ``entries`` items need ``word``
    and ``pronunciation`` keys; anything else is ignored.
    """

    if not text or not entries:
        return text, 0

    rules: list[tuple[str, str]] = []

    for entry in entries[:MAX_RULES]:
        if not isinstance(entry, dict):
            continue

        word = entry.get("word")
        pronunciation = entry.get("pronunciation")

        if (
            isinstance(word, str)
            and isinstance(pronunciation, str)
            and word.strip()
            and pronunciation.strip()
        ):
            rules.append((word.strip(), pronunciation.strip()))

    if not rules:
        return text, 0

    # Longest first so "New York" wins over "New".
    rules.sort(key=lambda rule: len(rule[0]), reverse=True)

    lookup = {word.lower(): pronunciation for word, pronunciation in rules}

    pattern = re.compile(
        r"(?<!\w)(" + "|".join(re.escape(word) for word, _ in rules) + r")(?!\w)",
        re.IGNORECASE,
    )

    replacements = 0

    def substitute(match: re.Match[str]) -> str:
        nonlocal replacements

        replacement = lookup.get(match.group(0).lower())

        if replacement is None:
            return match.group(0)

        replacements += 1

        return replacement

    return pattern.sub(substitute, text), replacements
