
import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Normalize English text before converting it into character IDs.
    """

    text = text.strip()

    # Convert smart punctuation to standard ASCII punctuation.
    replacements = {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "…": "...",
    }

    for original, replacement in replacements.items():
        text = text.replace(original, replacement)

    # Remove square brackets but preserve the words inside them.
    text = text.replace("[", " ")
    text = text.replace("]", " ")

    # Convert accented characters to their basic ASCII equivalents.
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Standardize whitespace and capitalization.
    text = re.sub(r"\s+", " ", text)
    return text.lower()


# Keep the original character order so existing character IDs do not change.
CHARACTERS = " abcdefghijklmnopqrstuvwxyz'.,!?-\";:()"

CHAR_TO_ID = {
    character: index
    for index, character in enumerate(CHARACTERS)
}


def text_to_sequence(text: str) -> list[int]:
    """
    Convert normalized text into a list of numerical character IDs.
    """

    text = normalize_text(text)

    unsupported = get_unsupported_characters(text)

    if unsupported:
        raise ValueError(
            f"Unsupported characters: {unsupported}"
        )

    return [
        CHAR_TO_ID[character]
        for character in text
    ]


def sequence_to_text(sequence: list[int]) -> str:
    """
    Convert numerical character IDs back into text.
    """

    id_to_char = {
        index: character
        for index, character in enumerate(CHARACTERS)
    }

    characters = []

    for index in sequence:
        if index not in id_to_char:
            raise ValueError(
                f"Invalid character ID: {index}"
            )

        characters.append(id_to_char[index])

    return "".join(characters)


def get_unsupported_characters(text: str) -> list[str]:
    """
    Return characters that are not supported by the vocabulary.
    """

    text = normalize_text(text)

    return sorted({
        character
        for character in text
        if character not in CHAR_TO_ID
    })
