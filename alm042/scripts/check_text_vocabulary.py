
import argparse
import csv
from collections import Counter
from pathlib import Path

from voiceforge.text import CHAR_TO_ID, normalize_text


def main():
    parser = argparse.ArgumentParser(
        description="Check LJSpeech text against VoiceForge vocabulary."
    )

    parser.add_argument(
        "--metadata",
        default="data/LJSpeech-1.1/LJSpeech-1.1/metadata.csv",
    )

    args = parser.parse_args()
    metadata_path = Path(args.metadata)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    unsupported_counts = Counter()
    examples = {}

    with metadata_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.reader(file, delimiter="|")

        for row_number, row in enumerate(reader, start=1):
            if len(row) < 3:
                print(
                    f"Skipping malformed row: {row_number} "
                    f"(columns={len(row)}): {row!r}"
                )
                continue

            # Use the normalized transcript, matching the dataset loader.
            text = normalize_text(row[2])

            for character in text:
                if character not in CHAR_TO_ID:
                    unsupported_counts[character] += 1
                    examples.setdefault(character, text)

    print(f"Vocabulary size: {len(CHAR_TO_ID)}")

    if unsupported_counts:
        print("\nUnsupported characters found:")

        for character, count in sorted(
            unsupported_counts.items()
        ):
            print(
                f"{repr(character)} "
                f"(Unicode {ord(character):04X}): "
                f"{count} occurrences"
            )
            print(f"  Example: {examples[character]}")

        raise SystemExit(
            "\nVocabulary check failed. Fix these characters before training."
        )

    print("\nSUCCESS: All dataset transcripts are supported.")
    

if __name__ == "__main__":
    main()
