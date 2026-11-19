#!/usr/bin/env python
"""Build a labelled manifest from the RAVDESS speech archive.

Reads ``modality-vocal-emotion-intensity-statement-repetition-actor`` filenames,
keeps audio-only speech, maps eight acted emotions onto the six canonical labels
and writes a CSV with ``path,emotion`` columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = ROOT / "data" / "raw" / "Audio_Speech_Actors_01-24.zip"
EXTRACT_DIR = ROOT / "data" / "ravdess"
DEFAULT_OUT = ROOT / "data" / "ravdess_manifest.csv"

# RAVDESS emotion code -> (source label, canonical label or None to exclude)
EMOTIONS: dict[str, tuple[str, str | None]] = {
    "01": ("neutral", "neutral"),
    "02": ("calm", "neutral"),
    "03": ("happy", "happy"),
    "04": ("sad", "sad"),
    "05": ("angry", "angry"),
    "06": ("fearful", "fear"),
    "07": ("disgust", "disgust"),
    "08": ("surprised", None),
}

LICENCE = "CC BY-NC-SA 4.0"
DOI = "10.5281/zenodo.1188976"


def _fields(member: str) -> list[str] | None:
    stem = Path(member).stem
    parts = stem.split("-")
    return parts if len(parts) == 7 else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP, help="RAVDESS speech archive")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="manifest CSV to write")
    parser.add_argument("--per-emotion", type=int, default=10, help="max clips per canonical class (0 = all)")
    parser.add_argument("--extract", action="store_true", default=True, help="extract selected members (default: on)")
    args = parser.parse_args(argv)

    if not args.zip.exists():
        print(f"archive not found: {args.zip}")
        print("Run: python scripts/fetch_ravdess.py --download")
        return 1

    selected: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    excluded = 0
    scanned = 0

    with zipfile.ZipFile(args.zip) as archive:
        members = sorted(
            n for n in archive.namelist() if n.lower().endswith(".wav") and "__MACOSX" not in n
        )
        print(f"archive members: {len(members)} wav files")

        for member in members:
            parts = _fields(member)
            if parts is None:
                continue
            modality, channel, code = parts[0], parts[1], parts[2]
            if modality != "03" or channel != "01":
                continue  # song (04) and audio-video (01) are out of scope
            if code not in EMOTIONS:
                continue
            scanned += 1
            source_label, canonical = EMOTIONS[code]
            if canonical is None:
                excluded += 1
                continue
            if args.per_emotion and counts[canonical] >= args.per_emotion:
                continue

            target = EXTRACT_DIR / Path(member).name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, target.open("wb") as dst:
                    dst.write(src.read())

            counts[canonical] += 1
            selected.append(
                {
                    "path": str(target.relative_to(ROOT)).replace("\\", "/"),
                    "emotion": canonical,
                    "source_emotion": source_label,
                    "actor": int(parts[6]),
                    "intensity": parts[3],
                    "statement": parts[4],
                    "original_member": member,
                }
            )

    if not selected:
        print("no clips selected -- archive layout may differ from the documented grammar")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "emotion", "source_emotion", "actor", "intensity", "statement"])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in selected)

    meta = {
        "dataset": "RAVDESS (audio-only speech)",
        "doi": DOI,
        "licence": LICENCE,
        "archive": str(args.zip.relative_to(ROOT)).replace("\\", "/"),
        "scanned_speech_clips": scanned,
        "selected": len(selected),
        "per_class": dict(sorted(counts.items())),
        "excluded_surprised": excluded,
        "mapping_note": (
            "calm -> neutral (low-arousal, low-negative, no canonical calm class); "
            "surprised excluded because the canonical ontology has no surprise class "
            "and mapping it would invent a label the annotators did not give."
        ),
        "actors": sorted({int(row["actor"]) for row in selected}),
    }
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"selected {len(selected)} clips -> {args.out}")
    print(f"per class: {dict(sorted(counts.items()))}")
    print(f"excluded {excluded} surprised clips (no canonical class)")
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
