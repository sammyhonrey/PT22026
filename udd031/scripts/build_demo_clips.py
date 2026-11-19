#!/usr/bin/env python
"""Build the six demo excerpts that ship with the repository.

One utterance per emotion, each from a held-out speaker (21-24), whole sentence,
mono 16 kHz, capped at four seconds, no gain rewriting -- the same preprocessing
the trainer uses. Writes ``demo_samples/PROVENANCE_RAVDESS.json``.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16_000
MAX_SECONDS = 4.0
DATASET_DOI = "10.5281/zenodo.1188976"

# emotion -> (RAVDESS emotion code, actor)
SELECTION: dict[str, tuple[str, int]] = {
    "neutral": ("01", 22),
    "happy": ("03", 23),
    "sad": ("04", 24),
    "angry": ("05", 21),
    "fearful": ("06", 22),
    "disgust": ("07", 23),
}

_CITATION = (
    "Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of "
    "Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal "
    "expressions in North American English. PLOS ONE, 13(5), e0196391."
)


def _load(path: Path, actor: int, code: str) -> tuple[str, np.ndarray]:
    """Return ``(member_name, float32 waveform at 16 kHz)`` for one actor/emotion."""
    with zipfile.ZipFile(path) as zf:
        candidates = []
        for name in zf.namelist():
            if not name.lower().endswith(".wav"):
                continue
            fields = Path(name).stem.split("-")
            # RAVDESS: modality-vocal_channel-emotion-intensity-statement-repetition-actor
            if len(fields) != 7:
                continue
            if fields[6] != f"{actor:02d}" or fields[2] != code:
                continue
            if fields[0] != "03" or fields[3:5] != ["01", "01"]:
                continue  # audio-only speech, normal intensity, statement 1
            candidates.append(name)
        if not candidates:
            raise SystemExit(f"no clip for actor {actor:02d} emotion {code} in {path}")
        member = sorted(candidates)[0]
        wav, sr = sf.read(io.BytesIO(zf.read(member)), dtype="float32")

    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != SAMPLE_RATE:
        import librosa

        wav = librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
    return member, np.asarray(wav[: int(SAMPLE_RATE * MAX_SECONDS)], dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", default=str(ROOT / "data" / "raw" / "Audio_Speech_Actors_01-24.zip"))
    parser.add_argument("--out", default=str(ROOT / "demo_samples"))
    parser.add_argument("--force", action="store_true", help="overwrite existing excerpts")
    args = parser.parse_args(argv)

    archive = Path(args.zip)
    if not archive.exists():
        raise SystemExit(f"archive not found: {archive}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("ravdess_*.wav"):
        if args.force:
            stale.unlink()

    clips: list[dict[str, object]] = []
    for emotion, (code, actor) in SELECTION.items():
        member, wav = _load(archive, actor, code)
        target = out_dir / f"ravdess_{emotion}_{Path(member).stem}.wav"
        if target.exists() and not args.force:
            print(f"  skip {target.name} (exists; use --force)")
        else:
            sf.write(target, wav, SAMPLE_RATE, subtype="PCM_16")
            print(f"  wrote {target.name} ({target.stat().st_size / 1024:.0f} KB, {len(wav) / SAMPLE_RATE:.2f} s)")
        clips.append(
            {
                "file": target.name,
                "kind": "ravdess",
                "emotion": emotion,
                "original_member": member,
                "speaker": f"actor {actor:02d} (held-out: never used for training)",
                "seconds": round(len(wav) / SAMPLE_RATE, 2),
                "sample_rate": SAMPLE_RATE,
                "preprocessing": "mono, 16 kHz, whole utterance, capped at 4 s; no silence trimming, no gain change",
                "licence": "CC BY-NC-SA 4.0",
                "doi": DATASET_DOI,
            }
        )

    manifest = out_dir / "PROVENANCE_RAVDESS.json"
    payload = {
        "dataset": "RAVDESS",
        "doi": DATASET_DOI,
        "licence": "CC BY-NC-SA 4.0",
        "citation": _CITATION,
        "note": (
            "Six short excerpts from held-out speakers (actors 21-24), normalised to mono 16 kHz "
            "and capped at 4 s, redistributed for demonstration only, with attribution, under the "
            "non-commercial share-alike terms of the licence. The archive itself is never committed."
        ),
        "built_by": "scripts/build_demo_clips.py",
        "clips": clips,
    }
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {manifest}")
    print(f"{len(clips)} excerpts in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
