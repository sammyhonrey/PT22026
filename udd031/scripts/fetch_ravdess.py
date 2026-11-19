#!/usr/bin/env python
"""Fetch RAVDESS excerpts, or print the links the report cites.

The archive is ~200 MB and CC BY-NC-SA, so it is never committed.

    --print-links        show the download URLs
    --download           fetch and extract
    --from-zip PATH      extract from an archive already on disk
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "demo_samples"
RAW_DIR = ROOT / "data" / "raw"

ZENODO_RECORD = "https://zenodo.org/records/1188976"
DOWNLOAD_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
DOI = "10.5281/zenodo.1188976"
LICENCE = "CC BY-NC-SA 4.0"

EMOTION_CODES: dict[str, str] = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
}

CITATION = (
    "Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of "
    "Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal "
    "expressions in North American English. PLOS ONE, 13(5), e0196391."
)


def _print_links() -> None:
    print("RAVDESS - Ryerson Audio-Visual Database of Emotional Speech and Song")
    print(f"  record    : {ZENODO_RECORD}")
    print(f"  DOI       : {DOI}")
    print(f"  audio zip : {DOWNLOAD_URL}")
    print(f"  licence   : {LICENCE}")
    print()
    print("Cited as:")
    print(f"  {CITATION}")
    print()
    print("Filename grammar:")
    print("  modality-vocal-emotion-intensity-statement-repetition-actor.wav")
    print("  emotions: 01 neutral, 02 calm, 03 happy, 04 sad, 05 angry,")
    print("            06 fearful, 07 disgust, 08 surprised")
    print()
    print("This repository commits no dataset audio. Only the synthetic clips written")
    print("by prepare_demo_samples.py, plus at most six RAVDESS excerpts extracted here,")
    print("appear in demo_samples/.")


def _download(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 1_000_000:
        print(f"  reusing existing archive: {target} ({target.stat().st_size / 1e6:.0f} MB)")
        return target
    print(f"  downloading {DOWNLOAD_URL}")
    print("  (~200 MB; this is the whole speech archive)")

    def _hook(block: int, size: int, total: int) -> None:
        if total > 0:
            done = min(100.0, block * size * 100.0 / total)
            print(f"\r    {done:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(DOWNLOAD_URL, target, _hook)  # noqa: S310 - fixed https URL
    print()
    return target


def _normalise(path: Path, max_seconds: float = 4.0, target_sr: int = 16_000) -> tuple[float, int]:
    """Downmix, resample, trim and cap one excerpt in place.

    Returns ``(seconds, sample_rate)``. On any failure the original file is left
    untouched -- an unusable-but-honest excerpt is better than a deleted one.
    """
    import librosa
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr != target_sr:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
    trimmed, _ = librosa.effects.trim(mono, top_db=35)
    if trimmed.size:
        mono = trimmed
    mono = mono[: int(max_seconds * target_sr)]
    peak = float(np.max(np.abs(mono)))
    if peak > 0.99:
        mono = mono * (0.99 / peak)
    sf.write(str(path), np.asarray(mono, dtype=np.float32), target_sr)
    return round(mono.size / target_sr, 2), target_sr


def _extract(zip_path: Path, count: int = 6) -> int:
    """Pull the first member for each wanted emotion code into ``demo_samples``."""
    if not zip_path.exists():
        print(f"archive not found: {zip_path}")
        print("Run --print-links to see where to obtain it.")
        return 1

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    found: dict[str, str] = {}
    written: list[dict[str, object]] = []

    with zipfile.ZipFile(zip_path) as archive:
        members = sorted(
            name for name in archive.namelist()
            if name.lower().endswith(".wav") and "__MACOSX" not in name
        )
        print(f"  archive contains {len(members)} wav files")

        for name in members:
            stem = Path(name).stem
            parts = stem.split("-")
            if len(parts) != 7:
                continue
            code = parts[2]
            if code not in EMOTION_CODES or code in found:
                continue
            found[code] = name

        for code, emotion in EMOTION_CODES.items():
            if code not in found:
                print(f"  missing emotion code {code} ({emotion}) - archive layout may differ")
                continue
            member = found[code]
            target = DEMO_DIR / f"ravdess_{emotion}_{Path(member).stem}.wav"
            with archive.open(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            try:
                seconds, sample_rate = _normalise(target)
            except Exception as exc:  # noqa: BLE001 - keep the raw excerpt if this fails
                print(f"  warn {target.name}: normalisation skipped ({type(exc).__name__})")
                seconds, sample_rate = 0.0, 0
            size_kb = target.stat().st_size / 1024
            print(f"  wrote {target.name} ({size_kb:.0f} KB, {seconds:.2f}s @ {sample_rate} Hz)")
            written.append(
                {
                    "file": target.name,
                    "kind": "ravdess",
                    "emotion": emotion,
                    "original_member": member,
                    "seconds": seconds,
                    "sample_rate": sample_rate,
                    "licence": LICENCE,
                    "doi": DOI,
                }
            )

    if not written:
        print("no clips extracted.")
        return 1

    manifest = DEMO_DIR / "PROVENANCE_RAVDESS.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "RAVDESS",
                "doi": DOI,
                "licence": LICENCE,
                "citation": CITATION,
                "note": (
                    "Six short excerpts, normalised to mono 16 kHz and capped at 4 s, are "
                    "redistributed for demonstration only, with attribution, under the "
                    "non-commercial share-alike terms of the licence. The archive itself is "
                    "never committed."
                ),
                "clips": written,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {manifest}")
    print(f"{len(written)} excerpt(s) in {DEMO_DIR}")
    print("Remember: the dataset itself must never be committed to the repository.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--print-links", action="store_true", help="print DOI, download URL and citation")
    parser.add_argument("--download", action="store_true", help="download the archive (~200 MB)")
    parser.add_argument("--from-zip", type=Path, help="extract from an archive already on disk")
    args = parser.parse_args(argv)

    if args.print_links:
        _print_links()
        return 0

    zip_path = args.from_zip
    if args.download:
        zip_path = _download(RAW_DIR / "Audio_Speech_Actors_01-24.zip")

    if zip_path is None:
        _print_links()
        print("\nNothing extracted. Pass --download or --from-zip to extract excerpts.")
        return 0

    return _extract(Path(zip_path))


if __name__ == "__main__":
    raise SystemExit(main())
