#!/usr/bin/env python
"""Build the reference statistics behind the stress instability term.

Robust-scales four prosodic features as (value - median) / IQR against a labelled
manifest, and writes ``config/prosody_ref.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from mmsa.audio_io import load_audio  # noqa: E402
from mmsa.errors import MMSAError  # noqa: E402
from mmsa.prosody import FEATURE_NAMES, extract_prosody  # noqa: E402

DEFAULT_OUT = ROOT / "config" / "prosody_ref.json"


def _display_path(path: Path) -> str:
    """Repo-relative path for metadata, tolerant of relative inputs."""
    resolved = path if path.is_absolute() else (Path.cwd() / path)
    try:
        return str(resolved.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "ravdess_manifest.csv")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0, help="cap clips used (0 = all)")
    args = parser.parse_args(argv)

    if not args.manifest.exists():
        print(f"manifest not found: {args.manifest}")
        print("Build one first: python scripts/make_ravdess_manifest.py")
        return 1

    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[: args.limit]

    collected: list[dict[str, float]] = []
    skipped = 0
    started = time.perf_counter()

    for i, row in enumerate(rows, start=1):
        path = Path(row["path"])
        if not path.is_absolute():
            path = ROOT / path
        try:
            audio, info = load_audio(str(path))
        except MMSAError:
            skipped += 1
            continue
        collected.append(extract_prosody(audio, info.sample_rate))
        if i % 10 == 0:
            print(f"  {i}/{len(rows)}")

    if not collected:
        print("no clips could be read; nothing written")
        return 1

    reference: dict[str, dict[str, float]] = {}
    for name in FEATURE_NAMES:
        values = np.asarray([row[name] for row in collected], dtype=np.float64)
        q25, median, q75 = np.percentile(values, [25, 50, 75])
        reference[name] = {"median": round(float(median), 6), "iqr": round(float(q75 - q25), 6)}

    payload = {
        "_meta": {
            "source": _display_path(args.manifest),
            "n_clips": len(collected),
            "n_skipped": skipped,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_s": round(time.perf_counter() - started, 1),
            "method": "median and interquartile range per feature; used by mmsa.stress.robust_z",
            "note": (
                "Reference statistics only. They are not a model and make no claim about "
                "stress; they set the scale of the prosody-instability term."
            ),
        }
    }
    payload.update(reference)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nwrote {args.out} from {len(collected)} clips ({skipped} skipped)")
    for name in ("f0_iqr", "rms_std", "pause_ratio", "zcr_std"):
        stats = reference[name]
        print(f"  {name:<12} median {stats['median']:>10.4f}   iqr {stats['iqr']:>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
