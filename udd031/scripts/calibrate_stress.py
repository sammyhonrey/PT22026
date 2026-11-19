#!/usr/bin/env python
"""Calibrate the stress band thresholds against measured scores.

No corpus labels stress levels, so the boundaries are set from where the composite
actually falls on held-out speakers: the 33rd and 67th percentiles. Hand-set
thresholds put every RAVDESS utterance in High, which is why this exists.

    python scripts/calibrate_stress.py [--write]

Writes ``results/stress_calibration.json``; ``--write`` updates config/stress.yaml.
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmsa.config import load_config  # noqa: E402
from mmsa.pipeline import build_pipeline  # noqa: E402

SAMPLE_RATE = 16_000
EMOTIONS = {
    "01": "neutral",
    "02": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
}
SPEAKERS = ("21", "22", "23", "24")  # held out from the acoustic head's fit
DEFAULT_ZIP = ROOT / "data" / "raw" / "Audio_Speech_Actors_01-24.zip"
OUT_JSON = ROOT / "results" / "stress_calibration.json"
STRESS_YAML = ROOT / "config" / "stress.yaml"


def sample_clips(zip_path: Path, per_emotion: int) -> list[tuple[str, str, bytes]]:
    """Deterministic sample: ``per_emotion`` clips of each emotion, held-out speakers."""
    chosen: dict[str, list[tuple[str, str, bytes]]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith(".wav"))
        for name in names:
            parts = Path(name).stem.split("-")
            if len(parts) != 7 or parts[0] != "03" or parts[6] not in SPEAKERS:
                continue
            emotion = EMOTIONS.get(parts[2])
            if emotion is None:
                continue
            bucket = chosen.setdefault(emotion, [])
            if len(bucket) < per_emotion:
                bucket.append((emotion, Path(name).stem, zf.read(name)))
    out: list[tuple[str, str, bytes]] = []
    for emotion in ("neutral", "happy", "sad", "angry", "fear", "disgust"):
        out.extend(chosen.get(emotion, []))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--per-emotion", type=int, default=4)
    parser.add_argument("--write", action="store_true", help="rewrite the thresholds in config/stress.yaml")
    parser.add_argument("--json", type=Path, default=OUT_JSON)
    args = parser.parse_args(argv)

    if not args.zip.exists():
        print(f"RAVDESS archive not found: {args.zip}")
        return 1

    clips = sample_clips(args.zip, args.per_emotion)
    print(f"sampling {len(clips)} held-out clips ({args.per_emotion} per emotion, speakers {','.join(SPEAKERS)})")

    cfg = load_config()
    pipeline = build_pipeline(cfg)

    records: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for emotion, stem, blob in clips:
            path = Path(tmp) / f"{stem}.wav"
            data, sr = sf.read(io.BytesIO(blob), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            sf.write(path, np.asarray(data), sr, subtype="PCM_16")
            try:
                result = pipeline.analyse(str(path))
            except Exception as exc:  # noqa: BLE001 - a failed clip must not stop the run
                print(f"  {emotion:<8} {stem}: failed ({type(exc).__name__}: {exc})")
                continue
            records.append(
                {
                    "clip": stem,
                    "gold": emotion,
                    "score": result.stress.score,
                    "band": result.stress.band,
                    "logit": result.stress.features.get("logit"),
                    "fused": result.fused_emotion,
                    "text_evidence": round(result.text_evidence, 3),
                }
            )
            print(
                f"  {emotion:<8} fused {result.fused_emotion:<8} score {result.stress.score:.3f} "
                f"band {result.stress.band:<6} evidence {result.text_evidence:.2f}"
            )

    if not records:
        print("no clips analysed")
        return 1

    scores = [float(r["score"]) for r in records]
    q33, q67 = (float(v) for v in np.quantile(scores, [1.0 / 3.0, 2.0 / 3.0]))
    per_emotion: dict[str, dict[str, float]] = {}
    for emotion in ("neutral", "happy", "sad", "angry", "fear", "disgust"):
        subset = [float(r["score"]) for r in records if r["gold"] == emotion]
        if subset:
            per_emotion[emotion] = {
                "n": len(subset),
                "mean": round(statistics.fmean(subset), 4),
                "min": round(min(subset), 4),
                "max": round(max(subset), 4),
            }

    bands_before = {b: sum(1 for r in records if r["band"] == b) for b in ("Low", "Medium", "High")}
    print(f"\nmeasured score distribution (n={len(records)}): min {min(scores):.3f} "
          f"median {statistics.median(scores):.3f} max {max(scores):.3f}")
    print(f"previous thresholds 0.35 / 0.65 -> bands {bands_before}")
    print(f"quantile thresholds (33% / 67%) -> {q33:.3f} / {q67:.3f}")

    payload = {
        "corpus": "RAVDESS held-out speakers 21-24",
        "n": len(records),
        "per_emotion": per_emotion,
        "score": {"min": min(scores), "median": statistics.median(scores), "max": max(scores)},
        "thresholds_before": {"low": cfg.stress.threshold_low, "high": cfg.stress.threshold_high},
        "thresholds_measured": {"low": round(q33, 3), "high": round(q67, 3)},
        "bands_before": bands_before,
        "note": (
            "Thresholds are the 33rd and 67th percentile of this measured distribution, so "
            "each band holds roughly a third of the reference corpus. The index itself is "
            "unchanged: only the boundaries were moved to where the score actually falls."
        ),
        "records": records,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.json}")

    if args.write:
        text = STRESS_YAML.read_text(encoding="utf-8")
        new_low, new_high = f"{q33:.3f}", f"{q67:.3f}"
        import re

        text = re.sub(r"^  low: .*$", f"  low: {new_low}", text, count=1, flags=re.MULTILINE)
        text = re.sub(r"^  high: .*$", f"  high: {new_high}", text, count=1, flags=re.MULTILINE)
        STRESS_YAML.write_text(text, encoding="utf-8")
        print(f"rewrote thresholds in {STRESS_YAML}: low {new_low} / high {new_high}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
