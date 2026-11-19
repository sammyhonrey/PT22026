#!/usr/bin/env python
"""Zero-shot evaluation of each branch on labelled RAVDESS audio.

Neither checkpoint was trained on RAVDESS, so these numbers measure transfer, not
fitted performance, and they are the baseline fusion is compared against.

    python scripts/evaluate_zeroshot.py --manifest data/ravdess_manifest.csv

Writes ``results/zeroshot.json`` and ``results/branch_probs.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from mmsa.config import load_config  # noqa: E402
from mmsa.errors import MMSAError  # noqa: E402
from mmsa.audio_io import load_audio  # noqa: E402
from mmsa.ontology import Ontology  # noqa: E402
from mmsa.schemas import CANONICAL_LABELS  # noqa: E402

RESULTS = ROOT / "results"


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    needed = {"path", "emotion"}
    if not rows:
        raise SystemExit(f"manifest {path} is empty")
    missing = needed - set(rows[0].keys())
    if missing:
        raise SystemExit(f"manifest {path} is missing column(s): {sorted(missing)}")
    return rows


def _resolve(manifest_path: Path, row: dict[str, str]) -> Path:
    raw = Path(row["path"])
    if raw.is_absolute():
        return raw
    for base in (manifest_path.parent, ROOT, ROOT / "data"):
        candidate = base / raw
        if candidate.exists():
            return candidate
    return raw


def _metrics(y_true: list[str], y_pred: list[str]) -> dict[str, object]:
    """Accuracy, macro-F1 and the confusion matrix, in fixed label order."""
    labels = list(CANONICAL_LABELS)
    index = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    confusion = np.zeros((n, n), dtype=int)
    for truth, pred in zip(y_true, y_pred):
        confusion[index[truth], index[pred]] += 1

    correct = int(np.trace(confusion))
    total = int(confusion.sum())
    accuracy = correct / total if total else 0.0

    per_class: dict[str, dict[str, float]] = {}
    f1_scores: list[float] = []
    for i, label in enumerate(labels):
        tp = float(confusion[i, i])
        fp = float(confusion[:, i].sum() - tp)
        fn = float(confusion[i, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": int(confusion[i, :].sum())}
        if confusion[i, :].sum() > 0:
            f1_scores.append(f1)

    return {
        "n": total,
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "per_class": per_class,
        "confusion": confusion.tolist(),
        "labels": labels,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "ravdess_manifest.csv")
    parser.add_argument("--limit", type=int, default=0, help="cap the number of clips (0 = all)")
    parser.add_argument("--out", type=Path, default=RESULTS / "zeroshot.json")
    args = parser.parse_args(argv)

    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}\n")
        print("This script measures transfer performance on labelled audio, so it needs labels.")
        print("Build a manifest with columns 'path,emotion' (emotion in", list(CANONICAL_LABELS), ")")
        print("pointing at RAVDESS wav files, then re-run. Nothing was measured and nothing")
        print("was written -- no placeholder metrics are produced on purpose.")
        return 1

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        print("torch/transformers are not installed. Run: pip install -r requirements.txt")
        return 1

    cfg = load_config()
    ontology = Ontology.from_yaml(cfg.resolve("config/ontology.yaml"))

    from mmsa.acoustic_branch import build_acoustic_branch
    from mmsa.asr import build_asr
    from mmsa.text_branch import build_text_branch

    print("loading models ...")
    acoustic = build_acoustic_branch(cfg, ontology)
    text = build_text_branch(cfg, ontology)
    asr = build_asr(cfg)
    for component in (acoustic, text, asr):
        hook = getattr(component, "warmup", None)
        if callable(hook):
            hook()

    rows = _read_manifest(args.manifest)
    invalid = [r for r in rows if r["emotion"].strip().lower() not in CANONICAL_LABELS]
    if invalid:
        print(f"{len(invalid)} row(s) have an emotion outside the canonical set; dropping them.")
        rows = [r for r in rows if r["emotion"].strip().lower() in CANONICAL_LABELS]
    if args.limit:
        rows = rows[: args.limit]

    print(f"evaluating {len(rows)} clips ...")
    gold: list[str] = []
    pred_ac: list[str] = []
    pred_tx: list[str] = []
    evidence: list[dict[str, object]] = []
    skipped = 0
    started = time.perf_counter()

    for i, row in enumerate(rows, start=1):
        path = _resolve(args.manifest, row)
        if not path.exists():
            skipped += 1
            continue
        try:
            audio, info = load_audio(path, min_duration_s=cfg.min_duration_s,
                                      max_duration_s=cfg.max_duration_s)
        except MMSAError as exc:
            skipped += 1
            print(f"  [{i}] skip {path.name}: {exc.code}")
            continue

        acoustic_result = acoustic.predict(audio, info.sample_rate)
        try:
            transcript = asr.transcribe(audio, info.sample_rate)
        except MMSAError:
            skipped += 1
            continue
        text_result = text.predict(transcript.text)

        label = row["emotion"].strip().lower()
        gold.append(label)
        pred_ac.append(acoustic_result.top_label)
        pred_tx.append(text_result.top_label)
        evidence.append(
            {
                "path": str(path),
                "gold": label,
                "asr_text": transcript.text,
                "asr_confidence": transcript.confidence,
                "acoustic": {
                    "raw_labels": list(acoustic_result.raw_labels),
                    "raw_probs": list(acoustic_result.raw_probs),
                    "mapped": dict(acoustic_result.mapped_probs),
                    "coverage": acoustic_result.coverage,
                    "confidence": acoustic_result.confidence,
                    "top_label": acoustic_result.top_label,
                },
                "text": {
                    "raw_labels": list(text_result.raw_labels),
                    "raw_probs": list(text_result.raw_probs),
                    "mapped": dict(text_result.mapped_probs),
                    "coverage": text_result.coverage,
                    "confidence": text_result.confidence,
                    "top_label": text_result.top_label,
                },
            }
        )
        if i % 10 == 0:
            print(f"  {i}/{len(rows)}")

    elapsed = time.perf_counter() - started
    if not gold:
        print("No clips could be processed. Nothing written.")
        return 1

    payload = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": cfg.device,
        "manifest": str(args.manifest),
        "n_requested": len(rows),
        "n_scored": len(gold),
        "n_skipped": skipped,
        "elapsed_s": round(elapsed, 2),
        "gold_distribution": dict(Counter(gold)),
        "models": {
            "acoustic": cfg.models.acoustic_id,
            "text": cfg.models.text_id,
            "asr": cfg.models.asr_id,
        },
        "note": (
            "Zero-shot transfer: the acoustic head was trained on IEMOCAP and the text head on "
            "the 'emotion' dataset, so these figures are a transfer baseline, not fitted "
            "performance on RAVDESS."
        ),
        "acoustic": _metrics(gold, pred_ac),
        "text": _metrics(gold, pred_tx),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    probs_path = RESULTS / "branch_probs.json"
    probs_path.write_text(
        json.dumps({"measured_at": payload["measured_at"], "samples": evidence}, indent=2),
        encoding="utf-8",
    )

    print()
    for branch in ("acoustic", "text"):
        block = payload[branch]
        print(f"{branch:<9} accuracy {block['accuracy']:.3f}   macro-F1 {block['macro_f1']:.3f}   n={block['n']}")
    print(f"\nwrote {args.out}")
    print(f"wrote {probs_path}")
    print("Next: python scripts/evaluate_fusion.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
