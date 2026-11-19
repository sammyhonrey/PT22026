#!/usr/bin/env python
"""Measure what the two applied fixes are worth on held-out speakers.

Replays 208 clips (actors 21-24) through fusion with and without the lexical
evidence gate and the fitted head, from cached embeddings and the two RAVDESS
carrier sentences. Writes ``results/fusion_ablation.json``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmsa.config import load_config  # noqa: E402
from mmsa.evidence import lexical_evidence  # noqa: E402
from mmsa.fusion import fuse  # noqa: E402
from mmsa.ontology import Ontology  # noqa: E402
from mmsa.schemas import BranchResult  # noqa: E402

CACHE = ROOT / "data" / "embeddings_wav2vec2.npz"
PROBE = ROOT / "models" / "acoustic_probe.npz"
OUT = ROOT / "results" / "fusion_ablation.json"
TEST_ACTORS = ("21", "22", "23", "24")
CARRIER = ("kids are talking by the door", "dogs are sitting by the door")


def _confidence(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    if p.size <= 1:
        return 1.0
    h = float(-(p * np.log(p)).sum())
    return float(max(0.0, 1.0 - h / math.log(probs.size)))


def _acoustic_branch(probs: np.ndarray, labels: tuple[str, ...], canonical: tuple[str, ...], coverage: float):
    mapped = {c: float(probs[labels.index(c)]) for c in canonical}
    return BranchResult(
        modality="acoustic",
        raw_labels=labels,
        raw_probs=tuple(float(p) for p in probs),
        mapped_probs=mapped,
        coverage=coverage,
        confidence=_confidence(probs),
        weight=0.0,
        per_class_weight={},
        top_label=max(mapped.items(), key=lambda kv: kv[1])[0],
        latency_ms=0.0,
    )


def main() -> int:
    if not CACHE.exists() or not PROBE.exists():
        print("need both data/embeddings_wav2vec2.npz and models/acoustic_probe.npz")
        print("  python scripts/train_acoustic_probe.py")
        return 1

    cfg = load_config()
    ontology = Ontology.from_yaml(cfg.resolve("config/ontology.yaml"))
    canonical = tuple(ontology.canonical)

    data = np.load(CACHE, allow_pickle=True)
    x, actors, labels = data["x"], data["actors"], data["labels"]
    probe = np.load(PROBE, allow_pickle=False)
    classes = tuple(str(c) for c in probe["classes"])
    logits = ((x - probe["mean"]) / probe["scale"]) @ probe["W"].T + probe["b"]
    soft = np.exp(logits - logits.max(1, keepdims=True))
    soft /= soft.sum(1, keepdims=True)

    half = x.shape[0] // 2  # the trainer stores natural-gain rows first, then -3 dBFS
    rows = np.flatnonzero(np.isin(actors[:half], TEST_ACTORS))
    gold = labels[:half][rows]
    print(f"held-out clips: {len(rows)}  gold mix: { {g: int((gold == g).sum()) for g in sorted(set(gold))} }")

    def accuracy(gate: float | None, offset: int = 0) -> tuple[float, dict[str, int]]:
        hits = 0
        confusion: dict[str, int] = {}
        for i, row in enumerate(rows):
            text = _text_cache[CARRIER[i % 2]]
            out = fuse(_acoustic_branch(soft[row + offset], classes, canonical, 1.0), text, cfg.fusion,
                       text_evidence=gate)
            hits += int(out.top_label == gold[i])
            if out.top_label != gold[i]:
                key = f"{gold[i]}->{out.top_label}"
                confusion[key] = confusion.get(key, 0) + 1
        return hits / len(rows), dict(sorted(confusion.items(), key=lambda kv: -kv[1]))

    from mmsa.text_branch import TextBranch  # imported late: pulls torch

    text_branch = TextBranch(cfg.models.text_id, device=cfg.device, ontology=ontology)
    _text_cache = {s: text_branch.predict(s) for s in CARRIER}
    evidences = {s: lexical_evidence(s, None, fallback_terms=["stressed", "anxious", "sad"]).score
                 for s in CARRIER}

    acc_nat, conf_nat = accuracy(None, 0)
    acc_pk, conf_pk = accuracy(None, half)
    acc_gate_nat, _ = accuracy(evidences[CARRIER[0]], 0)
    acc_gate_pk, conf_gate = accuracy(evidences[CARRIER[0]], half)
    acoustic_nat = float((np.array(classes)[soft[rows].argmax(1)] == gold).mean())
    acoustic_pk = float((np.array(classes)[soft[rows + half].argmax(1)] == gold).mean())

    print(f"acoustic, natural gain     : {acoustic_nat:.3f}")
    print(f"acoustic, -3 dBFS (app path): {acoustic_pk:.3f}")
    print(f"fusion, no evidence gate   : {acc_nat:.3f} natural / {acc_pk:.3f} app path")
    print(f"fusion, evidence gate      : {acc_gate_nat:.3f} natural / {acc_gate_pk:.3f} app path")

    payload = {
        "corpus": "RAVDESS held-out speakers 21-24",
        "n": int(len(rows)),
        "text_branch_on_carrier_sentences": {
            s: {"top": _text_cache[s].top_label,
                "confidence": round(_text_cache[s].confidence, 3),
                "evidence": evidences[s]}
            for s in CARRIER
        },
        "acoustic": {"natural_gain": round(acoustic_nat, 3), "minus_3dbfs_app_path": round(acoustic_pk, 3)},
        "without_evidence_gate": {"natural_gain": round(acc_nat, 3), "minus_3dbfs_app_path": round(acc_pk, 3)},
        "with_evidence_gate": {
            "natural_gain": round(acc_gate_nat, 3),
            "minus_3dbfs_app_path": round(acc_gate_pk, 3),
            "confusions_app_path": conf_gate,
        },
        "note": (
            "Replayed from the cached embeddings and the two RAVDESS carrier sentences, so "
            "the comparison isolates fusion: the only difference between the two rows is the "
            "text branch's evidence multiplier. Both gain variants are reported because the "
            "application feeds peak-normalised audio; the natural-gain row is the archive as "
            "recorded and is kept as a robustness signal, not as a headline."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
