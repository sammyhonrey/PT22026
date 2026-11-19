#!/usr/bin/env python
"""Sweep fusion strategies over measured branch probabilities.

Consumes ``results/branch_probs.json`` and recombines the same measured outputs as
acoustic-only, text-only, uniform, confidence-weighted, the shipped engine and an
oracle. Writes ``results/fusion_sweep.json`` (the report's Table 9).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataclasses import replace  # noqa: E402

from mmsa.config import load_config  # noqa: E402
from mmsa.fusion import fuse  # noqa: E402
from mmsa.schemas import BranchResult  # noqa: E402

RESULTS = ROOT / "results"


def _branch_from(sample: dict, modality: str) -> BranchResult:
    block = sample[modality]
    return BranchResult(
        modality=modality,  # type: ignore[arg-type]
        raw_labels=tuple(block["raw_labels"]),
        raw_probs=tuple(float(p) for p in block["raw_probs"]),
        mapped_probs={k: float(v) for k, v in block["mapped"].items()},
        coverage=float(block["coverage"]),
        confidence=float(block["confidence"]),
        weight=0.5,
        per_class_weight={},
        top_label=str(block["top_label"]),
        latency_ms=0.0,
    )


def _accuracy(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    return sum(1 for gold, pred in pairs if gold == pred) / len(pairs)


def _argmax(mapped: dict[str, float]) -> str:
    return max(mapped.items(), key=lambda kv: kv[1])[0]


def _single_branch(samples: list[dict], which: str) -> dict[str, float]:
    """Unimodal baseline: trust one branch completely."""
    pairs = [(str(s["gold"]), _argmax(s[which]["mapped"])) for s in samples]
    return {"accuracy": _accuracy(pairs), "n": len(pairs)}


def _uniform(samples: list[dict]) -> dict[str, float]:
    """Naive baseline: arithmetic mean of the two mapped vectors."""
    pairs: list[tuple[str, str]] = []
    for s in samples:
        a = {k: float(v) for k, v in s["acoustic"]["mapped"].items()}
        t = {k: float(v) for k, v in s["text"]["mapped"].items()}
        mean = {k: 0.5 * a.get(k, 0.0) + 0.5 * t.get(k, 0.0) for k in a}
        pairs.append((str(s["gold"]), _argmax(mean)))
    return {"accuracy": _accuracy(pairs), "n": len(pairs)}


def _evaluate(samples: list[dict], cfg, mode: str) -> dict[str, float]:
    """Evaluate one of the two pooled modes the engine actually implements."""
    fusion_cfg = replace(cfg.fusion, mode=mode)
    pairs: list[tuple[str, str]] = []
    for sample in samples:
        acoustic = _branch_from(sample, "acoustic")
        text = _branch_from(sample, "text")
        outcome = fuse(acoustic, text, fusion_cfg)
        pairs.append((str(sample["gold"]), outcome.top_label))
    return {"accuracy": _accuracy(pairs), "n": len(pairs)}


def _oracle(samples: list[dict]) -> dict[str, float]:
    """Upper bound: right if either branch alone is right. Not deployable."""
    hits = sum(
        1 for s in samples
        if s["gold"] in {str(s["acoustic"]["top_label"]), str(s["text"]["top_label"])}
    )
    return {"accuracy": hits / len(samples) if samples else 0.0, "n": len(samples)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probs", type=Path, default=RESULTS / "branch_probs.json")
    parser.add_argument("--out", type=Path, default=RESULTS / "fusion_sweep.json")
    args = parser.parse_args(argv)

    if not args.probs.exists():
        print(f"Branch probabilities not found: {args.probs}\n")
        print("Run the measurement step first:")
        print("  python scripts/evaluate_zeroshot.py")
        print("\nThis script only recombines measured outputs; it has nothing to sweep")
        print("without them, and it will not fabricate a sweep table.")
        return 1

    payload = json.loads(args.probs.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    if not samples:
        print("branch_probs.json contains no samples. Nothing to evaluate.")
        return 1

    cfg = load_config()
    print(f"samples: {len(samples)}")

    # -- strategies ------------------------------------------------------- #
    # Unimodal and uniform baselines are computed directly rather than through
    # fuse(): that engine only implements 'credal' and 'weighted', and forcing a
    # third behaviour through it would misrepresent what the shipped code does.
    strategies: dict[str, dict[str, float]] = {
        "acoustic": _single_branch(samples, "acoustic"),
        "text": _single_branch(samples, "text"),
        "uniform": _uniform(samples),
    }
    for mode in ("weighted", "credal"):
        try:
            strategies[mode] = _evaluate(samples, cfg, mode)
        except Exception as exc:  # noqa: BLE001 - a bad mode must not kill the sweep
            print(f"  mode '{mode}' unavailable: {type(exc).__name__}: {exc}")
    strategies["oracle"] = _oracle(samples)

    print()
    print(f"{'strategy':<12}{'accuracy':>10}{'n':>8}")
    print("-" * 30)
    for name, row in strategies.items():
        marker = "  (upper bound)" if name == "oracle" else ""
        print(f"{name:<12}{row['accuracy']:>10.3f}{row['n']:>8}{marker}")

    # -- gamma / delta sweep ---------------------------------------------- #
    grid: list[dict[str, float]] = []
    for gamma, delta in itertools.product((0.25, 0.50, 0.75, 1.00), (0.0, 0.5, 1.0)):
        fusion_cfg = replace(cfg.fusion, mode="credal", gamma=gamma, delta=delta)
        pairs = []
        for sample in samples:
            try:
                outcome = fuse(
                    _branch_from(sample, "acoustic"), _branch_from(sample, "text"), fusion_cfg
                )
            except Exception:  # noqa: BLE001
                continue
            pairs.append((str(sample["gold"]), outcome.top_label))
        grid.append({"gamma": gamma, "delta": delta, "accuracy": _accuracy(pairs), "n": len(pairs)})

    best = max(grid, key=lambda row: row["accuracy"])
    print()
    print(f"best grid point: gamma={best['gamma']} delta={best['delta']} -> {best['accuracy']:.3f}")

    # -- parameter provenance --------------------------------------------- #
    shipped = next(
        (row for row in grid if row["gamma"] == cfg.fusion.gamma and row["delta"] == cfg.fusion.delta),
        None,
    )

    result = {
        "source": str(args.probs),
        "n_samples": len(samples),
        "strategies": strategies,
        "grid": grid,
        "shipped": {
            "gamma": cfg.fusion.gamma,
            "delta": cfg.fusion.delta,
            "accuracy": shipped["accuracy"] if shipped else None,
        },
        "best": best,
        "note": (
            "The shipped gamma and delta are fixed priors chosen before measurement, not tuned "
            "on this evaluation set. Reporting the best grid point as a headline figure would be "
            "selection on the test data; it is recorded here only to show sensitivity."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
