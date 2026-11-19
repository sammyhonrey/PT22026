#!/usr/bin/env python
"""Measure per-stage latency on this machine.

Orchestration cost is measured with deterministic stand-ins and is repeatable;
model inference is measured only with ``--with-models``, because those timings
belong to the hardware that produced them.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
import wave
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from mmsa.config import load_config  # noqa: E402
from mmsa.pipeline import AnalysisPipeline, safe_analyse  # noqa: E402
from mmsa.schemas import ASRResult, BranchResult, DimensionalResult  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results"


# Deterministic stand-ins so orchestration timing needs no models


class FakeASR:
    backend = "fake"

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRResult:
        return ASRResult(
            text="I am fine, but the deadline has been difficult this week.",
            confidence=0.86,
            n_tokens=11,
            duration_s=float(len(audio) / max(1, sample_rate)),
            backend=self.backend,
        )

    def warmup(self) -> None:
        return None


def _branch(modality: str, labels: tuple[str, ...], probs: tuple[float, ...]) -> BranchResult:
    return BranchResult(
        modality=modality,  # type: ignore[arg-type]
        raw_labels=labels,
        raw_probs=probs,
        mapped_probs={},
        coverage=0.0,
        confidence=max(probs),
        weight=0.5,
        per_class_weight={},
        top_label="neutral",
        latency_ms=0.0,
    )


class FakeAcoustic:
    def predict(self, audio: np.ndarray, sample_rate: int) -> BranchResult:
        from mmsa.ontology import Ontology

        # Build a genuine canonical vector through the real adapter so the timing
        # includes the mapping work the production branch also performs.
        mapped, coverage = _ONTOLOGY.map_probs(
            "acoustic", ("ang", "hap", "neu", "sad"), (0.10, 0.20, 0.55, 0.15)
        )
        return BranchResult(
            modality="acoustic",
            raw_labels=("ang", "hap", "neu", "sad"),
            raw_probs=(0.10, 0.20, 0.55, 0.15),
            mapped_probs=mapped,
            coverage=coverage,
            confidence=0.55,
            weight=0.5,
            per_class_weight={},
            top_label="neutral",
            latency_ms=0.0,
        )

    def warmup(self) -> None:
        return None

    def predict_dimensional(self, audio: np.ndarray, sample_rate: int) -> DimensionalResult:
        return DimensionalResult(arousal=0.55, dominance=0.5, valence=0.45, model_id="fake")


class FakeText:
    def predict(self, text: str) -> BranchResult:
        mapped, coverage = _ONTOLOGY.map_probs(
            "text",
            ("sadness", "joy", "love", "anger", "fear", "surprise"),
            (0.44, 0.08, 0.10, 0.14, 0.16, 0.08),
        )
        return BranchResult(
            modality="text",
            raw_labels=("sadness", "joy", "love", "anger", "fear", "surprise"),
            raw_probs=(0.44, 0.08, 0.10, 0.14, 0.16, 0.08),
            mapped_probs=mapped,
            coverage=coverage,
            confidence=0.44,
            weight=0.5,
            per_class_weight={},
            top_label="sad",
            latency_ms=0.0,
        )

    def warmup(self) -> None:
        return None


_ONTOLOGY = None


def _write_clip(path: Path, seconds: float = 3.0, sample_rate: int = 16000) -> None:
    """A deterministic speech-like test signal: voiced buzz with syllable gating."""
    n = int(seconds * sample_rate)
    t = np.arange(n) / sample_rate
    f0 = 118.0 + 12.0 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    carrier = sum(np.sin(k * phase) / k for k in range(1, 7))
    gate = 0.5 * (1.0 + np.sin(2 * np.pi * 3.1 * t))
    signal = 0.35 * carrier * np.clip(gate, 0.0, 1.0)
    rng = np.random.default_rng(7)
    signal = signal + 0.004 * rng.standard_normal(n)

    # Written with the standard library on purpose: the benchmark must not depend
    # on soundfile/librosa being installed just to lay down 3 seconds of test audio.
    pcm = (np.clip(signal, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q / 100.0 * (len(ordered) - 1)))))
    return ordered[idx]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=3, help="timed repetitions (default: %(default)s)")
    parser.add_argument("--with-models", action="store_true", help="also benchmark real inference")
    parser.add_argument("--out", default=str(OUT_DIR / "latency.json"))
    args = parser.parse_args(argv)

    global _ONTOLOGY
    cfg = load_config()
    from mmsa.ontology import Ontology

    _ONTOLOGY = Ontology.from_yaml(cfg.resolve("config/ontology.yaml"))

    pipeline = AnalysisPipeline(
        cfg, ontology=_ONTOLOGY, asr=FakeASR(), acoustic=FakeAcoustic(), text=FakeText()
    )

    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "bench.wav"
        _write_clip(clip)

        # One untimed pass: warms the decoder, the FFT plans, and the import paths.
        _, err = safe_analyse(pipeline, str(clip))
        if err is not None:
            print(f"Warm-up pass failed: {err.code} {err.user_message}")
            return 1

        stage_runs: dict[str, list[float]] = {}
        totals: list[float] = []
        for _ in range(max(1, args.repeats)):
            started = time.perf_counter()
            result, err = safe_analyse(pipeline, str(clip))
            totals.append((time.perf_counter() - started) * 1000.0)
            if err is not None:
                print(f"Timed pass failed: {err.code}")
                return 1
            for stage, ms in result.timings_ms.items():
                stage_runs.setdefault(stage, []).append(float(ms))

    print("Orchestration latency (fakes; no model inference)")
    print(f"repeats: {args.repeats}")
    print()
    print(f"{'stage':<12}{'mean ms':>10}{'p50 ms':>10}{'max ms':>10}")
    print("-" * 42)
    summary: dict[str, dict[str, float]] = {}
    for stage, values in stage_runs.items():
        row = {
            "mean": statistics.fmean(values),
            "p50": _percentile(values, 50),
            "max": max(values),
        }
        summary[stage] = row
        print(f"{stage:<12}{row['mean']:>10.2f}{row['p50']:>10.2f}{row['max']:>10.2f}")

    print()
    print(f"wall-clock mean: {statistics.fmean(totals):.2f} ms over {len(totals)} runs")

    payload = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "device": cfg.device,
        "repeats": args.repeats,
        "model_inference": "fakes",
        "note": (
            "Orchestration only: audio decode, prosody, fusion, stress, narration. "
            "Transformer forward passes are excluded, so these numbers must never be "
            "presented as end-to-end inference latency."
        ),
        "stages_ms": summary,
        "wall_clock_ms": {"mean": statistics.fmean(totals), "max": max(totals)},
    }

    if args.with_models:
        real = _benchmark_models(cfg, args.repeats)
        if real is None:
            print("\nReal-model benchmark skipped: torch/transformers or the weights are unavailable.")
            payload["model_inference"] = "unavailable"
        else:
            payload["real_models"] = real
            print("\nReal-model latency")
            for stage, row in real["stages_ms"].items():
                print(f"  {stage:<12}{row['mean']:>9.1f} ms")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


def _benchmark_models(cfg, repeats: int) -> dict | None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return None

    from mmsa.pipeline import build_pipeline

    try:
        real = build_pipeline(cfg)
        real.warmup()
    except Exception as exc:  # noqa: BLE001 - offline is an expected outcome
        print(f"  warm-up failed: {type(exc).__name__}: {exc}")
        return None

    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "bench.wav"
        _write_clip(clip)
        runs: dict[str, list[float]] = {}
        for _ in range(max(1, repeats)):
            result, err = safe_analyse(real, str(clip))
            if err is not None:
                print(f"  inference failed: {err.code}")
                return None
            for stage, ms in result.timings_ms.items():
                runs.setdefault(stage, []).append(float(ms))

    return {
        "device": cfg.device,
        "stages_ms": {s: {"mean": statistics.fmean(v), "max": max(v)} for s, v in runs.items()},
    }


if __name__ == "__main__":
    raise SystemExit(main())
