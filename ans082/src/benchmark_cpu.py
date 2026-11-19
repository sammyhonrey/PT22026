"""Measure model size and single-image CPU speed for trained SmartSort checkpoints.

For every checkpoint it reports: file size, parameter count, and the time for one image on the CPU,
split into preprocessing (open the photo, resize, normalise) and the model forward pass.
Run it on the same laptop that will run the demo.

Run from the ans082 folder, for example:
    python src/benchmark_cpu.py --models E1_resnet50=models/E1_resnet50/best.pt ^
        E3_vit_full=models/E3_vit_full/best.pt --image "F:/path/to/one_photo.jpg"
"""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from data import build_transforms, load_rgb
from evaluate import load_checkpoint


def time_calls(fn, repeats: int, warmup: int) -> list:
    """Call fn() warmup times (not timed), then repeats times; return the times in ms."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        times.append((time.perf_counter() - started) * 1000.0)
    return times


def summarise(times: list) -> dict:
    """Mean, median and 95th percentile of a list of timings in ms."""
    ordered = sorted(times)
    return {
        "mean_ms": round(statistics.fmean(times), 1),
        "median_ms": round(statistics.median(times), 1),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 1),
    }


def benchmark_one(name: str, ckpt_path: Path, image_path: Path, repeats: int, warmup: int) -> dict:
    """Benchmark one checkpoint on the CPU with batch size 1."""
    device = torch.device("cpu")
    model, ckpt = load_checkpoint(ckpt_path, device)
    _, eval_tf = build_transforms(ckpt["mean"], ckpt["std"], 224)
    total_params = sum(p.numel() for p in model.parameters())

    prep = time_calls(lambda: eval_tf(load_rgb(image_path)), repeats, warmup)
    tensor = eval_tf(load_rgb(image_path)).unsqueeze(0)

    def forward():
        with torch.inference_mode():
            model(tensor)

    fwd = time_calls(forward, repeats, warmup)
    result = {
        "experiment": name,
        "checkpoint_mb": round(ckpt_path.stat().st_size / 1e6, 1),
        "parameters_m": round(total_params / 1e6, 2),
        "preprocess": summarise(prep),
        "forward": summarise(fwd),
        "total_median_ms": round(statistics.median(prep) + statistics.median(fwd), 1),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True, metavar="NAME=PATH",
                        help="one or more experiment names with their best.pt path")
    parser.add_argument("--image", required=True, type=Path, help="one sample photo")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("results/benchmark_cpu.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")
    results = []
    for item in args.models:
        name, _, path = item.partition("=")
        if not path or not Path(path).exists():
            raise SystemExit(f"Bad --models entry or missing file: {item}")
        print(f"Benchmarking {name} ...")
        results.append(benchmark_one(name, Path(path), args.image, args.repeats, args.warmup))
    report = {
        "machine": {"processor": platform.processor(), "system": platform.platform(),
                    "torch": torch.__version__, "threads": torch.get_num_threads(),
                    "python": platform.python_version()},
        "settings": {"batch_size": 1, "repeats": args.repeats, "warmup": args.warmup,
                     "image": args.image.name},
        "models": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{'experiment':26s}{'size MB':>9s}{'params M':>10s}{'prep ms':>9s}"
          f"{'model ms':>10s}{'total ms':>10s}")
    for r in results:
        print(f"{r['experiment']:26s}{r['checkpoint_mb']:9.1f}{r['parameters_m']:10.2f}"
              f"{r['preprocess']['median_ms']:9.1f}{r['forward']['median_ms']:10.1f}"
              f"{r['total_median_ms']:10.1f}")
    print(f"\nSaved {args.out}  (threads: {torch.get_num_threads()})")


if __name__ == "__main__":
    main()
