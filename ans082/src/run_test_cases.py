"""Run the app's prediction code on a folder of test photos and write a results table.

The folder has one sub-folder per kind of test, for example:
    app_tests/normal/            ordinary photos of one item
    app_tests/two_items/         photos with two different items
    app_tests/screenshot/        screenshots
    app_tests/synthetic/         blank, tiny, damaged and other files made by --make-synthetic
Every image is run through exactly the same code as the web app (src/predict.py).

Run from the ans082 folder, for example:
    python src/run_test_cases.py --make-synthetic --sample "F:/.../real_photos/glass/glass_002.jpg"
    python src/run_test_cases.py
Writes results/app_test_cases.csv and results/app_test_cases.md.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def make_synthetic(folder: Path, sample: Path = None) -> list:
    """Write edge-case files into folder/synthetic and return their names."""
    out = Path(folder) / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    Image.new("RGB", (500, 500), "white").save(out / "blank_white.jpg")
    Image.new("RGB", (500, 500), "black").save(out / "blank_black.png")
    Image.fromarray(rng.integers(0, 255, (30, 30, 3), dtype=np.uint8)).save(out / "tiny_30x30.png")
    (out / "damaged_text_named_jpg.jpg").write_text("this is not a picture")
    (out / "empty_file.jpg").write_bytes(b"")
    rgba = rng.integers(0, 255, (300, 300, 4), dtype=np.uint8)
    rgba[..., 3] = 140
    Image.fromarray(rgba, "RGBA").save(out / "transparent_noise.png")
    Image.fromarray(rng.integers(0, 255, (300, 300), dtype=np.uint8), "L").save(out / "grayscale_noise.png")
    y, x = np.mgrid[0:4000, 0:5000]
    gradient = np.stack([x * 255 // 5000, y * 255 // 4000, (x + y) * 255 // 9000], axis=-1).astype(np.uint8)
    Image.fromarray(gradient).save(out / "huge_5000x4000.jpg", quality=70)
    if sample is not None:
        with Image.open(sample) as photo:
            upright = photo.convert("RGB")
        upright.save(out / "reference_upright.jpg")
        exif = Image.Exif()
        exif[274] = 6                                   # "rotate 90 degrees clockwise to show correctly"
        upright.rotate(90, expand=True).save(out / "sideways_exif.jpg", exif=exif)
    return sorted(p.name for p in out.iterdir())


def describe(analysis) -> dict:
    """One short outcome for the table."""
    advice = analysis.advice
    return {
        "outcome": "answer" if advice["confident"] else "not sure",
        "result": f"{analysis.label} ({analysis.confidence:.0%})",
        "bin": advice["bin"] if advice["confident"] else "",
    }


def to_markdown(table: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(table.columns) + " |", "| " + " | ".join("---" for _ in table.columns) + " |"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", type=Path, default=Path("app_tests"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--make-synthetic", action="store_true", help="create the synthetic edge cases first")
    parser.add_argument("--sample", type=Path, help="one upright photo, used to make the sideways test")
    args = parser.parse_args()

    if args.make_synthetic:
        print("Created:", ", ".join(make_synthetic(args.folder, args.sample)))

    from predict import ImageProblem, load_predictor     # loaded here so --make-synthetic needs no model

    files = sorted(p for p in args.folder.rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"No files found in {args.folder}")
    predictor = load_predictor()
    rows = []
    for path in files:
        started = time.perf_counter()
        try:
            row = describe(predictor.analyse(str(path)))
        except ImageProblem as problem:
            row = {"outcome": "rejected", "result": str(problem), "bin": ""}
        except Exception as error:                        # a crash is a failed test, so record it
            row = {"outcome": "CRASH", "result": f"{type(error).__name__}: {error}", "bin": ""}
        row.update(case=path.parent.name, file=path.name,
                   ms=round((time.perf_counter() - started) * 1000))
        rows.append(row)
        print(f"{row['case']:14s}{row['file']:30s}{row['outcome']:10s}{row['result'][:60]}")
    predictor.close()

    table = pd.DataFrame(rows)[["case", "file", "outcome", "result", "bin", "ms"]]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.results_dir / "app_test_cases.csv", index=False)
    (args.results_dir / "app_test_cases.md").write_text(to_markdown(table), encoding="utf-8")
    print(f"\n{len(table)} images tested; crashes: {int((table['outcome'] == 'CRASH').sum())}")


if __name__ == "__main__":
    main()