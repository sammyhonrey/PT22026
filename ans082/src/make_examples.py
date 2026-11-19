"""Make the small example photos for the app (app/examples/).

Rule, fixed in advance: take the most confident CORRECT real-photo prediction of each class, then keep
the n most confident of those. The photos are shrunk to at most 512 px (JPEG) so that the folder
stays well under 1 MB. The photos are my own phone photos, so they can be committed.

Run from the ans082 folder, for example:
    python src/make_examples.py --real-root "F:/Studies/Projects/SmartSort AI/real_photos"
"""
import argparse
from pathlib import Path, PurePosixPath

import pandas as pd
from PIL import Image, ImageOps

MAX_SIDE = 512
LIMIT_BYTES = 1_000_000


def choose(predictions: pd.DataFrame, n: int) -> pd.DataFrame:
    """The n most confident correct predictions, one per class."""
    correct = predictions[predictions["correct"]].sort_values("confidence", ascending=False)
    return correct.drop_duplicates("true").head(n)


def photo_path(path_text: str, real_root: Path) -> Path:
    """Map a path from the results file (any machine) onto the local real-photo folder."""
    parts = PurePosixPath(str(path_text).replace("\\", "/")).parts
    return real_root / parts[-2] / parts[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="E3_vit_full")
    parser.add_argument("--eval-dir", type=Path, default=Path("results/eval"))
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("app/examples"))
    parser.add_argument("--n", type=int, default=6)
    args = parser.parse_args()

    predictions = pd.read_csv(args.eval_dir / f"{args.experiment}_real_predictions.csv")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for _, row in choose(predictions, args.n).iterrows():
        source = photo_path(row["path"], args.real_root)
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((MAX_SIDE, MAX_SIDE))
            target = args.out_dir / f"{row['true']}_{source.stem.split('_')[-1]}.jpg"
            image.save(target, "JPEG", quality=80)
        size = target.stat().st_size
        total += size
        print(f"{target.name:22s}{size / 1000:7.1f} KB   confidence {row['confidence']:.2f}")
    print(f"Total {total / 1000:.0f} KB")
    if total > LIMIT_BYTES:
        print("warning: the examples are larger than 1 MB; lower MAX_SIDE or use fewer examples")


if __name__ == "__main__":
    main()
