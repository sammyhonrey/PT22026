"""Check that the app's model gives the same answers as the evaluation did (E3 on the real photos).

The evaluation ran from the training checkpoint on a GPU. The app runs the exported model on the CPU.
If both give (almost) the same answers, the exported model really is the model that was evaluated.

Run from the ans082 folder (about 4 minutes for all 214 photos):
    python src/check_app_matches_eval.py --real-root "F:/Studies/Projects/SmartSort AI/real_photos"
"""
import argparse
from pathlib import Path, PurePosixPath

import pandas as pd

from predict import load_predictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-dir", type=Path, default=Path("results/eval"))
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--experiment", default="E3_vit_full")
    args = parser.parse_args()

    saved = pd.read_csv(args.eval_dir / f"{args.experiment}_real_predictions.csv")
    predictor = load_predictor()
    same, differences, different_photos = 0, [], []
    for _, row in saved.iterrows():
        parts = PurePosixPath(str(row["path"]).replace("\\", "/")).parts
        photo = args.real_root / parts[-2] / parts[-1]
        result = predictor.analyse(str(photo))
        if result.label == row["pred"]:
            same += 1
        else:
            different_photos.append(f"{photo.name}: evaluation said {row['pred']}, app says {result.label}")
        differences.append(abs(result.confidence - row["confidence"]))
    predictor.close()

    total = len(saved)
    print(f"{same}/{total} predictions are identical ({same / total:.1%})")
    print(f"confidence difference: mean {sum(differences) / total:.4f}, largest {max(differences):.4f}")
    for line in different_photos:
        print("different:", line)


if __name__ == "__main__":
    main()