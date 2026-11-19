"""Train the pose classifier and save it to models/pose_classifier.joblib."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xray import features, trainer, training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", default=training.DEFAULT_ALGORITHM, choices=sorted(training.ALGORITHMS))
    parser.add_argument("--features", default=features.DEFAULT_SET, choices=features.FEATURE_SETS)
    parser.add_argument("--no-user", action="store_true", help="ignore poses recorded in the Teach tab")
    parser.add_argument("--no-validation", action="store_true")
    args = parser.parse_args()

    try:
        model = trainer.train_and_save(args.algorithm, args.features, include_user=not args.no_user,
                                       validate=not args.no_validation)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(str(exc))

    info = model.info
    print(f"{info['algorithm_name']} trained on {info['n_samples']} poses, {info['n_features']} features")
    for label, n in info["per_class"].items():
        print(f"  {label:10s} {n}")
    if "validation" in info:
        v = info["validation"]
        print(f"Validation ({v['method']}, {v['folds']} folds): accuracy {v['accuracy']:.3f}, macro F1 {v['macro_f1']:.3f}")
    print("Saved to models/pose_classifier.joblib")


if __name__ == "__main__":
    main()
