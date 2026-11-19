"""Recreate the selected version 2 sentiment model."""

from pathlib import Path
import argparse
import hashlib
import json
import platform
from importlib.metadata import version

import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent


def load_reviews(data_dir):
    """Reuse the original sample or download it when unavailable."""
    path = data_dir / "amazon_polarity_v2_raw.csv"

    if path.exists():
        print("Using saved Amazon sample.", flush=True)
        return pd.read_csv(path), path

    print("Downloading 20,000 training reviews...", flush=True)

    stream = load_dataset(
        "fancyzhx/amazon_polarity",
        split="train",
        streaming=True
    ).shuffle(seed=42, buffer_size=10000)

    rows = []
    for item in stream.take(20000):
        rows.append({
            "review": item["content"],
            "label": int(item["label"])
        })
        if len(rows) % 2000 == 0:
            print(f"Loaded {len(rows):,} reviews.", flush=True)

    if len(rows) != 20000:
        raise RuntimeError("Dataset download was incomplete.")

    data = pd.DataFrame(rows)
    data.to_csv(path, index=False)
    return data, path


def clean_reviews(data):
    """Remove empty, conflicting, and duplicate review texts."""
    clean = data.dropna(subset=["review", "label"]).copy()

    clean["review"] = (
        clean["review"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    clean = clean[clean["review"] != ""].copy()

    if not clean["label"].isin([0, 1]).all():
        raise ValueError("Unexpected sentiment labels.")

    clean["label"] = clean["label"].astype(int)
    clean["review_key"] = clean["review"].str.casefold()

    counts = clean.groupby("review_key")["label"].nunique()
    conflicts = counts[counts > 1].index

    clean = clean[~clean["review_key"].isin(conflicts)]

    return clean.drop_duplicates("review_key").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="models/sentiment_pipeline_v2_reproduced.joblib",
        help="Output path, relative to the project folder."
    )
    args = parser.parse_args()

    output = BASE_DIR / args.output
    if output.exists():
        raise FileExistsError(
            f"{output} already exists. Choose a different --output filename."
        )

    data_dir = BASE_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    raw, raw_path = load_reviews(data_dir)
    clean = clean_reviews(raw)

    # Recreate the original split and final training-row order.
    train, validation = train_test_split(
        clean,
        test_size=0.20,
        random_state=42,
        stratify=clean["label"]
    )
    development = pd.concat([train, validation], ignore_index=True)

    # Fixed configuration selected in the notebook experiment.
    # No test data is used for training or model selection here.
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words=None,
            min_df=2,
            max_features=100000,
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            C=10.0,
            solver="liblinear",
            max_iter=2000,
            random_state=42
        ))
    ])

    print(f"Training on {len(development):,} reviews...", flush=True)
    model.fit(development["review"], development["label"])
    joblib.dump(model, output)

    metadata = {
        "dataset": "fancyzhx/amazon_polarity",
        "source_split": "train",
        "raw_reviews": len(raw),
        "clean_reviews": len(clean),
        "sampling_seed": 42,
        "shuffle_buffer": 10000,
        "raw_csv_sha256": hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest(),
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ["pandas", "scikit-learn", "joblib", "datasets"]
        }
    }

    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    print("Saved model:", output)
    print("Training completed successfully.")


if __name__ == "__main__":
    main()
