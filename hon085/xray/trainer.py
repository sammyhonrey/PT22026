"""Assemble the training table, validate, fit and save the classifier."""

import numpy as np

from . import config, dataset, features, training
from .classifier import PoseClassifier

MIN_SAMPLES_PER_POSE = 20


def training_table(include_user=True):
    """Public training split plus everything the user recorded."""
    tables = []
    if config.LANDMARK_CSV.exists():
        base = dataset.load(config.LANDMARK_CSV)
        tables.append(base.select(base.splits == "train"))
    if include_user and config.USER_CSV.exists():
        tables.append(dataset.load(config.USER_CSV))
    table = dataset.concat(tables)
    if not len(table):
        raise FileNotFoundError(
            "no training data; run scripts/download_data.py and scripts/extract_landmarks.py, "
            "or record poses in the Teach tab")
    return table


def check(table):
    sizes = dict(zip(*np.unique(table.labels, return_counts=True)))
    if len(sizes) < 2:
        raise ValueError("at least two different poses are needed for training")
    small = [f"{label} ({n})" for label, n in sizes.items() if n < MIN_SAMPLES_PER_POSE]
    if small:
        raise ValueError(f"too few samples, need {MIN_SAMPLES_PER_POSE} per pose: " + ", ".join(small))


def train_and_save(algorithm=training.DEFAULT_ALGORITHM, feature_set=features.DEFAULT_SET, include_user=True,
                   validate=True, max_folds=None, path=None):
    table = training_table(include_user)
    check(table)
    model = PoseClassifier(algorithm, feature_set).fit(table.points, table.labels)
    if validate:
        cv = training.cross_validate(table.points, table.labels, table.subjects, algorithm, feature_set,
                                      max_folds=max_folds)
        model.info["validation"] = {
            "method": cv["method"],
            "folds": len(cv["folds"]),
            "accuracy": round(cv["accuracy"], 4),
            "macro_f1": round(cv["macro_f1"], 4),
        }
    model.info["user_samples"] = int((table.splits == "user").sum())
    model.save(path)
    return model
