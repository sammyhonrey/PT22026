"""Model candidates, augmentation and subject-wise validation."""

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import features

SEED = 7
DEFAULT_ALGORITHM = "mlp"

ALGORITHMS = {
    "knn": ("k-nearest neighbours (k=5)", lambda: KNeighborsClassifier(n_neighbors=5, weights="distance")),
    "logreg": ("Logistic regression", lambda: LogisticRegression(C=1.0, max_iter=3000)),
    "svm": ("SVM, RBF kernel", lambda: CalibratedClassifierCV(SVC(C=10.0, gamma="scale"), cv=3)),
    "forest": ("Random forest (300 trees)", lambda: RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1)),
    "mlp": ("Neural network (128-64)", lambda: MLPClassifier(
        hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=800, random_state=SEED)),
}


def build(algorithm=DEFAULT_ALGORITHM):
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {algorithm!r}; choose from {sorted(ALGORITHMS)}")
    return make_pipeline(StandardScaler(), ALGORITHMS[algorithm][1]())


def augment(points, labels, mirror=True, jitter=0.0, copies=1, rng=None):
    """Add mirrored poses and, optionally, copies with Gaussian noise in metres."""
    rng = rng or np.random.default_rng(SEED)
    pts, lab = [points], [labels]
    if mirror:
        pts.append(features.mirror(points))
        lab.append(labels)
    if jitter > 0:
        base_pts, base_lab = np.concatenate(pts), np.concatenate(lab)
        for _ in range(copies):
            pts.append(base_pts + rng.normal(0.0, jitter, base_pts.shape))
            lab.append(base_lab)
    return np.concatenate(pts), np.concatenate(lab)


def splitter(groups, labels, max_folds=None):
    """Hold out whole subjects when there are several, otherwise use stratified folds."""
    subjects = len(set(groups.tolist()))
    if subjects >= 3:
        if max_folds and subjects > max_folds:
            return GroupKFold(n_splits=max_folds), f"{max_folds} folds of unseen subjects"
        return LeaveOneGroupOut(), "leave one subject out"
    smallest = min(np.unique(labels, return_counts=True)[1])
    folds = int(max(2, min(5, smallest)))
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED), f"{folds}-fold"


def cross_validate(points, labels, groups, algorithm=DEFAULT_ALGORITHM, feature_set=features.DEFAULT_SET,
                   mirror=True, model=None, max_folds=None):
    """Validate on subjects the model has not seen.

    Accuracy and macro F1 are computed over the pooled out-of-fold predictions;
    the per-fold accuracies show how much the result varies between subjects.
    Pass `model` to validate a pipeline other than the named algorithm.
    """
    model = model if model is not None else build(algorithm)
    cv, method = splitter(groups, labels, max_folds)
    folds, truth, predicted = [], [], []
    for train_idx, test_idx in cv.split(points, labels, groups):
        x_pts, y = augment(points[train_idx], labels[train_idx], mirror=mirror)
        fitted = clone(model).fit(features.extract(x_pts, feature_set), y)
        guess = fitted.predict(features.extract(points[test_idx], feature_set))
        folds.append({
            "held_out": sorted(set(groups[test_idx].tolist())),
            "n": int(len(test_idx)),
            "accuracy": float(accuracy_score(labels[test_idx], guess)),
        })
        truth.append(labels[test_idx])
        predicted.append(guess)
    truth, predicted = np.concatenate(truth), np.concatenate(predicted)
    return {
        "method": method,
        "folds": folds,
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro")),
        "fold_accuracy_std": float(np.std([f["accuracy"] for f in folds])),
        "truth": truth,
        "predicted": predicted,
    }
