"""Evaluate the pose classifier and write reports/metrics.json and the figures.

Experiments:
  1. algorithm x feature set, validated on subjects left out of training
  2. the chosen model on the held-out test subjects
  3. effect of mirror augmentation
  4. robustness to landmark noise and to the camera viewing angle
  5. learning curve, prediction time and feature importance
  6. size and regularisation of the neural network (--tune)
  7. optional check on videos the model has never seen (--external)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xray import config, dataset, features, training
from xray.classifier import MIN_CONFIDENCE, PoseClassifier

FIGURES = config.REPORT_DIR / "figures"
NOISE_CM = [0, 1, 2, 3, 5, 8]
YAW_DEGREES = [0, 30, 60, 90, 120, 150, 180]
FRACTIONS = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
COLOURS = ["#0f766e", "#f59e0b", "#6366f1", "#ef4444", "#64748b"]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def fit(train, algorithm, feature_set, mirror=True):
    return PoseClassifier(algorithm, feature_set).fit(train.points, train.labels, mirror=mirror)


def predict(model, points):
    return model.pipeline.predict(features.extract(points, model.feature_set))


def rotate_yaw(points, degrees):
    angle = np.radians(degrees)
    c, s = np.cos(angle), np.sin(angle)
    out = points.copy()
    out[..., 0] = c * points[..., 0] + s * points[..., 2]
    out[..., 2] = -s * points[..., 0] + c * points[..., 2]
    return out


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)


def dataset_summary(table):
    summary = {}
    for split in ("train", "test"):
        part = table.select(table.splits == split)
        summary[split] = {
            "poses": int(len(part)),
            "subjects": sorted(set(part.subjects.tolist())),
            "per_class": {c: int((part.labels == c).sum()) for c in table.classes},
        }
    classes = table.classes
    fig, ax = plt.subplots(figsize=(6.4, 3))
    x = np.arange(len(classes))
    for k, split in enumerate(("train", "test")):
        ax.bar(x + (k - 0.5) * 0.38, [summary[split]["per_class"][c] for c in classes], 0.38, label=split, color=COLOURS[k])
    ax.set_xticks(x, classes)
    ax.set_ylabel("poses")
    ax.legend(frameon=False)
    save(fig, "class_distribution.png")
    return summary


def compare(train):
    rows = []
    for feature_set in features.FEATURE_SETS:
        for algorithm in training.ALGORITHMS:
            cv = training.cross_validate(train.points, train.labels, train.subjects, algorithm, feature_set)
            rows.append({"features": feature_set, "algorithm": algorithm, "accuracy": cv["accuracy"],
                         "macro_f1": cv["macro_f1"], "fold_accuracy_std": cv["fold_accuracy_std"]})
            print(f"  {feature_set:11s} {algorithm:7s} accuracy {cv['accuracy']:.3f}  macro F1 {cv['macro_f1']:.3f}")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    names = list(training.ALGORITHMS)
    x = np.arange(len(names))
    width = 0.8 / len(features.FEATURE_SETS)
    for k, feature_set in enumerate(features.FEATURE_SETS):
        scores = [next(r["macro_f1"] for r in rows if r["features"] == feature_set and r["algorithm"] == a) for a in names]
        ax.bar(x + (k - 1.5) * width, scores, width, label=feature_set, color=COLOURS[k])
    ax.set_xticks(x, [training.ALGORITHMS[a][0].split(" (")[0].split(",")[0] for a in names], fontsize=8.5)
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("macro F1, unseen subjects")
    ax.legend(title="feature set", frameon=False, ncols=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "model_comparison.png")
    return rows


def tune_network(train):
    """Validate a few network sizes and weight penalties on unseen subjects."""
    rows = []
    for hidden in ((32,), (64,), (128, 64), (256, 128)):
        for alpha in (1e-4, 1e-3, 1e-2):
            network = MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, max_iter=800, random_state=training.SEED)
            cv = training.cross_validate(train.points, train.labels, train.subjects,
                                         model=make_pipeline(StandardScaler(), network))
            rows.append({"hidden": list(hidden), "alpha": alpha, "accuracy": cv["accuracy"], "macro_f1": cv["macro_f1"]})
            print(f"  hidden {str(hidden):11s} alpha {alpha:<6g} accuracy {cv['accuracy']:.3f}  macro F1 {cv['macro_f1']:.3f}")
    return rows


def test_report(model, test):
    guess = predict(model, test.points)
    classes = model.classes
    matrix = confusion_matrix(test.labels, guess, labels=classes)
    report = classification_report(test.labels, guess, labels=classes, output_dict=True, zero_division=0)
    confidence = model.predict_proba(test.points).max(axis=1)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(matrix, cmap="GnBu")
    ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if matrix[i, j]:
                ax.text(j, i, matrix[i, j], ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black", fontsize=9)
    ax.spines[:].set_visible(False)
    save(fig, "confusion_matrix.png")

    return {
        "n": int(len(test)),
        "accuracy": float(accuracy_score(test.labels, guess)),
        "macro_f1": float(f1_score(test.labels, guess, average="macro")),
        "per_class": {c: {k: round(float(report[c][k]), 4) for k in ("precision", "recall", "f1-score")}
                      | {"support": int(report[c]["support"])} for c in classes},
        "confusion_matrix": matrix.tolist(),
        "accepted_at_threshold": float((confidence >= MIN_CONFIDENCE).mean()),
        "accuracy_when_accepted": float(accuracy_score(
            test.labels[confidence >= MIN_CONFIDENCE], guess[confidence >= MIN_CONFIDENCE])),
    }


def mirror_ablation(train, test, algorithm, feature_set):
    out = {}
    flipped = features.mirror(test.points)
    for mirror in (False, True):
        model = fit(train, algorithm, feature_set, mirror=mirror)
        out["with_mirror" if mirror else "without_mirror"] = {
            "test": float(accuracy_score(test.labels, predict(model, test.points))),
            "mirrored_test": float(accuracy_score(test.labels, predict(model, flipped))),
        }
    return out


def robustness(train, test, algorithm):
    rng = np.random.default_rng(training.SEED)
    models = {fs: fit(train, algorithm, fs) for fs in features.FEATURE_SETS}
    noise = {fs: [] for fs in models}
    view = {fs: [] for fs in models}
    for cm in NOISE_CM:
        noisy = test.points + rng.normal(0.0, cm / 100.0, test.points.shape)
        for fs, model in models.items():
            noise[fs].append(float(accuracy_score(test.labels, predict(model, noisy))))
    for degrees in YAW_DEGREES:
        turned = rotate_yaw(test.points, degrees)
        for fs, model in models.items():
            view[fs].append(float(accuracy_score(test.labels, predict(model, turned))))

    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for k, fs in enumerate(models):
        axes[0].plot(NOISE_CM, noise[fs], marker="o", ms=4, color=COLOURS[k], label=fs)
        axes[1].plot(YAW_DEGREES, view[fs], marker="o", ms=4, color=COLOURS[k], label=fs)
    axes[0].set_xlabel("landmark noise, standard deviation (cm)")
    axes[0].set_ylabel("test accuracy")
    axes[1].set_xlabel("extra rotation of the body about the vertical axis (degrees)")
    axes[1].legend(title="feature set", frameon=False, fontsize=8)
    save(fig, "robustness.png")
    return {"noise_cm": NOISE_CM, "noise_accuracy": noise, "yaw_degrees": YAW_DEGREES, "yaw_accuracy": view}


def learning_curve(train, test, algorithm, feature_set):
    rng = np.random.default_rng(training.SEED)
    sizes, scores = [], []
    for fraction in FRACTIONS:
        runs = []
        for _ in range(5):
            pick = np.concatenate([
                rng.choice(np.flatnonzero(train.labels == c), max(2, int(fraction * (train.labels == c).sum())), replace=False)
                for c in train.classes])
            model = PoseClassifier(algorithm, feature_set).fit(train.points[pick], train.labels[pick])
            runs.append(accuracy_score(test.labels, predict(model, test.points)))
        sizes.append(int(fraction * len(train)))
        scores.append([float(np.mean(runs)), float(np.std(runs))])
    mean, std = np.array(scores).T
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(sizes, mean, marker="o", ms=4, color=COLOURS[0])
    ax.fill_between(sizes, mean - std, mean + std, color=COLOURS[0], alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("training poses")
    ax.set_ylabel("test accuracy")
    save(fig, "learning_curve.png")
    return {"sizes": sizes, "accuracy_mean_std": scores}


def timing(train, test, feature_set):
    out = {}
    one = test.points[:1]
    for algorithm in training.ALGORITHMS:
        start = time.perf_counter()
        model = fit(train, algorithm, feature_set)
        fit_seconds = time.perf_counter() - start
        model.predict_proba(one)
        start = time.perf_counter()
        for _ in range(200):
            model.predict_proba(one)
        out[algorithm] = {"fit_seconds": round(fit_seconds, 3),
                          "predict_ms": round((time.perf_counter() - start) / 200 * 1000, 3)}
    return out


def importance(train, feature_set):
    x_pts, y = training.augment(train.points, train.labels)
    forest = RandomForestClassifier(n_estimators=300, random_state=training.SEED, n_jobs=-1)
    forest.fit(features.extract(x_pts, feature_set), y)
    names = features.feature_names(feature_set)
    order = np.argsort(forest.feature_importances_)[::-1][:15]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.barh([names[i].replace("__", " to ").replace("_", " ") for i in order][::-1],
            forest.feature_importances_[order][::-1], color=COLOURS[0])
    ax.set_xlabel("random forest importance")
    ax.tick_params(axis="y", labelsize=8)
    save(fig, "feature_importance.png")
    kinds = {"coordinates": 0.0, "joint angles": 0.0, "distances": 0.0}
    for name, value in zip(names, forest.feature_importances_):
        key = "joint angles" if name.startswith(("angle_", "trunk")) else "distances" if name.startswith("dist_") else "coordinates"
        kinds[key] += float(value)
    return {"top": [[names[i], round(float(forest.feature_importances_[i]), 4)] for i in order],
            "share_by_kind": {k: round(v, 3) for k, v in kinds.items()}}


def external_check(model, path):
    table = dataset.load(path)
    proba = model.predict_proba(table.points)
    guess = np.array(model.classes)[proba.argmax(axis=1)]
    accepted = proba.max(axis=1) >= MIN_CONFIDENCE
    out = {}
    for source in sorted(set(table.labels.tolist())):
        rows = table.labels == source
        shares = {c: round(float(((guess == c) & accepted & rows).sum() / rows.sum()), 3) for c in model.classes}
        out[source] = {"frames": int(rows.sum()), "unknown": round(float((~accepted & rows).sum() / rows.sum()), 3),
                       **{c: s for c, s in shares.items() if s > 0}}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithm", default=training.DEFAULT_ALGORITHM, choices=sorted(training.ALGORITHMS))
    parser.add_argument("--features", default=features.DEFAULT_SET, choices=features.FEATURE_SETS)
    parser.add_argument("--external", type=Path, help="landmark CSV of videos from outside the dataset")
    parser.add_argument("--quick", action="store_true", help="skip the algorithm x feature set comparison")
    parser.add_argument("--tune", action="store_true", help="also compare neural network sizes")
    args = parser.parse_args()

    try:
        table = dataset.load(config.LANDMARK_CSV)
    except FileNotFoundError:
        sys.exit("No landmark table. Run scripts/download_data.py and scripts/extract_landmarks.py first.")
    train, test = table.select(table.splits == "train"), table.select(table.splits == "test")

    metrics = {"algorithm": args.algorithm, "features": args.features, "dataset": dataset_summary(table)}
    if not args.quick:
        print("Comparing algorithms and feature sets")
        metrics["comparison"] = compare(train)
    if args.tune:
        print("Tuning the neural network")
        metrics["tuning"] = tune_network(train)
    model = fit(train, args.algorithm, args.features)
    metrics["test"] = test_report(model, test)
    print(f"Held-out test: accuracy {metrics['test']['accuracy']:.3f}, macro F1 {metrics['test']['macro_f1']:.3f}")
    metrics["mirror_augmentation"] = mirror_ablation(train, test, args.algorithm, args.features)
    metrics["robustness"] = robustness(train, test, args.algorithm)
    metrics["learning_curve"] = learning_curve(train, test, args.algorithm, args.features)
    metrics["timing"] = timing(train, test, args.features)
    metrics["feature_importance"] = importance(train, args.features)
    if args.external:
        metrics["external_videos"] = external_check(model, args.external)
    metrics["summary"] = {
        "test_accuracy": metrics["test"]["accuracy"],
        "test_macro_f1": metrics["test"]["macro_f1"],
        "test_poses": metrics["test"]["n"],
        "test_subjects": len(metrics["dataset"]["test"]["subjects"]),
    }
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics in {config.METRICS_PATH}, figures in {FIGURES}")


if __name__ == "__main__":
    main()
