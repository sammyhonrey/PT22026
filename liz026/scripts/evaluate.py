"""
Evaluate the pretrained model on a labeled test set.

Expected folder layout (as in the Kaggle "Chest X-Ray Images (Pneumonia)"
dataset, https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia):

    chest_xray/test/NORMAL/*.jpeg
    chest_xray/test/PNEUMONIA/*.jpeg

Usage:
    python scripts/evaluate.py --data path/to/chest_xray/test

Outputs (in results/):
    metrics.json            accuracy, precision, recall, specificity, F1, ROC-AUC
    threshold_sweep.csv     the same metrics for thresholds 0.05 ... 0.95, 0.98 ... 0.999
    predictions.csv         per-image probability
    confusion_matrix.png    } only if matplotlib is installed
    roc_curve.png           }
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402  (sets the Keras backend)
from src.model_loader import get_model  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png"}


def load_dataset(root: Path):
    paths, labels = [], []
    for label, folder in enumerate(["NORMAL", "PNEUMONIA"]):
        d = root / folder
        if not d.is_dir():
            sys.exit(f"Missing folder: {d}")
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
        paths += files
        labels += [label] * len(files)
    return paths, np.array(labels)


def predict_all(model, paths, batch_size=32):
    probs = []
    for i in range(0, len(paths), batch_size):
        batch = np.concatenate([preprocess(Image.open(p)) for p in paths[i:i + batch_size]])
        out = model.predict(batch, verbose=0)
        probs.append(np.asarray(out).reshape(-1))
        print(f"\r{min(i + batch_size, len(paths))}/{len(paths)} images", end="", flush=True)
    print()
    return np.concatenate(probs)


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"threshold": round(thr, 3), "accuracy": (tp + tn) / len(y), "precision": prec,
            "recall_sensitivity": rec, "specificity": spec, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def roc_auc(y, p):
    """ROC curve + AUC with numpy only (rank-based)."""
    order = np.argsort(-p)
    y_sorted = y[order]
    tpr = np.concatenate([[0], np.cumsum(y_sorted) / max(y.sum(), 1)])
    fpr = np.concatenate([[0], np.cumsum(1 - y_sorted) / max((1 - y).sum(), 1)])
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # renamed in NumPy 2.0
    return fpr, tpr, float(trapezoid(tpr, fpr))


def save_plots(out_dir, m, fpr, tpr, auc):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plots (pip install matplotlib)")
        return
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1], config.CLASS_NAMES); ax.set_yticks([0, 1], config.CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (threshold {m['threshold']})")
    fig.tight_layout(); fig.savefig(out_dir / "confusion_matrix.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}"); ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out_dir / "roc_curve.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="folder containing NORMAL/ and PNEUMONIA/")
    ap.add_argument("--threshold", type=float, default=config.DEFAULT_THRESHOLD)
    ap.add_argument("--out", type=Path, default=config.PROJECT_ROOT / "results")
    args = ap.parse_args()

    paths, y = load_dataset(args.data)
    print(f"{len(paths)} images: {int((y == 0).sum())} normal, {int((y == 1).sum())} pneumonia")
    p = predict_all(get_model(), paths)

    args.out.mkdir(parents=True, exist_ok=True)
    m = metrics_at(y, p, args.threshold)
    fpr, tpr, auc = roc_auc(y, p)
    m["roc_auc"] = auc
    (args.out / "metrics.json").write_text(json.dumps(m, indent=2))

    with open(args.out / "threshold_sweep.csv", "w", newline="") as f:
        thresholds = list(np.round(np.arange(0.05, 0.96, 0.05), 2)) + [0.98, 0.99, 0.995, 0.999]
        rows = [metrics_at(y, p, float(t)) for t in thresholds]
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with open(args.out / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "true_label", "p_pneumonia"])
        for path, label, prob in zip(paths, y, p):
            w.writerow([path.name, config.CLASS_NAMES[label], f"{prob:.6f}"])
    save_plots(args.out, m, fpr, tpr, auc)

    print(json.dumps(m, indent=2))
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
