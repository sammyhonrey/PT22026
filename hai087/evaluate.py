"""Evaluate a sentiment model on 500 random IMDB test reviews.

Usage:
    python evaluate.py                              # baseline model
    python evaluate.py models/finetuned-imdb finetuned

Saves accuracy, precision, recall, F1 and the confusion matrix to the
eval_results/ folder (JSON file and PNG image).
"""
import json
import os
import sys

import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import pipeline

MODEL = sys.argv[1] if len(sys.argv) > 1 else "distilbert-base-uncased-finetuned-sst-2-english"
TAG = sys.argv[2] if len(sys.argv) > 2 else "baseline"
N_SAMPLES = 500
SEED = 42
OUT_DIR = "eval_results"

os.makedirs(OUT_DIR, exist_ok=True)

data = load_dataset("stanfordnlp/imdb", split="test").shuffle(seed=SEED).select(range(N_SAMPLES))
texts = [str(t) for t in data["text"]]
y_true = [int(l) for l in data["label"]]

clf = pipeline("sentiment-analysis", model=MODEL, device=-1)
outputs = clf(texts, batch_size=16, truncation=True, max_length=256)
y_pred = [1 if o["label"].upper() in ("POSITIVE", "LABEL_1") else 0 for o in outputs]

acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
cm = confusion_matrix(y_true, y_pred)

print(f"Model: {MODEL}")
print(f"Samples: {N_SAMPLES}")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1: {f1:.4f}")
print(classification_report(y_true, y_pred, target_names=["negative", "positive"]))
print("Confusion matrix:")
print(cm)

with open(os.path.join(OUT_DIR, f"{TAG}_metrics.json"), "w") as f:
    json.dump(
        {
            "model": MODEL,
            "samples": N_SAMPLES,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
        },
        f,
        indent=2,
    )

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["negative", "positive"])
ax.set_yticklabels(["negative", "positive"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion matrix ({TAG})")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"{TAG}_confusion_matrix.png"), dpi=150)