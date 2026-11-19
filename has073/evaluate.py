import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from train import load_data

FEATURE_CACHE = "data/features_cache.joblib"
FIG_DIR = "figures"
MODEL_NAMES = ["logreg", "random_forest", "gradient_boosting"]
LABELS = {
    "logreg": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}


def get_data():
    if os.path.exists(FEATURE_CACHE):
        return joblib.load(FEATURE_CACHE)
    X, y = load_data()
    joblib.dump((X, y), FEATURE_CACHE)
    return X, y


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    X, y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {n: joblib.load(f"models/{n}.joblib") for n in MODEL_NAMES}

    rows = []
    for n, m in models.items():
        pred = m.predict(X_test)
        proba = m.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": LABELS[n],
            "Phishing recall": recall_score(y_test, pred),
            "Phishing F1": f1_score(y_test, pred),
            "ROC-AUC": roc_auc_score(y_test, proba),
        })
    df = pd.DataFrame(rows).set_index("Model")
    print(df.round(4))
    df.to_csv(f"{FIG_DIR}/model_comparison.csv")

    ax = df.plot(kind="bar", figsize=(9, 5), rot=0)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison on the test set")
    ax.set_ylabel("Score")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_comparison.png", dpi=200)
    plt.close()

    best = models["random_forest"]
    ConfusionMatrixDisplay.from_estimator(
        best, X_test, y_test, display_labels=["Safe", "Phishing"], cmap="Blues"
    )
    plt.title("Confusion matrix: Random Forest")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/confusion_matrix_rf.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 6))
    for n, m in models.items():
        proba = m.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        plt.plot(fpr, tpr, label=f"{LABELS[n]} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/roc_curves.png", dpi=200)
    plt.close()

    imp = pd.Series(best.feature_importances_, index=X.columns).sort_values()
    imp.plot(kind="barh", figsize=(8, 6))
    plt.title("Random Forest feature importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/rf_feature_importance.png", dpi=200)
    plt.close()

    sample = X_test.sample(500, random_state=42)
    explainer = shap.TreeExplainer(best)
    sv = explainer.shap_values(sample)
    if isinstance(sv, list):
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:
        sv = sv[:, :, 1]
    shap.summary_plot(sv, sample, show=False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Done. Charts saved in the figures/ folder.")


if __name__ == "__main__":
    main()