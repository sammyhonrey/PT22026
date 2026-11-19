import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from train import AUG_CACHE, load_data

FEATURE_CACHE = "data/features_cache.joblib"
FIG_DIR = "figures"


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    if os.path.exists(FEATURE_CACHE):
        X, y = joblib.load(FEATURE_CACHE)
    else:
        X, y = load_data()
        joblib.dump((X, y), FEATURE_CACHE)

    _, X_test, _, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train_aug, y_train_aug = joblib.load(AUG_CACHE)

    print("Training a small depth-limited Random Forest for SHAP...", flush=True)
    rf = RandomForestClassifier(
        n_estimators=50, max_depth=10, n_jobs=-1, random_state=42
    )
    rf.fit(X_train_aug, y_train_aug)

    print("Computing SHAP values...", flush=True)
    sample = X_test.sample(300, random_state=42)
    sv = shap.TreeExplainer(rf).shap_values(sample)
    if isinstance(sv, list):
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:
        sv = sv[:, :, 1]

    shap.summary_plot(sv, sample, show=False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Done. Saved figures/shap_summary.png", flush=True)


if __name__ == "__main__":
    main()