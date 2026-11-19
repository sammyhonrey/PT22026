import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from train import URL_COL, domain_only, featurize, load_frame

FEATURE_CACHE = "data/features_cache.joblib"


def main():
    df = load_frame()
    if os.path.exists(FEATURE_CACHE):
        X, y = joblib.load(FEATURE_CACHE)
    else:
        X = featurize(df[URL_COL])
        y = df["y"]

    X_train, X_test, y_train, y_test, u_train, u_test = train_test_split(
        X, y, df[URL_COL], test_size=0.2, stratify=y, random_state=42
    )
    X_test_dom = featurize([domain_only(u) for u in u_test])

    print("Training Random Forest WITHOUT augmentation...", flush=True)
    rf = RandomForestClassifier(
        n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)

    print("=== random_forest, no augmentation ===")
    pred = rf.predict(X_test)
    print(classification_report(y_test, pred, target_names=["safe", "phishing"]))
    print("ROC-AUC:", round(roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]), 4))

    pred_d = rf.predict(X_test_dom)
    rep = classification_report(y_test, pred_d, output_dict=True, zero_division=0)
    auc_d = roc_auc_score(y_test, rf.predict_proba(X_test_dom)[:, 1])
    print(
        "Domain-only test (path removed): "
        f"safe recall={rep['0']['recall']:.2f}, "
        f"phishing recall={rep['1']['recall']:.2f}, "
        f"phishing precision={rep['1']['precision']:.2f}, "
        f"phishing F1={rep['1']['f1-score']:.2f}, "
        f"ROC-AUC={auc_d:.4f}"
    )

    from features import FEATURE_NAMES, extract_features
    for u in ["www.wikipedia.org", "https://github.com", "paypa1-secure-login.xyz"]:
        p = rf.predict_proba(pd.DataFrame([extract_features(u)], columns=FEATURE_NAMES))[0, 1]
        print(f"phishing probability {p:.2f} <- {u}")


if __name__ == "__main__":
    main()