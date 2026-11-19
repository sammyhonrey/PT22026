import re

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES, extract_features, normalize_url

DATA_PATH = "data/urls.csv"
URL_COL = "URL"
LABEL_COL = "Label"

AUG_FRAC = 1.0
AUG_CACHE = "data/train_aug_cache.joblib"


def load_frame():
    df = pd.read_csv(DATA_PATH).dropna(subset=[URL_COL, LABEL_COL])
    before = len(df)
    df = df.drop_duplicates(subset=[URL_COL]).reset_index(drop=True)
    print(f"Rows: {before} -> {len(df)} after removing duplicate URLs")

    labels = df[LABEL_COL].astype(str).str.lower()
    df["y"] = (labels == "bad").astype(int)  
    print("Class counts (0=safe, 1=phishing):")
    print(df["y"].value_counts())
    return df


def featurize(urls):
    return pd.DataFrame([extract_features(u) for u in urls], columns=FEATURE_NAMES)


def load_data():
    """Used by evaluate.py: features and labels of the original URLs."""
    df = load_frame()
    print("Extracting features, this can take a minute...")
    return featurize(df[URL_COL]), df["y"]


def domain_only(url):
    """Remove the scheme and everything after the host (path, query, fragment)."""
    return re.split(r"[/?#]", normalize_url(url), maxsplit=1)[0]


def main():
    df = load_frame()
    print("Extracting features, this can take a few minutes...", flush=True)
    X = featurize(df[URL_COL])
    y = df["y"]

    X_train, X_test, y_train, y_test, u_train, u_test = train_test_split(
        X, y, df[URL_COL], test_size=0.2, stratify=y, random_state=42
    )

    has_path = u_train.map(lambda u: domain_only(u) != normalize_url(u))
    aug_urls = u_train[has_path].sample(frac=AUG_FRAC, random_state=42)
    if len(aug_urls) > 0:
        X_aug = featurize([domain_only(u) for u in aug_urls])
        y_aug = y_train.loc[aug_urls.index]
        X_train_aug = pd.concat([X_train, X_aug], ignore_index=True)
        y_train_aug = pd.concat([y_train, y_aug], ignore_index=True)
    else:
        X_train_aug, y_train_aug = X_train, y_train
    print(f"Training rows: {len(X_train)} original + {len(aug_urls)} domain-only copies")

    X_test_dom = featurize([domain_only(u) for u in u_test])
    joblib.dump((X_train_aug, y_train_aug), AUG_CACHE)

    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "random_forest": RandomForestClassifier(
            n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=42
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...", flush=True)
        model.fit(X_train_aug, y_train_aug)

        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        print(f"=== {name} ===")
        print(classification_report(y_test, pred, target_names=["safe", "phishing"]))
        print("Confusion matrix:\n", confusion_matrix(y_test, pred))
        auc = roc_auc_score(y_test, proba)
        print("ROC-AUC:", round(auc, 4))
        results[name] = auc

        pred_d = model.predict(X_test_dom)
        rep = classification_report(
            y_test, pred_d, output_dict=True, zero_division=0
        )
        auc_d = roc_auc_score(y_test, model.predict_proba(X_test_dom)[:, 1])
        print(
            "Domain-only test (path removed): "
            f"safe recall={rep['0']['recall']:.2f}, "
            f"phishing recall={rep['1']['recall']:.2f}, "
            f"phishing precision={rep['1']['precision']:.2f}, "
            f"phishing F1={rep['1']['f1-score']:.2f}, "
            f"ROC-AUC={auc_d:.4f}"
        )
        joblib.dump(model, f"models/{name}.joblib")

    best = max(results, key=results.get)
    print("\nBest model by ROC-AUC:", best)
    joblib.dump(joblib.load(f"models/{best}.joblib"), "models/best_model.joblib")


if __name__ == "__main__":
    main()