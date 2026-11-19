"""Train and compare spam detection models on the SMS Spam Collection."""

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")  # lets us save pictures without opening a window
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DATA_PATH = "data/SMSSpamCollection"
MODEL_PATH = "model.joblib"


def load_data(path):
    """Read the file. Each line is: label, TAB, message."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="latin-1",
        quoting=3,
    )
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def build_models():
    """Three different robots. Each one has the same TF-IDF 'reading glasses'."""
    def tfidf():
        return TfidfVectorizer(lowercase=True, stop_words="english")

    return {
        "Naive Bayes": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB())]),
        "Logistic Regression": Pipeline([
            ("tfidf", tfidf()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", tfidf()),
            ("clf", LinearSVC(class_weight="balanced")),
        ]),
    }


def main():
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} messages")
    print(df["label"].value_counts().rename({0: "ham", 1: "spam"}), "\n")

    # Hide 20% of the messages as the secret test paper
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2,
        random_state=42, stratify=df["label"],
    )

    results = []
    trained = {}
    for name, model in build_models().items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        trained[name] = (model, pred)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
        })

    table = pd.DataFrame(results).round(4)
    print(table.to_string(index=False))
    table.to_csv("results.csv", index=False)

    # Pick the robot with the best F1 score
    best_name = table.sort_values("F1", ascending=False).iloc[0]["Model"]
    best_model, best_pred = trained[best_name]
    print(f"\nBest model: {best_name}")

    # Save the best robot's brain
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    # Save a picture of the mistakes (confusion matrix)
    ConfusionMatrixDisplay.from_predictions(
        y_test, best_pred, display_labels=["ham", "spam"], cmap="Blues")
    plt.title(f"Confusion matrix - {best_name}")
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("Saved confusion_matrix.png")


if __name__ == "__main__":
    main()