import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("=" * 50)
print("       SPAM EMAIL DETECTION")
print("=" * 50)

# STEP 1: Load Dataset
csv_files = [
    p for p in Path("dataset").rglob("*.csv")
    if p.is_file()
]

if not csv_files:
    raise FileNotFoundError("Dataset CSV file not found!")

df = pd.read_csv(csv_files[0], encoding="latin-1")

print("\nDataset Loaded:", df.shape)

# STEP 2: Data Cleaning
df["Subject"] = df["Subject"].fillna("")
df["Message"] = df["Message"].fillna("")

# Combine Subject and Message
df["Text"] = df["Subject"] + " " + df["Message"]

# Remove empty text
df = df[df["Text"].str.strip() != ""]

# Clean labels
df["Spam/Ham"] = df["Spam/Ham"].str.strip().str.lower()

# Keep only valid labels
df = df[df["Spam/Ham"].isin(["spam", "ham"])]

print("\nData Cleaning Completed!")
print("Clean Dataset:", df.shape)

print("\nSpam/Ham Distribution:")
print(df["Spam/Ham"].value_counts())

# STEP 3: Prepare Data
X = df["Text"]
y = df["Spam/Ham"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Emails:", len(X_train))
print("Testing Emails:", len(X_test))

# STEP 4: Create Machine Learning Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=10000
    )),
    ("classifier", MultinomialNB())
])

# STEP 5: Train Model
print("\nTraining Model...")

model.fit(X_train, y_train)

print("Model Training Completed!")

# STEP 6: Model Evaluation
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# STEP 7: Save Model
Path("models").mkdir(exist_ok=True)

joblib.dump(model, "models/spam_classifier.joblib")

print("\nModel Saved Successfully!")

print("=" * 50)
print("       TRAINING COMPLETED")
print("=" * 50)