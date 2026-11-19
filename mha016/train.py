from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Get real movie reviews (downloaded automatically)
print("Downloading dataset...")
data = load_dataset("stanfordnlp/imdb")

# Use smaller parts so training is fast
train = data["train"].shuffle(seed=42).select(range(10000))
test = data["test"].shuffle(seed=42).select(range(2000))

# 2. Turn words into numbers (TF-IDF)
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train["text"])
X_test = vectorizer.transform(test["text"])

# 3. Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, train["label"])

# 4. Check how good it is
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(test["label"], predictions))
print(classification_report(test["label"], predictions,
                            target_names=["NEGATIVE", "POSITIVE"]))

# 5. Save the model so the app can use it
joblib.dump((vectorizer, model), "review_model.joblib")
print("Saved review_model.joblib")