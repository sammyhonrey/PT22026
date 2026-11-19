import json
import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = Path("housing_clean.csv")
MODEL_PATH = Path("models/house_price_model.joblib")
PIPELINE_PATH = Path("models/preprocessing_pipeline.joblib")

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["median_house_value"])
y = df["median_house_value"]

# Create a stratified 80/20 evaluation split.
# The original model training indices have not been verified.
df["income_cat"] = pd.cut(
    df["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)

splitter = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(df, df["income_cat"])
)

X_test = X.iloc[test_idx].copy()
y_test = y.iloc[test_idx].copy()

model = joblib.load(MODEL_PATH)
pipeline = joblib.load(PIPELINE_PATH)

X_test_processed = pipeline.transform(X_test)
predictions = model.predict(X_test_processed)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

results = {
    "model": "Pre-trained Random Forest Regressor",
    "dataset_rows": len(df),
    "training_samples_in_split": len(train_idx),
    "testing_samples": len(test_idx),
    "MAE_USD": round(float(mae), 2),
    "RMSE_USD": round(float(rmse), 2),
    "R2_score": round(float(r2), 4),
    "evaluation_note": "Uses reconstructed stratified split; original model training indices are not independently verified."
}

print("\nMODEL EVALUATION RESULTS")
print("=" * 40)

for key, value in results.items():
    print(f"{key}: {value}")

with open("evaluation_results.json", "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\nSaved: evaluation_results.json")
