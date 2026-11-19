from pathlib import Path

import joblib
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "house_price_model.joblib"
PIPELINE_PATH = BASE_DIR / "models" / "preprocessing_pipeline.joblib"

FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity",
]

OCEAN_CATEGORIES = [
    "<1H OCEAN",
    "INLAND",
    "ISLAND",
    "NEAR BAY",
    "NEAR OCEAN",
]

_cache = {}


def load_artifacts():
    if not _cache:
        missing = [
            p.name for p in (MODEL_PATH, PIPELINE_PATH)
            if not p.exists()
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing model file(s): {', '.join(missing)}. "
                "Run `python download_model.py` first."
            )

        _cache["model"] = joblib.load(MODEL_PATH)
        _cache["pipeline"] = joblib.load(PIPELINE_PATH)

    return _cache["model"], _cache["pipeline"]


def validate_inputs(data):
    for name in FEATURES:
        if data.get(name) is None:
            raise ValueError(f"Please provide a value for '{name}'.")

    if data["ocean_proximity"] not in OCEAN_CATEGORIES:
        raise ValueError("Unknown ocean proximity category.")

    numeric_features = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
    ]

    for name in numeric_features:
        try:
            value = float(data[name])
        except (TypeError, ValueError):
            raise ValueError(f"'{name}' must be a valid number.")

        if not np.isfinite(value):
            raise ValueError(f"'{name}' must be a finite number.")

    if not -125 <= float(data["longitude"]) <= -113:
        raise ValueError("Longitude is outside the California range.")

    if not 32 <= float(data["latitude"]) <= 42.5:
        raise ValueError("Latitude is outside the California range.")

    if not 1 <= float(data["housing_median_age"]) <= 52:
        raise ValueError("Housing age must be between 1 and 52.")

    if min(
        float(data["total_rooms"]),
        float(data["total_bedrooms"]),
        float(data["population"]),
        float(data["households"]),
    ) <= 0:
        raise ValueError("Room and population values must be positive.")

    if float(data["median_income"]) <= 0:
        raise ValueError("Median income must be positive.")

    if float(data["total_bedrooms"]) > float(data["total_rooms"]):
        raise ValueError("Bedrooms cannot exceed total rooms.")

    if float(data["households"]) > float(data["population"]):
        raise ValueError("Households cannot exceed population.")


def predict_price(data):
    validate_inputs(data)

    model, pipeline = load_artifacts()

    row = {
        name: (
            float(data[name])
            if name != "ocean_proximity"
            else data[name]
        )
        for name in FEATURES
    }

    processed = pipeline.transform(
        pd.DataFrame([row], columns=FEATURES)
    )

    return float(model.predict(processed)[0])