"""
Plant disease classifier module.

Loads an image classification model (MobileNetV2) and predicts the plant
disease from a leaf image. Two models are supported:
  - "ours":  AgroGuard Model 3, fine-tuned on PlantVillage (lab) + PlantDoc (field)
  - "ready": a ready-made baseline trained on PlantVillage only
"""

from pathlib import Path

import torch
from torchvision import transforms
from transformers import AutoModelForImageClassification

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
OUR_MODEL_DIR = BASE_DIR / "models" / "model3"  # Local weights (not stored in GitHub)
READY_MODEL_NAME = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"

MODEL_CHOICES = {
    "AgroGuard Model 3 (ours: lab + field data)": "ours",
    "Ready-made baseline (lab data only)": "ready",
}

# The 38 PlantVillage classes in the order used by BOTH models.
# Using one shared list keeps the labels consistent for the knowledge base.
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# Same preprocessing that was used for evaluation during training:
# resize, center crop to 224x224, normalize pixel values to [-1, 1]
PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class PlantClassifier:
    """A wrapper around one trained image classification model."""

    def __init__(self, source: str):
        self.model = AutoModelForImageClassification.from_pretrained(source)
        self.model.eval()  # Inference mode (no training)

    def predict(self, image, top_k: int = 5):
        """Return the top-k predictions as a list of (class label, probability)."""
        pixel_values = PREPROCESS(image.convert("RGB")).unsqueeze(0)

        # No gradients are needed for prediction, which saves memory and time
        with torch.no_grad():
            logits = self.model(pixel_values=pixel_values).logits

        # Convert raw scores into probabilities that sum to 1
        probabilities = torch.softmax(logits, dim=-1)[0]
        top_probs, top_ids = torch.topk(probabilities, k=top_k)

        return [(CLASS_NAMES[idx.item()], float(prob)) for prob, idx in zip(top_probs, top_ids)]


# Models are loaded only once and then reused
_cache = {}


def get_classifier(key: str) -> PlantClassifier:
    """Return the classifier for 'ours' or 'ready', loading it on first use."""
    if key not in _cache:
        if key == "ours":
            if not OUR_MODEL_DIR.exists():
                raise FileNotFoundError(
                    f"Model weights not found in '{OUR_MODEL_DIR}'. "
                    "See the README for download instructions."
                )
            _cache[key] = PlantClassifier(str(OUR_MODEL_DIR))
        else:
            _cache[key] = PlantClassifier(READY_MODEL_NAME)
    return _cache[key]
