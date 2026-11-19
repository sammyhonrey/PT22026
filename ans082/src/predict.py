"""Prediction for the SmartSort app: check the photo, run the ViT, make the heatmap, give bin advice.

The model is the E3 ViT-B/16 exported to a folder (config.json, model.safetensors,
preprocessor_config.json). If the folder is not on this computer yet, it is downloaded once from
the Hugging Face Hub. After that the app works without internet.

Settings can be changed with environment variables:
    SMARTSORT_MODEL_DIR   folder with the exported model (default: models/smartsort-vit-waste)
    SMARTSORT_THRESHOLD   confidence below this means "not sure" (default: 0.90)
"""
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from bins import DEFAULT_THRESHOLD, decide
from data import CLASSES
from explain import ViTExplainer
from model_loader import HFClassifier

logger = logging.getLogger(__name__)

MODEL_REPO = "ansari-ai-dev/smartsort-vit-waste"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "smartsort-vit-waste"

MIN_SIDE = 64        # smaller photos are rejected
MAX_SIDE = 1024      # larger photos are shrunk (the same limit as the real-photo test set)
BLANK_STD = 4.0      # grey-level spread below this means a blank or one-colour image


class ImageProblem(Exception):
    """The photo cannot be used. The message is written for the user."""


class ModelLoadError(Exception):
    """The model could not be loaded. The message is written for the user."""


@dataclass
class Analysis:
    """Everything the app needs to show for one photo."""

    label: str
    confidence: float
    top3: list            # [(class name, probability), ...] from most to least likely
    probabilities: dict   # class name -> probability, for all classes
    overlay: np.ndarray   # Grad-CAM picture, uint8 (height, width, 3)
    advice: dict          # result of bins.decide(): bin, zh, color, tip, confident, ...


def open_image(source):
    """Open a photo that is given as a file path. A PIL image or None is returned unchanged.

    Raises ImageProblem if the file is not a readable picture (damaged, empty or not an image).
    """
    if source is None or isinstance(source, Image.Image):
        return source
    if not isinstance(source, (str, os.PathLike)):
        raise ImageProblem("I could not read this file as a picture. Please use a JPG or PNG photo.")
    try:
        image = Image.open(source)
        image.load()
    except Exception as error:
        raise ImageProblem("I could not open this file as a picture. It may be damaged or not an image. "
                           "Please use a JPG or PNG photo.") from error
    return image


def prepare_image(image) -> Image.Image:
    """Check a photo and return it as an upright RGB image of at most MAX_SIDE pixels.

    Raises ImageProblem with a friendly message if the photo cannot be used.
    """
    if image is None:
        raise ImageProblem("Please upload or take a photo first.")
    if not isinstance(image, Image.Image):
        raise ImageProblem("I could not read this file as a picture. Please use a JPG or PNG photo.")
    try:
        image = ImageOps.exif_transpose(image).convert("RGB")   # phone photos can be stored sideways
    except Exception as error:                                  # damaged or cut-off files
        raise ImageProblem("This picture file looks damaged and cannot be opened. "
                           "Please try another photo.") from error
    width, height = image.size
    if min(width, height) < MIN_SIDE:
        raise ImageProblem(f"This picture is too small ({width} x {height} pixels). "
                           f"Please use a photo of at least {MIN_SIDE} x {MIN_SIDE} pixels.")
    grey = np.asarray(image.convert("L"), dtype=np.float32)
    if float(grey.std()) < BLANK_STD:
        raise ImageProblem("This picture looks blank or has only one colour. "
                           "Please take a photo of one waste item.")
    image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    return image


def read_normalization(model_dir: Path):
    """Mean and std from preprocessor_config.json (0.5 and 0.5 for this ViT if the file is missing)."""
    path = Path(model_dir) / "preprocessor_config.json"
    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8"))
        return list(config["image_mean"]), list(config["image_std"])
    logger.warning("preprocessor_config.json not found; using mean 0.5 and std 0.5")
    return [0.5] * 3, [0.5] * 3


def download_model(model_dir: Path, repo_id: str = MODEL_REPO) -> None:
    """Download the exported model from the Hugging Face Hub into model_dir (needs internet once)."""
    logger.info("Downloading the model %s to %s (first start only) ...", repo_id, model_dir)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=repo_id, local_dir=str(model_dir))
    except Exception as error:
        raise ModelLoadError(
            "The model files are not on this computer yet and the download did not work. "
            "Connect to the internet once and start the app again, or copy the model folder to: "
            f"{model_dir}") from error


class Predictor:
    """Holds the loaded model. Create it once and call analyse() for every photo."""

    def __init__(self, model, mean, std, threshold: float = DEFAULT_THRESHOLD, image_size: int = 224):
        self.threshold = threshold
        self.explainer = ViTExplainer(model, mean, std, image_size=image_size)

    def close(self) -> None:
        """Release the Grad-CAM hooks. Call once when the app stops."""
        self.explainer.close()

    def analyse(self, image) -> Analysis:
        """Classify one photo (a file path or a PIL image) and explain the decision.

        Raises ImageProblem for photos that cannot be used.
        """
        image = prepare_image(open_image(image))
        overlay, _, probs, index = self.explainer.explain(image)
        order = np.argsort(probs)[::-1]
        label = CLASSES[index]
        confidence = float(probs[index])
        return Analysis(
            label=label,
            confidence=confidence,
            top3=[(CLASSES[i], float(probs[i])) for i in order[:3]],
            probabilities={name: float(probs[i]) for i, name in enumerate(CLASSES)},
            overlay=overlay,
            advice=decide(label, confidence, self.threshold),
        )


def load_predictor(model_dir=None, repo_id: str = MODEL_REPO, threshold=None,
                   image_size: int = 224) -> Predictor:
    """Load the exported model (downloading it first if needed) and return a Predictor."""
    from transformers import AutoModelForImageClassification

    model_dir = Path(model_dir or os.environ.get("SMARTSORT_MODEL_DIR") or DEFAULT_MODEL_DIR)
    if threshold is None:
        threshold = float(os.environ.get("SMARTSORT_THRESHOLD", DEFAULT_THRESHOLD))
    if not (model_dir / "config.json").exists():
        download_model(model_dir, repo_id)
    try:
        hf_model = AutoModelForImageClassification.from_pretrained(str(model_dir))
        mean, std = read_normalization(model_dir)
    except Exception as error:
        raise ModelLoadError(f"The model in {model_dir} could not be loaded. "
                             "Delete that folder and start the app again to download it fresh.") from error
    names = [hf_model.config.id2label[i] for i in range(hf_model.config.num_labels)]
    if names != list(CLASSES):
        raise ModelLoadError("The model has different class names than this app expects: "
                             f"{names}. Delete {model_dir} and start the app again.")
    logger.info("Model loaded from %s (threshold %.2f)", model_dir, threshold)
    return Predictor(HFClassifier(hf_model).eval(), mean, std, threshold, image_size)