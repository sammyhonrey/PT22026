"""
Inference pipeline: validate -> preprocess -> CNN + Grad-CAM -> report.

Kept separate from the UI so it can be reused by scripts and tests.
"""
import logging
import math
import time
from dataclasses import dataclass, field

from PIL import Image

from src import config
from src.gradcam import overlay_heatmap, predict_with_gradcam
from src.model_loader import get_model
from src.preprocessing import prepare_display_image, preprocess, validate_image

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    label: str
    prob_pneumonia: float
    confidence: float
    threshold: float
    heatmap: Image.Image
    resized: Image.Image
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def probabilities(self) -> dict[str, float]:
        return {config.CLASS_NAMES[0]: 1 - self.prob_pneumonia,
                config.CLASS_NAMES[1]: self.prob_pneumonia}


def _logit(p: float) -> float:
    p = min(max(p, 1e-7), 1 - 1e-7)
    return math.log(p / (1 - p))


def _threshold_warnings(prob: float, threshold: float, label: str) -> list[str]:
    """Warnings that stay correct for any decision threshold, not only 0.5."""
    warnings = []
    # Close to the decision boundary, measured on the logit scale so the band
    # is 0.35-0.65 at the default threshold of 0.5 and narrower near 0 or 1.
    if abs(_logit(prob) - _logit(threshold)) < config.UNCERTAIN_LOGIT_MARGIN:
        warnings.append(f"The score P(Pneumonia) = {prob:.3f} is close to the decision "
                        f"threshold of {threshold:.2f}. Treat this result with extra caution.")
    # With a threshold other than 0.5 the label can go against the more likely class.
    if label == config.CLASS_NAMES[0] and prob >= 0.5:
        warnings.append(f"The model leans toward pneumonia (P = {prob:.3f}), but this is below "
                        f"your threshold of {threshold:.2f}, so the result is Normal.")
    elif label == config.CLASS_NAMES[1] and prob < 0.5:
        warnings.append(f"The model leans toward normal (P(Pneumonia) = {prob:.3f}), but this is "
                        f"above your threshold of {threshold:.2f}, so the result is Pneumonia.")
    return warnings


def analyze(image: Image.Image, threshold: float = config.DEFAULT_THRESHOLD) -> Prediction:
    """Run the full pipeline on one PIL image."""
    warnings = validate_image(image)          # may raise InvalidImageError
    start = time.perf_counter()

    model = get_model()
    batch = preprocess(image)
    prob, cam = predict_with_gradcam(model, batch, threshold)

    label = config.CLASS_NAMES[1] if prob >= threshold else config.CLASS_NAMES[0]
    confidence = prob if label == config.CLASS_NAMES[1] else 1 - prob
    warnings += _threshold_warnings(prob, threshold, label)

    resized = prepare_display_image(image)
    heatmap = overlay_heatmap(resized, cam)
    elapsed = time.perf_counter() - start
    logger.info("Prediction %s p=%.4f (%.2fs)", label, prob, elapsed)

    return Prediction(label, prob, confidence, threshold, heatmap, resized,
                      warnings, elapsed)
