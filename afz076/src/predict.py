"""Single-image inference with temperature-scaled confidence and a reject ("not sure") option."""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from src.calibration import softmax_np
from src.classes import CLASS_NAMES
from src.config import DEFAULT_REJECT_THRESHOLD, IMG_SIZE
from src.data import to_tensor
from src.gradcam import grad_cam, overlay_heatmap
from src.models import gradcam_layer
from src.trainer import get_device, load_checkpoint


@dataclass
class Prediction:
    top: List[dict]                 # [{"class_id", "name", "probability"}, ...] best first
    accepted: bool                  # False -> the model says "I'm not sure"
    confidence: float
    heatmap: Optional[np.ndarray] = None   # uint8 RGB overlay (64x64 or original size)


class SignPredictor:
    def __init__(self, ckpt_path: str):
        self.device = get_device()
        self.model, self.ckpt = load_checkpoint(ckpt_path, self.device)
        self.name = self.ckpt["model_name"]
        self.temperature = float(self.ckpt.get("temperature", 1.0))

    @staticmethod
    def prepare(image: Image.Image) -> np.ndarray:
        """Any PIL image -> 64x64 uint8 RGB array (the same plain resize the training images got)."""
        return np.asarray(image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.uint8)

    def predict(self, img64: np.ndarray, threshold: float = DEFAULT_REJECT_THRESHOLD,
                top_k: int = 3, with_heatmap: bool = False) -> Prediction:
        x = to_tensor(img64).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x).float().cpu().numpy()
        probs = softmax_np(logits, self.temperature)[0]
        order = np.argsort(-probs)[:top_k]
        top = [{"class_id": int(i), "name": CLASS_NAMES[i], "probability": float(probs[i])} for i in order]
        heat = None
        if with_heatmap:
            cam = grad_cam(self.model, gradcam_layer(self.model, self.name), x.cpu().to(self.device), int(order[0]))
            heat = overlay_heatmap(img64, cam)
        return Prediction(top=top, accepted=bool(probs[order[0]] >= threshold),
                          confidence=float(probs[order[0]]), heatmap=heat)
