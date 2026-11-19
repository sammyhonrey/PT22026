"""Minimal Grad-CAM (Selvaraju et al., 2017): which image regions pushed the model towards a class."""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def grad_cam(model: nn.Module, layer: nn.Module, x: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
    """Return a heatmap in [0, 1] with the same height/width as `x` (shape H x W) for a single image.

    x: normalised tensor of shape (1, 3, H, W).  The model is put in eval mode.
    """
    model.eval()
    store = {}

    def fwd_hook(_, __, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))   # gradient w.r.t. the layer output

    handle = layer.register_forward_hook(fwd_hook)
    try:
        x = x.clone().requires_grad_(True)
        logits = model(x)
        cls = int(logits.argmax(1)) if target_class is None else int(target_class)
        model.zero_grad()
        logits[0, cls].backward()
    finally:
        handle.remove()

    act, grad = store["act"].detach(), store["grad"].detach()
    weights = grad.mean(dim=(2, 3), keepdim=True)                     # importance of each feature map
    cam = F.relu((weights * act).sum(1, keepdim=True))                 # weighted sum, keep positive evidence
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    cam -= cam.min()
    return cam / cam.max() if cam.max() > 0 else cam


def overlay_heatmap(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a red-yellow heatmap onto a uint8 RGB image (any size); returns uint8 RGB."""
    h, w = img.shape[:2]
    cam_img = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR),
                         dtype=np.float32) / 255.0
    heat = np.stack([cam_img, cam_img ** 2, np.zeros_like(cam_img)], axis=-1) * 255   # red -> yellow-ish
    out = (1 - alpha) * img.astype(np.float32) + alpha * heat
    return np.clip(out, 0, 255).astype(np.uint8)
