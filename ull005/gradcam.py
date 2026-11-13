"""Grad-CAM heatmaps: show which image regions drove the CNN's prediction."""
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image


class GradCAM:
    """Grad-CAM for a CNN: weight the last conv feature maps by the
    average gradient of the predicted class score."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()
        output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def generate(self, input_tensor, class_idx=None):
        """Return (heatmap in [0, 1], class index, class probabilities)."""
        with torch.enable_grad():
            self.model.zero_grad()
            logits = self.model(input_tensor)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1))
            logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                            mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        probs = torch.softmax(logits.detach(), dim=1)[0].cpu().tolist()
        return cam, class_idx, probs


def overlay_heatmap(image, cam, alpha=0.4):
    """Blend the heatmap on top of the original image and return a PIL image."""
    base = np.asarray(image.convert("RGB").resize(cam.shape[::-1]),
                      dtype=np.float32) / 255.0
    heat = colormaps["jet"](cam)[..., :3]
    blended = (1 - alpha) * base + alpha * heat
    return Image.fromarray((blended * 255).astype(np.uint8)) 