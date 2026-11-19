"""
Explainable AI module (Grad-CAM).

Grad-CAM shows WHERE in the image the model looked when it made its decision.
It uses the gradients of the predicted class with respect to the last
convolutional feature map of MobileNetV2, and turns them into a heatmap.
Red / yellow areas = the regions that influenced the prediction the most.
"""

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from classifier import PREPROCESS

# The same resize and crop that the model sees (without normalization),
# so that the heatmap lines up exactly with the displayed picture
DISPLAY_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])


class _LogitsWrapper(torch.nn.Module):
    """Makes the Hugging Face model return a plain tensor, which Grad-CAM expects."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits


def _find_target_layer(model):
    """Return the last convolution block of MobileNetV2 (the usual Grad-CAM layer)."""
    backbone = model.mobilenet_v2
    layer = getattr(backbone, "conv_1x1", None)
    if layer is None:
        # Fallback: use the last inverted residual block
        layer = backbone.layer[-1]
    return layer


def explain(classifier, image):
    """
    Create a Grad-CAM heatmap for the top predicted class.

    Returns: (overlay image, predicted class index, probability)
    """
    rgb_image = image.convert("RGB")

    # Picture that will be shown to the user (values from 0 to 1)
    shown = DISPLAY_TRANSFORM(rgb_image)
    rgb_float = np.asarray(shown, dtype=np.float32) / 255.0

    # Tensor that goes into the model
    input_tensor = PREPROCESS(rgb_image).unsqueeze(0)

    wrapped = _LogitsWrapper(classifier.model)
    wrapped.eval()

    # First, find the top predicted class
    with torch.no_grad():
        probabilities = torch.softmax(wrapped(input_tensor), dim=-1)[0]
    class_index = int(probabilities.argmax())
    probability = float(probabilities[class_index])

    # Then, compute the heatmap for that class (gradients are needed here)
    target_layer = _find_target_layer(classifier.model)
    with GradCAM(model=wrapped, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(class_index)],
        )[0]

    # Blend the heatmap with the original picture
    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    return Image.fromarray(overlay), class_index, probability
