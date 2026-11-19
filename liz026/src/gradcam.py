"""
Grad-CAM (Gradient-weighted Class Activation Mapping).

Shows WHICH regions of the X-ray drove the prediction, which the original
project did not provide.

How it works (Selvaraju et al., 2017):
  1. Run the image through the MobileNetV2 backbone and keep the last
     convolutional feature maps A (7 x 7 x 1280).
  2. Run A through the classification head to get the pneumonia logit z
     (the score before the final sigmoid).
  3. Compute dz/dA. Average the gradient over the 7x7 grid -> one weight
     per channel (how important that channel is for "pneumonia").
  4. Weighted sum of the channels, ReLU, normalize -> 7x7 heat-map,
     upsampled to the image size and overlaid in color.

For a "Normal" prediction we explain -z instead, so the map
always highlights evidence for the predicted class.
"""
import numpy as np
from PIL import Image

from src import config
import keras


def _split_model(model):
    """Return (backbone, head_layers) from the saved functional model."""
    backbone = model.get_layer(config.BACKBONE_LAYER)
    idx = model.layers.index(backbone)
    head = model.layers[idx + 1:]
    return backbone, head


def _head_logit(head, feats):
    """Run the classification head but stop before the final sigmoid.

    Gradients of the raw logit do not vanish when the model is very
    confident (sigmoid saturated at 0 or 1), so the heat-map stays useful.
    """
    out = feats
    for layer in head[:-1]:
        out = layer(out, training=False)
    last = head[-1]  # Dense(1, activation="sigmoid")
    return keras.ops.matmul(out, last.kernel) + last.bias


def predict_with_gradcam(model, batch: np.ndarray, threshold: float = 0.5):
    """
    Single forward + backward pass.

    Returns (pneumonia_probability: float, heatmap: np.ndarray 7x7 in [0,1]).
    """
    backbone, head = _split_model(model)
    backend = keras.backend.backend()

    if backend == "torch":
        import torch

        x = torch.as_tensor(batch, dtype=torch.float32)
        with torch.no_grad():
            feats = backbone(x, training=False)
        feats = feats.detach().clone().requires_grad_(True)
        logit = _head_logit(head, feats)[0, 0]
        prob = float(torch.sigmoid(logit).detach())
        target = logit if prob >= threshold else -logit
        target.backward()
        grads = feats.grad[0].cpu().numpy()
        fmap = feats[0].detach().cpu().numpy()

    elif backend == "tensorflow":
        import tensorflow as tf

        x = tf.convert_to_tensor(batch, dtype=tf.float32)
        feats = backbone(x, training=False)
        with tf.GradientTape() as tape:
            tape.watch(feats)
            logit = _head_logit(head, feats)[0, 0]
            prob = float(tf.sigmoid(logit))
            target = logit if prob >= threshold else -logit
        grads = tape.gradient(target, feats)[0].numpy()
        fmap = feats[0].numpy()
    else:
        raise RuntimeError(f"Grad-CAM not implemented for backend '{backend}'")

    weights = grads.mean(axis=(0, 1))                 # (1280,)
    cam = np.maximum((fmap * weights).sum(axis=-1), 0)  # (7, 7)
    if cam.max() > 0:
        cam = cam / cam.max()
    return prob, cam


def _jet(values: np.ndarray) -> np.ndarray:
    """Tiny 'jet' color-map (avoids a matplotlib dependency). values in [0,1]."""
    v = np.clip(values, 0, 1)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """Upsample the 7x7 CAM to the image size and blend it over the X-ray."""
    image = image.convert("RGB")
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        image.size, Image.Resampling.BICUBIC)
    cam_up = np.asarray(cam_img, dtype=np.float32) / 255.0
    color = _jet(cam_up).astype(np.float32)
    base = np.asarray(image, dtype=np.float32)
    # Blend more strongly where activation is high; keep cold regions readable.
    a = (alpha * (0.3 + 0.7 * cam_up))[..., None]
    out = base * (1 - a) + color * a
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))
