"""Grad-CAM heatmaps for the SmartSort ViT.

Grad-CAM asks: which parts of the image made the model choose this class? For a ViT the
image is cut into 14 x 14 = 196 patches, so the heatmap is a 14 x 14 grid that is stretched
to the size of the image.

How it is set up for the Hugging Face ViT (wrapped in model_loader.HFClassifier):
  * the model returns plain logits, which is what the grad-cam library needs;
  * the target layer is the layer-norm in front of the attention of the last encoder block
    (found by name, so it works with older and newer versions of the transformers library);
  * a reshape function turns the 197 tokens into a 14 x 14 grid and drops the [CLS] token.

The heatmap is drawn on the 224 x 224 centre crop, because that is the part of the photo
the model actually sees (resize the short side to 256, then centre-crop 224).
"""
from functools import partial

import numpy as np
import torch
from torchvision import transforms as T


def vit_reshape_transform(tensor: torch.Tensor, grid: int) -> torch.Tensor:
    """(batch, 1 + grid*grid, channels) -> (batch, channels, grid, grid); the [CLS] token is dropped."""
    patches = tensor[:, 1:, :]
    patches = patches.reshape(patches.size(0), grid, grid, patches.size(2))
    return patches.permute(0, 3, 1, 2)


def find_target_layer_name(model) -> str:
    """Name of the layer-norm before the attention in the LAST encoder block of the wrapped ViT.

    The layers are found by name, so this works with both layouts of the transformers library
    (vit.encoder.layer.N.layernorm_before in older versions, vit.layer.N.layernorm_before in newer ones).
    """
    names = [name for name, _ in model.named_modules() if name.endswith("layernorm_before")]
    if not names:
        norm_names = [name for name, _ in model.named_modules() if "norm" in name]
        raise ValueError("Could not find a 'layernorm_before' layer in the model. "
                         f"Layers with 'norm' in their name (last 8): {norm_names[-8:]}")
    return names[-1]          # modules are listed in order, so the last one belongs to the last block


def find_target_layer(model):
    """The layer-norm module that Grad-CAM should explain."""
    return dict(model.named_modules())[find_target_layer_name(model)]


class ViTExplainer:
    """Makes a Grad-CAM overlay for one PIL image. Create it once and reuse it."""

    def __init__(self, model, mean, std, image_size: int = 224):
        from pytorch_grad_cam import GradCAM

        self.model = model.eval()
        self.image_size = image_size
        patch_size = model.model.config.patch_size
        self.grid = image_size // patch_size
        self.view_transform = T.Compose([
            T.Resize(int(image_size * 256 / 224)),
            T.CenterCrop(image_size),
        ])
        self.tensor_transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
        self.cam = GradCAM(
            model=self.model,
            target_layers=[find_target_layer(model)],
            reshape_transform=partial(vit_reshape_transform, grid=self.grid),
        )

    def view(self, pil_image):
        """The square crop (PIL image) that the model sees."""
        return self.view_transform(pil_image.convert("RGB"))

    def explain(self, pil_image, class_index=None):
        """Return (overlay, heat, probabilities, class_index).

        overlay: uint8 array (size, size, 3), the crop with the heatmap on top
        heat: float array (size, size) with values from 0 (ignored) to 1 (most important)
        probabilities: numpy array with one probability per class
        class_index: the class that was explained (the top class if none is given)
        """
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        view = self.view(pil_image)
        rgb = np.asarray(view, dtype=np.float32) / 255.0
        batch = self.tensor_transform(view).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.model(batch).float(), dim=1)[0].cpu().numpy()
        if class_index is None:
            class_index = int(probs.argmax())
        batch = batch.clone().requires_grad_(True)
        heat = self.cam(input_tensor=batch, targets=[ClassifierOutputTarget(class_index)])[0]
        overlay = show_cam_on_image(rgb, heat, use_rgb=True)
        return overlay, heat, probs, int(class_index)

    def close(self) -> None:
        """Remove the hooks that Grad-CAM added to the model. Call it once, when you are finished.

        The Grad-CAM object is dropped here, while Python is still fully running, so that it does not
        print an error message when the program exits.
        """
        if self.cam is not None:
            self.cam.activations_and_grads.release()
            self.cam = None