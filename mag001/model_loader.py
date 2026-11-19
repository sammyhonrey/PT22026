"""Model loading utilities for the hybrid Food-101 classifier.

This module loads a custom PyTorch checkpoint (best_model.pth) and 
real_class_mapping.json from Hugging Face.

The model is a hybrid ResNet50 + DeiT-Base network with cross-attention fusion.
This script:
1. Downloads the weights and class mapping.
2. Rebuilds the exact hybrid architecture.
3. Loads the state dict, handling key name differences between transformers 4.x and 5.x.
4. Returns (model, transform, idx_to_label) for use in the Gradio app and evaluation script.
"""

# Constants for the model on Hugging Face Hub
# IMAGE_SIZE = 224 is the standard input size for both ResNet50 and DeiT
from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torchvision.models as tv_models
from huggingface_hub import hf_hub_download
from torchvision import transforms
from transformers import DeiTConfig, DeiTModel

LOGGER = logging.getLogger(__name__)

REPO_ID = os.getenv(
    "FOOD_CLASSIFIER_REPO_ID", "codealchemist01/food-image-classifier-hybrid"
)
CHECKPOINT_FILENAME = "best_model.pth"
CLASS_MAPPING_FILENAME = "real_class_mapping.json"
FALLBACK_CLASS_NAMES_FILENAME = "food101_class_names.json"

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# Network definition
class CNNBranch(nn.Module):

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_early_layers: bool = True,
        dropout: float = 0.3,
        feature_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim

        if backbone != "resnet50":
            raise ValueError(f"Unsupported backbone: {backbone}")

        try:
            resnet = tv_models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        except TypeError:  # very old torchvision
            resnet = tv_models.resnet50(pretrained=pretrained)

     # I removed the last two layers (avgpool and fc) to get the raw 7x7 feature map.
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        if freeze_early_layers:
            self._freeze_early_layers()

        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.feature_proj = nn.Sequential(
            nn.Conv2d(2048, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
# Freeze the first 6 layers so they don't get retrained (saves memory)
    def _freeze_early_layers(self) -> None: 
        for index, child in enumerate(self.backbone.children()):
            if index < 6:
                for parameter in child.parameters():
                    parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        features = self.adaptive_pool(features)
        spatial_features = self.feature_proj(features)
        global_features = self.global_pool(spatial_features).flatten(1)
        global_features = self.feature_head(global_features)
        return spatial_features, global_features

class ViTBranch(nn.Module):

    def __init__(
        self,
        model_name: str = "facebook/deit-base-distilled-patch16-224",
        pretrained: bool = True,
        freeze_early_layers: bool = True,
        dropout: float = 0.1,
        feature_dim: int = 768,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.model_name = model_name

        # The ViT branch uses DeiT-Base, a distilled Vision Transformer.
        # DeiT splits the image into 16x16 patches and attends to them globally.
        config = DeiTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            image_size=IMAGE_SIZE,
            patch_size=16,
            qkv_bias=True,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        self.vit = DeiTModel(config)

        if freeze_early_layers:
            self._freeze_early_layers()

        self.hidden_size = self.vit.config.hidden_size
        self.num_patches = (IMAGE_SIZE // 16) ** 2

        self.feature_proj = nn.Sequential(
            nn.Linear(self.hidden_size, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.spatial_proj = nn.Sequential(
            nn.Linear(self.hidden_size, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.feature_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _freeze_early_layers(self) -> None:
        # transformers 4.x calls this ``encoder.layer``; 5.x calls it
        # ``layers``.  Support both so the module can be built anywhere.
        layers = getattr(self.vit, "encoder", None)
        layers = getattr(layers, "layer", None) if layers is not None else None
        if layers is None:
            layers = getattr(self.vit, "layers", [])
        for index, layer in enumerate(layers):
            if index < 8:
                for parameter in layer.parameters():
                    parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.vit(pixel_values=x)
        last_hidden_states = getattr(outputs, "last_hidden_state", None)
        if last_hidden_states is None:
            last_hidden_states = outputs[0]

        # The CLS token (index 0) captures the global image representation
        cls_token = last_hidden_states[:, 0]
        # The remaining tokens are the spatial patch features
        patch_tokens = last_hidden_states[:, 1:]

        global_features = self.feature_proj(cls_token)
        spatial_features = self.spatial_proj(patch_tokens)
        global_features = self.feature_head(global_features)
        return spatial_features, global_features

class AdaptiveFusionModule(nn.Module):
    # Cross-attention fusion of CNN and ViT features.

    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.2,
        spatial_size: int = 7,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.spatial_size = spatial_size

        self.cnn_to_vit_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.vit_to_cnn_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cnn_spatial_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.vit_spatial_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.global_fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.adaptive_weight = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1),
        )
        self.final_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        cnn_spatial: torch.Tensor,
        cnn_global: torch.Tensor,
        vit_spatial: torch.Tensor,
        vit_global: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cnn_spatial_seq = cnn_spatial.flatten(2).transpose(1, 2)
        cnn_spatial_proj = self.cnn_spatial_proj(cnn_spatial_seq)
        vit_spatial_proj = self.vit_spatial_proj(vit_spatial)
        # Cross-attention: CNN features query the ViT features
        # This lets the CNN learn which spatial regions the ViT thinks are important
        cnn_attended, _ = self.cnn_to_vit_attention(
            query=cnn_spatial_proj, key=vit_spatial_proj, value=vit_spatial_proj
        )
        vit_attended, _ = self.vit_to_cnn_attention(
            query=vit_spatial_proj, key=cnn_spatial_proj, value=cnn_spatial_proj
        )

        combined_spatial = torch.cat(
            [cnn_attended + cnn_spatial_proj, vit_attended + vit_spatial_proj], dim=1
        )
        fused_spatial, _ = self.self_attention(
            query=combined_spatial, key=combined_spatial, value=combined_spatial
        )

        global_concat = torch.cat([cnn_global, vit_global], dim=-1)
        fused_global_base = self.global_fusion(global_concat)

        # The adaptive weighting learns how much to trust CNN vs ViT
        # weights shape: (batch, 2) -> [cnn_weight, vit_weight]
        weights = self.adaptive_weight(global_concat)
        cnn_weight = weights[:, 0:1]
        vit_weight = weights[:, 1:2]
        fused_global = (
            cnn_weight * cnn_global + vit_weight * vit_global + fused_global_base
        ) / 2
        fused_global = self.final_proj(fused_global)
        return fused_spatial, fused_global

        # The full hybrid model:
        # 1. CNNBranch -> spatial + global features
        # 2. ViTBranch -> spatial + global features
        # 3. AdaptiveFusionModule -> fused features
        # 4. Classifier (2-layer MLP) -> 101 class logits
class HybridFoodClassifier(nn.Module):

    def __init__(
        self,
        num_classes: int = 101,
        feature_dim: int = 768,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        pretrained: bool = False,
        freeze_early_layers: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.cnn_branch = CNNBranch(
            pretrained=pretrained,
            freeze_early_layers=freeze_early_layers,
            dropout=dropout,
            feature_dim=feature_dim,
        )
        self.vit_branch = ViTBranch(
            pretrained=pretrained,
            freeze_early_layers=freeze_early_layers,
            dropout=dropout,
            feature_dim=feature_dim,
        )
        self.fusion_module = AdaptiveFusionModule(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.cnn_aux_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.vit_aux_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in [
            self.classifier,
            self.cnn_aux_classifier,
            self.vit_aux_classifier,
        ]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        cnn_spatial, cnn_global = self.cnn_branch(x)
        vit_spatial, vit_global = self.vit_branch(x)
        fused_spatial, fused_global = self.fusion_module(
            cnn_spatial, cnn_global, vit_spatial, vit_global
        )
        logits = self.classifier(fused_global)

        output: Dict[str, torch.Tensor] = {"logits": logits}
        if return_features:
            output.update(
                {
                    "cnn_spatial": cnn_spatial,
                    "cnn_global": cnn_global,
                    "vit_spatial": vit_spatial,
                    "vit_global": vit_global,
                    "fused_spatial": fused_spatial,
                    "fused_global": fused_global,
                }
            )
        return output

# The checkpoint may have been saved with different wrapper prefixes (module.)
# or different transformers versions (4.x vs 5.x), so we convert the keys here.
def _torch_load(path: str | Path):
    """Load a checkpoint on CPU, supporting old and new PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0 does not have weights_only
        return torch.load(path, map_location="cpu")


def _extract_state_dict(checkpoint) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """Return ``(state_dict, metadata)`` from common checkpoint layouts."""
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model_state", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict) and value:
                return value, checkpoint
        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint, {}
    raise RuntimeError(
        "Unrecognised checkpoint format. Expected a dict containing "
        "'model_state_dict' or a raw state dict."
    )


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove common wrapper prefixes (DataParallel, torch.compile, Lightning)."""
    prefixes = ("module.", "_orig_mod.", "model.")
    changed = True
    while changed and state_dict:
        changed = False
        for prefix in prefixes:
            if all(key.startswith(prefix) for key in state_dict):
                state_dict = {
                    key[len(prefix) :]: value for key, value in state_dict.items()
                }
                changed = True
                break
    return state_dict


def _convert_state_dict_keys(
    state_dict: Dict[str, torch.Tensor], model_keys
) -> Dict[str, torch.Tensor]:
    """Translate DeiT key names between transformers 4.x and 5.x.

    transformers 4.x:
        vit_branch.vit.encoder.layer.0.attention.attention.query.weight
        vit_branch.vit.encoder.layer.0.attention.output.dense.weight
        vit_branch.vit.encoder.layer.0.intermediate.dense.weight
        vit_branch.vit.encoder.layer.0.output.dense.weight

    transformers 5.x:
        vit_branch.vit.layers.0.attention.q_proj.weight
        vit_branch.vit.layers.0.attention.o_proj.weight
        vit_branch.vit.layers.0.mlp.fc1.weight
        vit_branch.vit.layers.0.mlp.fc2.weight
    """
        # Check if the installed transformers version uses the new (5.x) or old (4.x) naming
    model_uses_layers = any(key.startswith("vit_branch.vit.layers.") for key in model_keys)
    model_uses_qkv_proj = any(".attention.q_proj." in key for key in model_keys)

    converted: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key

        if model_uses_layers:
            new_key = new_key.replace("vit_branch.vit.encoder.layer.", "vit_branch.vit.layers.")
        else:
            new_key = new_key.replace("vit_branch.vit.layers.", "vit_branch.vit.encoder.layer.")

        if model_uses_qkv_proj:
            new_key = new_key.replace(".attention.attention.query.", ".attention.q_proj.")
            new_key = new_key.replace(".attention.attention.key.", ".attention.k_proj.")
            new_key = new_key.replace(".attention.attention.value.", ".attention.v_proj.")
            new_key = new_key.replace(".attention.output.dense.", ".attention.o_proj.")
            new_key = new_key.replace(".intermediate.dense.", ".mlp.fc1.")
            new_key = new_key.replace(".output.dense.", ".mlp.fc2.")
        else:
            new_key = new_key.replace(".attention.q_proj.", ".attention.attention.query.")
            new_key = new_key.replace(".attention.k_proj.", ".attention.attention.key.")
            new_key = new_key.replace(".attention.v_proj.", ".attention.attention.value.")
            new_key = new_key.replace(".attention.o_proj.", ".attention.output.dense.")
            new_key = new_key.replace(".mlp.fc1.", ".intermediate.dense.")
            new_key = new_key.replace(".mlp.fc2.", ".output.dense.")

        converted[new_key] = value
    return converted


def _infer_num_classes(state_dict: Dict[str, torch.Tensor], metadata: Dict) -> int:
    value = metadata.get("num_classes") if isinstance(metadata, dict) else None
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass

    for key in ("classifier.4.weight", "classifier.2.weight"):
        tensor = state_dict.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
            return int(tensor.shape[0])

    LOGGER.warning("Could not infer num_classes from checkpoint; assuming 101.")
    return 101


def _load_class_mapping(path: str | Path, num_classes: int) -> Dict[int, str]:
    """Parse the class mapping JSON into ``{index: category_name}``."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    raw = None
    if isinstance(data, dict):
        if "idx_to_class" in data:
            raw = data["idx_to_class"]
        elif "real_class_names" in data:
            raw = {str(i): name for i, name in enumerate(data["real_class_names"])}
        elif "food101_classes" in data:
            raw = {str(i): name for i, name in enumerate(data["food101_classes"])}
        elif data and all(str(key).lstrip("-").isdigit() for key in data):
            raw = data

    if raw is None:
        raise RuntimeError(
            f"Could not find a class mapping in {path}. Expected 'idx_to_class', "
            "'real_class_names', or 'food101_classes'."
        )

    mapping = {int(key): str(value) for key, value in raw.items()}
    if len(mapping) != num_classes:
        LOGGER.warning(
            "Class mapping has %d entries but the model has %d classes.",
            len(mapping),
            num_classes,
        )
    return mapping


def _download(repo_id: str, filename: str, cache_dir: str | None) -> str:
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
    except Exception as exc:  # network, auth, missing file, ...
        raise RuntimeError(
            f"Failed to download '{filename}' from '{repo_id}'. "
            "Check your internet connection and Hugging Face access."
        ) from exc

# Public API
def load_model(
    device: str | torch.device | None = None,
    repo_id: str = REPO_ID,
    cache_dir: str | None = None,
) -> Tuple[nn.Module, transforms.Compose, Dict[int, str]]:
    """Download and load the pre-trained hybrid Food-101 classifier.

    Parameters
    ----------
    device:
        ``"cpu"``, ``"cuda"``, ``"mps"`` or a ``torch.device``.  Defaults to
        CUDA when available, otherwise CPU.
    repo_id:
        Hugging Face model repository.  Defaults to the model specified in
        the assignment.
    cache_dir:
        Optional Hugging Face cache directory.

    Returns
    -------
    (model, transform, idx_to_label)
        ``model`` is in eval mode with gradients disabled, ``transform`` is
        the torchvision preprocessing pipeline (224x224, ImageNet
        normalisation), and ``idx_to_label`` maps model output indices to
        Food-101 category names.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    LOGGER.info("Downloading model files from %s ...", repo_id)
    checkpoint_path = _download(repo_id, CHECKPOINT_FILENAME, cache_dir)
    mapping_path = _download(repo_id, CLASS_MAPPING_FILENAME, cache_dir)

    LOGGER.info("Loading checkpoint %s", checkpoint_path)
    checkpoint = _torch_load(checkpoint_path)
    state_dict, metadata = _extract_state_dict(checkpoint)
    state_dict = _strip_module_prefix(state_dict)
    num_classes = _infer_num_classes(state_dict, metadata)

    idx_to_label = _load_class_mapping(mapping_path, num_classes)
    class_names = [idx_to_label.get(i, f"class_{i}") for i in range(num_classes)]

    LOGGER.info("Building model (%d classes) ...", num_classes)
    model = HybridFoodClassifier(
        num_classes=num_classes,
        pretrained=False,
        freeze_early_layers=False,
    )

    # First try an exact load (handles checkpoints saved with the same
    # transformers version that is installed here).
    try:
        model.load_state_dict(state_dict, strict=True)
        LOGGER.info("Checkpoint loaded with exact key match.")
    except RuntimeError as first_error:
        LOGGER.debug("Exact load failed: %s", first_error)
        converted = _convert_state_dict_keys(state_dict, set(model.state_dict().keys()))
        try:
            model.load_state_dict(converted, strict=True)
            LOGGER.info("Checkpoint loaded after DeiT key conversion.")
        except RuntimeError:
            missing, unexpected = model.load_state_dict(converted, strict=False)
            # Auxiliary heads are only used during training, and the DeiT
            # pooler is not used for inference, so those may legitimately be
            # absent from a checkpoint.  Anything else missing indicates a
            # real architecture/key-mapping mismatch.
            allowed_missing = (
                "vit_branch.vit.pooler.",
                "cnn_aux_classifier.",
                "vit_aux_classifier.",
            )
            critical_missing = [
                key for key in missing if not key.startswith(allowed_missing)
            ]
            if critical_missing:
                raise RuntimeError(
                    "Could not load the checkpoint into the hybrid model. "
                    f"Missing keys: {critical_missing[:10]} "
                    f"(total missing: {len(missing)})."
                ) from first_error
            if missing or unexpected:
                LOGGER.warning(
                    "Loaded checkpoint with %d missing and %d unexpected keys "
                    "(allowed non-inference keys only).",
                    len(missing),
                    len(unexpected),
                )

    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    # Define the preprocessing pipeline: resize to 224x224, convert to tensor, and normalize
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    # Convenient attributes for callers.
    model.device = device  # type: ignore[attr-defined]
    model.idx_to_label = idx_to_label  # type: ignore[attr-defined]
    model.class_names = class_names  # type: ignore[attr-defined]

    LOGGER.info("Model ready on %s.", device)
    return model, transform, idx_to_label

# tells Python exactly what to expose when another file does `from model_loader import *`.
__all__ = [ 
    "REPO_ID",
    "CHECKPOINT_FILENAME",
    "CLASS_MAPPING_FILENAME",
    "IMAGE_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "CNNBranch",
    "ViTBranch",
    "AdaptiveFusionModule",
    "HybridFoodClassifier",
    "load_model",
]
