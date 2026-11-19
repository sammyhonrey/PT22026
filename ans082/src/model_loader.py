"""Model construction: ResNet-50 baseline (torchvision) and ViT (Hugging Face)."""
from torch import nn
from torchvision import models

from data import CLASSES, get_normalization


class HFClassifier(nn.Module):
    """Wraps a Hugging Face image classifier so that model(images) returns a logits tensor."""

    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, images):
        return self.model(pixel_values=images).logits


def build_resnet50(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
    """ImageNet-pre-trained ResNet-50 with a new classification layer.

    Returns (model, mean, std). Mean and std come from the pretrained weights' own
    preprocessing preset, not from guessed numbers.
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    preset = weights.transforms()
    model = models.resnet50(weights=weights if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc.")
    return model, list(preset.mean), list(preset.std)


def build_vit(hf_name: str, num_classes: int, freeze_backbone: bool = False,
              mean=None, std=None):
    """Hugging Face ViT with a new classification head.

    hf_name is a Hub name such as google/vit-base-patch16-224 (or a local folder).
    If mean and std are not given they are read from the model's own image processor.
    Returns (model, mean, std).
    """
    from transformers import AutoModelForImageClassification

    id2label = {i: name for i, name in enumerate(CLASSES)}
    label2id = {name: i for i, name in id2label.items()}
    hf_model = AutoModelForImageClassification.from_pretrained(
        hf_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # the 1000-class ImageNet head is replaced
    )
    model = HFClassifier(hf_model)
    # keep our class names in the model's own config (used when the model is exported)
    hf_model.config.id2label = id2label
    hf_model.config.label2id = label2id
    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("model.classifier.")
    if mean is None or std is None:
        mean, std = get_normalization(hf_name)
    return model, list(mean), list(std)


def build_model(model_cfg: dict, num_classes: int = len(CLASSES)):
    """Build the model named in the config. Returns (model, mean, std)."""
    name = model_cfg["name"]
    if name == "resnet50":
        return build_resnet50(
            num_classes,
            pretrained=model_cfg.get("pretrained", True),
            freeze_backbone=model_cfg.get("freeze_backbone", False),
        )
    if name == "vit":
        return build_vit(
            model_cfg["hf_name"],
            num_classes,
            freeze_backbone=model_cfg.get("freeze_backbone", False),
            mean=model_cfg.get("mean"),
            std=model_cfg.get("std"),
        )
    raise ValueError(f"Unknown model name: {name}")


def count_parameters(model: nn.Module):
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
