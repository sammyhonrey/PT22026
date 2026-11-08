"""Model definitions: ResNet-50 (CNN) and ViT-base (Transformer)."""
import torch.nn as nn
from torchvision import models
from transformers import ViTForImageClassification


class ViTClassifier(nn.Module):
    """Wrap the Hugging Face ViT so forward() returns plain logits."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        return self.backbone(pixel_values=x).logits


def build_model(name, num_classes=2):
    """Return a pre-trained model with a new classification head."""
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "vit":
        return ViTClassifier(num_classes)
    raise ValueError(f"Unknown model '{name}'. Use 'resnet50' or 'vit'.")