"""Three models of increasing capacity, all with a 43-way output head.

smallcnn       - our own 4-block CNN trained from scratch (the baseline architecture we can fully explain)
resnet18       - ImageNet-pretrained residual network (He et al., 2016), fine-tuned
convnext_tiny  - ImageNet-pretrained modern ConvNet (Liu et al., 2022), fine-tuned
"""
import torch
import torch.nn as nn
from torchvision import models as tvm

from src.classes import NUM_CLASSES
from src.config import MODEL_NAMES


class SmallCNN(nn.Module):
    """conv-BN-ReLU x2 + max-pool, repeated 4 times, then global pooling and a linear classifier."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(self.drop(x))


def build_model(name: str, pretrained: bool = True, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Create a model by name. `pretrained=False` skips the ImageNet weight download (tests, inference)."""
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model '{name}', choose from {MODEL_NAMES}")
    if name == "smallcnn":
        return SmallCNN(num_classes)
    if name == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    return m


def gradcam_layer(model: nn.Module, name: str) -> nn.Module:
    """Target layer for Grad-CAM. With 64x64 inputs the deepest stages are only 2x2, which gives an
    uselessly coarse heatmap, so we pick the last stage that still has a 4x4 or 8x8 feature map."""
    if name == "smallcnn":
        return model.features[-1][3]          # last conv, 8x8 feature map
    if name == "resnet18":
        return model.layer3[-1]               # 4x4 feature map
    return model.features[5]                  # convnext_tiny stage 3, 4x4 feature map
