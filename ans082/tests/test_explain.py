"""Checks of the Grad-CAM code on a very small random ViT (no download needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("pytorch_grad_cam")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import CLASSES  # noqa: E402
from explain import (ViTExplainer, find_target_layer, find_target_layer_name,  # noqa: E402
                     vit_reshape_transform)
from model_loader import build_model  # noqa: E402
from transformers import ViTConfig, ViTForImageClassification  # noqa: E402


@pytest.fixture()
def tiny_model(tmp_path):
    config = ViTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                       intermediate_size=64, image_size=64, patch_size=16)
    folder = tmp_path / "tiny_vit"
    ViTForImageClassification(config).save_pretrained(folder)
    cfg = {"name": "vit", "hf_name": str(folder), "freeze_backbone": False,
           "mean": [0.5] * 3, "std": [0.5] * 3}
    model, _, _ = build_model(cfg)
    return model


def test_reshape_transform_drops_cls_and_makes_a_grid():
    tokens = torch.arange(2 * 17 * 8, dtype=torch.float32).reshape(2, 17, 8)   # 1 CLS + 4 x 4 patches
    grid = vit_reshape_transform(tokens, grid=4)
    assert grid.shape == (2, 8, 4, 4)
    assert torch.equal(grid[0, :, 0, 0], tokens[0, 1, :])      # first patch is token 1, not the CLS token
    assert torch.equal(grid[0, :, 0, 1], tokens[0, 2, :])


def test_target_layer_is_the_layer_norm_of_the_last_block(tiny_model):
    name = find_target_layer_name(tiny_model)
    assert name.endswith(".1.layernorm_before")                 # the tiny model has 2 blocks: 0 and 1
    layer = find_target_layer(tiny_model)
    assert isinstance(layer, torch.nn.LayerNorm)
    assert layer is dict(tiny_model.named_modules())[name]


def test_explain_returns_overlay_heat_and_probabilities(tiny_model):
    explainer = ViTExplainer(tiny_model, [0.5] * 3, [0.5] * 3, image_size=64)
    image = Image.new("RGB", (90, 70), color=(200, 120, 40))
    overlay, heat, probs, index = explainer.explain(image)
    assert overlay.shape == (64, 64, 3) and overlay.dtype == np.uint8
    assert heat.shape == (64, 64)
    assert 0.0 <= float(heat.min()) and float(heat.max()) <= 1.0
    assert probs.shape == (len(CLASSES),) and abs(float(probs.sum()) - 1.0) < 1e-4
    assert index == int(probs.argmax())
    explainer.close()


def test_explain_can_target_another_class(tiny_model):
    explainer = ViTExplainer(tiny_model, [0.5] * 3, [0.5] * 3, image_size=64)
    image = Image.new("RGB", (64, 64), color=(10, 200, 90))
    _, _, _, index = explainer.explain(image, class_index=3)
    assert index == 3
    explainer.close()


def test_explain_works_for_a_palette_image(tiny_model):
    explainer = ViTExplainer(tiny_model, [0.5] * 3, [0.5] * 3, image_size=64)
    overlay, _, _, _ = explainer.explain(Image.new("P", (50, 80)))
    assert overlay.shape == (64, 64, 3)
    explainer.close()