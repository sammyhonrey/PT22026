import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import build_transforms, load_rgb  # noqa: E402

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


def _dummy_image(size=(400, 300)):
    return Image.new("RGB", size, color=(200, 120, 40))


def test_output_shape_and_type():
    train_tf, eval_tf = build_transforms(MEAN, STD)
    for tf in (train_tf, eval_tf):
        out = tf(_dummy_image())
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 224, 224)
        assert out.dtype == torch.float32


def test_eval_transform_is_deterministic():
    _, eval_tf = build_transforms(MEAN, STD)
    img = _dummy_image()
    assert torch.equal(eval_tf(img), eval_tf(img))


def test_train_transform_is_random():
    train_tf, _ = build_transforms(MEAN, STD)
    img = Image.effect_noise((300, 300), 60).convert("RGB")   # non-uniform, so crops differ
    torch.manual_seed(0)
    assert not torch.equal(train_tf(img), train_tf(img))


def test_normalised_range():
    _, eval_tf = build_transforms(MEAN, STD)
    out = eval_tf(_dummy_image())
    assert out.min() >= -1.0 - 1e-5 and out.max() <= 1.0 + 1e-5


def test_palette_image_converts_to_rgb(tmp_path):
    path = tmp_path / "palette.png"
    Image.new("P", (50, 50)).save(path)
    assert load_rgb(path).mode == "RGB"


def test_strong_augmentation_shape_and_randomness():
    train_tf, eval_tf = build_transforms(MEAN, STD, augmentation="strong")
    img = Image.effect_noise((300, 300), 60).convert("RGB")
    torch.manual_seed(0)
    first, second = train_tf(img), train_tf(img)
    assert first.shape == (3, 224, 224)
    assert not torch.equal(first, second)
    assert eval_tf(img).shape == (3, 224, 224)


def test_unknown_augmentation_name_is_rejected():
    with pytest.raises(ValueError):
        build_transforms(MEAN, STD, augmentation="extreme")
