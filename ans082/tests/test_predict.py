"""Checks of the photo checks, the model loading and one full analysis on a tiny random ViT."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("pytorch_grad_cam")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import CLASSES  # noqa: E402
from predict import (MAX_SIDE, ImageProblem, ModelLoadError, load_predictor,  # noqa: E402
                     open_image, prepare_image, read_normalization)
from transformers import ViTConfig, ViTForImageClassification  # noqa: E402


def _photo(size=(120, 90)):
    """A coloured image with some structure (not blank)."""
    array = np.random.default_rng(0).integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(array)


# ---- prepare_image ----

def test_no_image_gives_a_friendly_message():
    with pytest.raises(ImageProblem, match="upload or take a photo"):
        prepare_image(None)


def test_something_that_is_not_an_image_is_rejected():
    with pytest.raises(ImageProblem, match="JPG or PNG"):
        prepare_image("a text file")


def test_tiny_image_is_rejected():
    with pytest.raises(ImageProblem, match="too small"):
        prepare_image(_photo((30, 30)))


@pytest.mark.parametrize("colour", [(255, 255, 255), (0, 0, 0), (120, 40, 200)])
def test_blank_image_is_rejected(colour):
    with pytest.raises(ImageProblem, match="blank"):
        prepare_image(Image.new("RGB", (200, 200), color=colour))


def test_palette_image_is_converted_to_rgb():
    palette = _photo().convert("P")
    assert prepare_image(palette).mode == "RGB"


def test_large_image_is_shrunk():
    result = prepare_image(_photo((3000, 2000)))
    assert max(result.size) == MAX_SIDE


def test_phone_rotation_is_applied(tmp_path):
    path = tmp_path / "sideways.jpg"
    exif = Image.Exif()
    exif[274] = 6                       # "rotate 90 degrees to show correctly"
    _photo((120, 80)).save(path, exif=exif)
    with Image.open(path) as opened:
        assert opened.size == (120, 80)
        assert prepare_image(opened).size == (80, 120)


# ---- open_image (photos arrive as file paths from the app) ----

def test_open_image_reads_a_real_file(tmp_path):
    path = tmp_path / "photo.jpg"
    _photo().save(path)
    assert open_image(str(path)).size == (120, 90)


def test_open_image_passes_none_and_pil_images_through():
    image = _photo()
    assert open_image(None) is None and open_image(image) is image


def test_open_image_rejects_a_text_file_named_jpg(tmp_path):
    path = tmp_path / "fake.jpg"
    path.write_text("this is not a picture")
    with pytest.raises(ImageProblem, match="damaged or not an image"):
        open_image(str(path))


def test_open_image_rejects_an_empty_file_and_a_missing_file(tmp_path):
    empty = tmp_path / "empty.jpg"
    empty.write_bytes(b"")
    for path in (empty, tmp_path / "missing.jpg"):
        with pytest.raises(ImageProblem):
            open_image(str(path))


# ---- model loading ----

@pytest.fixture()
def tiny_model_dir(tmp_path):
    config = ViTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                       intermediate_size=64, image_size=64, patch_size=16,
                       id2label=dict(enumerate(CLASSES)), label2id={n: i for i, n in enumerate(CLASSES)})
    folder = tmp_path / "tiny_export"
    ViTForImageClassification(config).save_pretrained(folder)
    return folder


def test_missing_normalization_file_gives_half_and_half(tmp_path):
    assert read_normalization(tmp_path) == ([0.5] * 3, [0.5] * 3)


def test_failed_download_gives_a_friendly_error(tmp_path, monkeypatch):
    import huggingface_hub

    def no_internet(*args, **kwargs):
        raise RuntimeError("no internet")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_internet)
    with pytest.raises(ModelLoadError, match="download did not work"):
        load_predictor(model_dir=tmp_path / "not_there")


def test_wrong_class_names_are_rejected(tmp_path):
    config = ViTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                       intermediate_size=64, image_size=64, patch_size=16, num_labels=10)
    folder = tmp_path / "wrong_names"
    ViTForImageClassification(config).save_pretrained(folder)
    with pytest.raises(ModelLoadError, match="different class names"):
        load_predictor(model_dir=folder, image_size=64)


# ---- one full analysis ----

def test_analyse_returns_label_top3_heatmap_and_advice(tiny_model_dir):
    predictor = load_predictor(model_dir=tiny_model_dir, threshold=0.0, image_size=64)
    result = predictor.analyse(_photo((90, 70)))
    assert result.label in CLASSES
    assert len(result.top3) == 3
    assert result.top3[0][1] >= result.top3[1][1] >= result.top3[2][1]
    assert result.top3[0][0] == result.label
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-4
    assert result.overlay.shape == (64, 64, 3) and result.overlay.dtype == np.uint8
    assert result.advice["bin"] and result.advice["confident"] is True      # threshold 0 accepts everything


def test_high_threshold_means_not_sure(tiny_model_dir):
    predictor = load_predictor(model_dir=tiny_model_dir, threshold=1.01, image_size=64)
    assert predictor.analyse(_photo()).advice["confident"] is False


def test_analyse_accepts_a_file_path(tiny_model_dir, tmp_path):
    path = tmp_path / "photo.jpg"
    _photo((90, 70)).save(path)
    predictor = load_predictor(model_dir=tiny_model_dir, image_size=64)
    assert predictor.analyse(str(path)).label in CLASSES
    predictor.close()


def test_analyse_rejects_a_blank_image_before_the_model(tiny_model_dir):
    predictor = load_predictor(model_dir=tiny_model_dir, image_size=64)
    with pytest.raises(ImageProblem):
        predictor.analyse(Image.new("RGB", (200, 200), color="white"))