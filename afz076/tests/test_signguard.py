"""Unit and integration tests. Everything runs on CPU with synthetic data (no download needed)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.calibration import (expected_calibration_error, fit_temperature, nll, selective_accuracy,
                             softmax_np)
from src.classes import CLASS_NAMES, NUM_CLASSES
from src.config import CONDITIONS, SEVERITIES
from src.corruptions import corrupt, random_weather
from src.data import SignDataset, find_root, group_split, load_arrays, load_index, to_tensor
from src.gradcam import grad_cam, overlay_heatmap
from src.models import build_model, gradcam_layer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def img():
    return np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8)


# ---------------------------------------------------------------- corruptions
@pytest.mark.parametrize("cond", CONDITIONS)
@pytest.mark.parametrize("sev", SEVERITIES)
def test_corruption_shape_dtype(img, cond, sev):
    out = corrupt(img, cond, sev, np.random.default_rng(0))
    assert out.shape == img.shape and out.dtype == np.uint8


@pytest.mark.parametrize("cond", CONDITIONS)
def test_corruption_is_reproducible_and_changes_image(img, cond):
    a = corrupt(img, cond, 3, np.random.default_rng(5))
    b = corrupt(img, cond, 3, np.random.default_rng(5))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, img)


@pytest.mark.parametrize("cond", ["fog", "night", "noise", "blur", "glare"])
def test_higher_severity_is_stronger(img, cond):
    diffs = [np.abs(corrupt(img, cond, s, np.random.default_rng(0)).astype(float) - img).mean() for s in (1, 3, 5)]
    assert diffs[0] < diffs[1] < diffs[2]


def test_corruption_input_validation(img):
    with pytest.raises(ValueError):
        corrupt(img, "snow", 1)
    with pytest.raises(ValueError):
        corrupt(img, "fog", 9)
    with pytest.raises(ValueError):
        corrupt(img.astype(np.float32), "fog", 1)


def test_random_weather_probability(img):
    rng = np.random.default_rng(0)
    assert np.array_equal(random_weather(img, rng, p=0.0), img)
    assert not np.array_equal(random_weather(img, rng, p=1.0), img)


def test_training_augmentation_never_uses_glare_or_severity_5(img, monkeypatch):
    seen = []
    import src.corruptions as c
    original = c.corrupt
    monkeypatch.setattr(c, "corrupt", lambda im, cond, sev, rng=None: (seen.append((cond, sev)), original(im, cond, sev, rng))[1])
    rng = np.random.default_rng(0)
    for _ in range(300):
        c.random_weather(img, rng, p=1.0)
    assert seen and all(cond != "glare" and sev <= 4 for cond, sev in seen)


# ---------------------------------------------------------------- data
def test_class_names():
    assert NUM_CLASSES == 43 and len(set(CLASS_NAMES)) == 43


def test_find_root_and_index(synthetic_root):
    root = find_root(str(synthetic_root))
    assert (root / "Train").is_dir()
    train, test = load_index(root, "train"), load_index(root, "test")
    assert len(train) == 43 * 3 * 5 and len(test) == 86
    assert set(train["label"]) == set(range(43))


def test_find_root_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_root(str(tmp_path))


def test_group_split_has_no_leakage(synthetic_root):
    df = load_index(find_root(str(synthetic_root)), "train")
    tr, va = group_split(df, val_frac=0.34)
    assert set(tr["group"]).isdisjoint(set(va["group"]))          # same physical sign never in both
    assert set(va["label"]) == set(range(43))                      # every class is validated
    assert set(tr["label"]) == set(range(43))
    assert len(tr) + len(va) == len(df)


def test_load_arrays_and_cache(synthetic_root, tmp_path):
    df = load_index(find_root(str(synthetic_root)), "test")
    x, y = load_arrays(df, cache=str(tmp_path / "c.npz"))
    assert x.shape == (86, 64, 64, 3) and x.dtype == np.uint8 and y.shape == (86,)
    x2, y2 = load_arrays(df, cache=str(tmp_path / "c.npz"))        # second call reads the cache
    assert np.array_equal(x, x2) and np.array_equal(y, y2)


def test_dataset_items(synthetic_root):
    df = load_index(find_root(str(synthetic_root)), "test")
    x, y = load_arrays(df)
    ds = SignDataset(x, y, fixed_condition=("fog", 3))
    a, b = ds[0], ds[0]
    assert a[0].shape == (3, 64, 64) and torch.equal(a[0], b[0]) and a[1] == int(y[0])
    clean = SignDataset(x, y)[0][0]
    assert not torch.equal(clean, a[0])
    train_item = SignDataset(x, y, train=True, weather_p=1.0)[0][0]
    assert train_item.shape == (3, 64, 64)


def test_to_tensor_range():
    t = to_tensor(np.full((64, 64, 3), 255, dtype=np.uint8))
    assert t.shape == (3, 64, 64) and t.max() < 4


# ---------------------------------------------------------------- calibration
def test_ece_perfectly_calibrated_vs_overconfident():
    rng = np.random.default_rng(0)
    n = 20000
    labels = np.zeros(n, dtype=int)
    probs = np.full((n, 2), 0.2); probs[:, 0] = 0.8                 # always 80% confident in class 0
    correct = rng.random(n) < 0.8                                   # ...and right 80% of the time
    labels[~correct] = 1
    assert expected_calibration_error(probs, labels) < 0.02
    labels_bad = (rng.random(n) < 0.5).astype(int)                  # right only ~50% of the time
    over = np.full((n, 2), 0.01); over[:, 0] = 0.99
    assert expected_calibration_error(over, labels_bad) > 0.4


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    n, k = 20000, 5
    logits = rng.normal(0, 2, (n, k))
    p = softmax_np(logits)
    labels = np.array([rng.choice(k, p=row) for row in p])          # labels drawn from the model's own
    #                                                                 probabilities -> perfectly calibrated
    assert 0.9 < fit_temperature(logits, labels) < 1.1              # calibrated model -> T close to 1
    overconfident = logits * 4                                      # same ranking, but 4x too sure
    T = fit_temperature(overconfident, labels)
    assert 3.4 < T < 4.6                                            # recovers the factor we inflated by
    assert nll(overconfident, labels, T) < nll(overconfident, labels, 1.0)
    assert expected_calibration_error(softmax_np(overconfident, T), labels) < \
        expected_calibration_error(softmax_np(overconfident), labels)


def test_selective_accuracy_tradeoff():
    rng = np.random.default_rng(0)
    n = 5000
    labels = rng.integers(0, 3, n)
    conf = rng.random(n) * 0.5 + 0.5
    correct = rng.random(n) < conf                                  # confidence is informative
    pred = np.where(correct, labels, (labels + 1) % 3)
    probs = np.full((n, 3), 0.0)
    probs[np.arange(n), pred] = conf
    probs += (1 - conf)[:, None] / 3
    rows = selective_accuracy(probs, labels, thresholds=(0.5, 0.8, 0.95))
    assert rows[0]["coverage"] > rows[1]["coverage"] > rows[2]["coverage"]
    assert rows[0]["accuracy_on_answered"] < rows[2]["accuracy_on_answered"]


def test_softmax_rows_sum_to_one():
    p = softmax_np(np.random.default_rng(0).normal(size=(4, 7)) * 50, temperature=2.0)
    assert np.allclose(p.sum(1), 1.0) and np.isfinite(p).all()


# ---------------------------------------------------------------- models / grad-cam
@pytest.mark.parametrize("name", ["smallcnn", "resnet18", "convnext_tiny"])
def test_model_forward_and_gradcam(name):
    model = build_model(name, pretrained=False).eval()
    x = torch.randn(1, 3, 64, 64)
    assert model(x).shape == (1, 43)
    cam = grad_cam(model, gradcam_layer(model, name), x)
    assert cam.shape == (64, 64) and 0.0 <= cam.min() and cam.max() <= 1.0


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        build_model("vgg")


def test_overlay_heatmap_shape():
    out = overlay_heatmap(np.zeros((100, 120, 3), dtype=np.uint8), np.random.default_rng(0).random((64, 64)))
    assert out.shape == (100, 120, 3) and out.dtype == np.uint8


# ---------------------------------------------------------------- end to end
def test_full_pipeline_smoke(synthetic_root, tmp_path):
    """train -> evaluate (with temperature scaling) -> plots -> single-image prediction."""
    py = sys.executable
    run = lambda *a: subprocess.run([py, *map(str, a)], cwd=tmp_path, capture_output=True, text=True, timeout=900)

    r = run(ROOT / "training/train.py", "--data-root", synthetic_root, "--model", "smallcnn", "--no-pretrained",
            "--epochs", 6, "--batch-size", 32, "--weather-p", 0.5, "--name", "smoke", "--cache-dir", tmp_path / "cache")
    assert r.returncode == 0, r.stderr[-2000:]
    history = json.loads((tmp_path / "outputs/smoke/history.json").read_text())
    assert history["history"][-1]["train_loss"] < history["history"][0]["train_loss"]
    assert history["best_val_acc"] > 0.2                            # chance is ~2%

    r = run(ROOT / "evaluation/evaluate.py", "--data-root", synthetic_root, "--ckpt", "outputs/smoke/model.pt",
            "--severities", 1, 3, "--cache-dir", tmp_path / "cache")
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads((tmp_path / "evaluation/results/smoke.json").read_text())
    assert res["clean"]["accuracy"] > 0.2 and res["temperature"] > 0
    assert len(res["conditions"]) == 6 * 2 and "selective_sev3_pooled" in res
    assert (tmp_path / "evaluation/results/robustness.csv").exists()

    r = run(ROOT / "evaluation/make_plots.py")
    assert r.returncode == 0, r.stderr[-2000:]
    for fig in ("robustness", "reliability", "coverage", "training_curves"):
        assert (tmp_path / f"evaluation/figures/{fig}.png").stat().st_size > 1000

    from PIL import Image
    from src.predict import SignPredictor
    predictor = SignPredictor(str(tmp_path / "outputs/smoke/model.pt"))
    assert predictor.temperature == pytest.approx(res["temperature"])    # evaluation stored it in the checkpoint
    sample = Image.open(next((find_root(str(synthetic_root)) / "Test").glob("*.png")))
    pred = predictor.predict(SignPredictor.prepare(sample), threshold=0.0, with_heatmap=True)
    assert len(pred.top) == 3 and pred.accepted and pred.heatmap.shape == (64, 64, 3)
    assert predictor.predict(SignPredictor.prepare(sample), threshold=1.01).accepted is False
