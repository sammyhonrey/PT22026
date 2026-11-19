"""Checks of the evaluation helpers and of one full evaluation run on a tiny synthetic dataset."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import CLASSES  # noqa: E402
from evaluate import (compute_metrics, list_real_photos, make_predictions_table,  # noqa: E402
                      run_evaluation, threshold_table, top_confident_errors)
from model_loader import build_model  # noqa: E402

TEST_CLASSES = ["battery", "glass", "paper"]


def test_metrics_are_perfect_for_perfect_predictions():
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    metrics = compute_metrics(labels, labels)
    assert metrics["accuracy"] == 1.0 and metrics["macro_f1"] == 1.0
    assert set(metrics["per_class"]) == set(CLASSES)


def test_metrics_with_one_mistake():
    labels = [0, 0, 1, 1]
    preds = [0, 1, 1, 1]
    metrics = compute_metrics(labels, preds)
    assert metrics["accuracy"] == 0.75
    assert metrics["per_class"]["battery"]["recall"] == 0.5


def test_predictions_table_and_top_errors():
    probs = np.zeros((4, len(CLASSES)))
    probs[0, 0], probs[1, 1], probs[2, 2], probs[3, 3] = 0.9, 0.8, 0.7, 0.6
    table = make_predictions_table(["a", "b", "c", "d"], [0, 1, 0, 0], probs)
    assert list(table["correct"]) == [True, True, False, False]
    errors = top_confident_errors(table, k=1)
    assert len(errors) == 1 and errors.loc[0, "path"] == "c"   # confidence 0.7 beats 0.6


def test_threshold_table_coverage_and_accuracy():
    confidence = [0.95, 0.8, 0.55, 0.4]
    correct = [True, True, False, False]
    table = threshold_table(confidence, correct, thresholds=(0.0, 0.6, 0.99))
    assert list(table["coverage"]) == [1.0, 0.5, 0.0]
    assert table.loc[1, "accuracy_on_accepted"] == 1.0
    assert pd.isna(table.loc[2, "accuracy_on_accepted"])


def test_list_real_photos_reads_class_folders(tmp_path):
    for name in ("glass", "paper"):
        (tmp_path / name).mkdir()
        Image.new("RGB", (20, 20)).save(tmp_path / name / f"{name}_001.jpg")
    (tmp_path / "glass" / "notes.txt").write_text("not an image")
    paths, labels = list_real_photos(tmp_path)
    assert len(paths) == 2
    assert labels == [CLASSES.index("glass"), CLASSES.index("paper")]


def test_list_real_photos_rejects_unknown_class_folder(tmp_path):
    (tmp_path / "spaceship").mkdir()
    with pytest.raises(ValueError):
        list_real_photos(tmp_path)


@pytest.fixture()
def tiny_setup(tmp_path):
    """A random ResNet-50 checkpoint plus 24 small images (4 val images per class)."""
    root = tmp_path / "images"
    rows = []
    for class_index, name in enumerate(TEST_CLASSES):
        (root / name).mkdir(parents=True)
        for i in range(8):
            rel = f"{name}/{name}{i}.jpg"
            Image.new("RGB", (90, 70), color=(40 * class_index + 10, 90, 160)).save(root / rel)
            split = "train" if i < 4 else "val" if i < 6 else "test"
            rows.append({"path": rel, "label": name, "split": split, "orig_class": name})
    csv_path = tmp_path / "split.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    model_cfg = {"name": "resnet50", "pretrained": False, "freeze_backbone": False}
    model, mean, std = build_model(model_cfg)
    ckpt_path = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "classes": CLASSES, "mean": mean, "std": std,
                "model_cfg": model_cfg, "epoch": 3, "val_macro_f1": 0.0, "val_acc": 0.0}, ckpt_path)
    return ckpt_path, csv_path, root


def test_run_evaluation_writes_every_output_file(tiny_setup, tmp_path):
    ckpt_path, csv_path, root = tiny_setup
    out_dir = tmp_path / "eval"
    metrics = run_evaluation(ckpt_path, "val", csv_path, root, None, out_dir, "unit",
                             image_size=64, batch_size=4, workers=0)
    assert metrics["n_images"] == 6 and metrics["checkpoint_epoch"] == 3
    for suffix in ("predictions.csv", "metrics.json", "confusion.png",
                   "thresholds.csv", "top_errors.csv"):
        assert (out_dir / f"unit_val_{suffix}").exists()
    saved = json.loads((out_dir / "unit_val_metrics.json").read_text())
    assert saved["split"] == "val"
    predictions = pd.read_csv(out_dir / "unit_val_predictions.csv")
    assert len(predictions) == 6 and "p_battery" in predictions.columns


def test_run_evaluation_on_real_photo_folder(tiny_setup, tmp_path):
    ckpt_path, csv_path, root = tiny_setup
    metrics = run_evaluation(ckpt_path, "real", csv_path, root, root, tmp_path / "eval", "unit",
                             image_size=64, batch_size=4, workers=0)
    assert metrics["n_images"] == 24
