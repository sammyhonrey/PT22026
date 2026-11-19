import numpy as np
import pytest

from xray import config, features, training
from xray.classifier import ModelNotTrained, PoseClassifier, canonical_side


@pytest.mark.parametrize("algorithm", sorted(training.ALGORITHMS))
def test_every_algorithm_learns_the_toy_poses(toy_data, algorithm):
    points, labels = toy_data
    model = PoseClassifier(algorithm).fit(points, labels)
    guess = np.array(model.classes)[model.predict_proba(points).argmax(axis=1)]
    assert (guess == labels).mean() > 0.95


def test_predict_reports_label_and_probabilities(toy_data, poses):
    model = PoseClassifier().fit(*toy_data)
    result = model.predict(poses["arms_up"])
    assert result["label"] == "arms_up" and result["known"]
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)


def test_low_confidence_is_unknown(toy_data, poses):
    model = PoseClassifier().fit(*toy_data)
    result = model.predict(poses["arms_up"], min_confidence=1.1)
    assert result["label"] == config.UNKNOWN and result["closest"] == "arms_up"


def test_mirrored_pose_is_recognised(toy_data, poses):
    model = PoseClassifier().fit(*toy_data)
    assert model.predict(features.mirror(poses["one_leg"]))["label"] == "one_leg"


def test_needs_two_classes(toy_data):
    points, labels = toy_data
    with pytest.raises(ValueError):
        PoseClassifier().fit(points[:40], labels[:40])


def test_save_and_load(toy_data, poses, tmp_path):
    path = PoseClassifier().fit(*toy_data).save(tmp_path / "model.joblib")
    loaded = PoseClassifier.load(path)
    assert loaded.predict(poses["standing"])["label"] == "standing"
    assert set(loaded.angle_reference) == set(loaded.classes)


def test_load_without_file(tmp_path):
    with pytest.raises(ModelNotTrained):
        PoseClassifier.load(tmp_path / "missing.joblib")


def test_canonical_side_puts_bent_knee_on_the_left(poses):
    _, flipped = canonical_side(poses["one_leg"])
    assert not flipped[0]
    _, flipped = canonical_side(features.mirror(poses["one_leg"]))
    assert flipped[0]


def test_cross_validation_uses_subjects(toy_data):
    points, labels = toy_data
    groups = np.array([f"s{i % 4}" for i in range(len(labels))])
    result = training.cross_validate(points, labels, groups)
    assert result["method"] == "leave one subject out" and len(result["folds"]) == 4
    assert result["accuracy"] > 0.95
