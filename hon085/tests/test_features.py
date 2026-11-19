import numpy as np
import pytest

from xray import features
from xray import landmarks as lm


def rotate_yaw(points, degrees):
    a = np.radians(degrees)
    out = points.copy()
    out[..., 0] = np.cos(a) * points[..., 0] + np.sin(a) * points[..., 2]
    out[..., 2] = -np.sin(a) * points[..., 0] + np.cos(a) * points[..., 2]
    return out


def test_feature_vector_size(poses):
    x = features.extract(poses["standing"])
    assert x.shape == (1, len(features.feature_names()))
    assert x.shape[1] == len(lm.BODY) * 3 + len(lm.ANGLES) + 1 + len(lm.DISTANCES)


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        features.extract(np.zeros((10, 3)))


def test_straight_limbs_measure_180_degrees(poses):
    angles = dict(zip(lm.ANGLES, features.joint_angles(poses["standing"])[0]))
    assert angles["left_knee"] == pytest.approx(180, abs=1)
    assert angles["right_knee"] == pytest.approx(180, abs=1)


def test_trunk_is_upright_when_standing(poses):
    assert features.trunk_lean(poses["standing"])[0] < 2


def test_features_ignore_body_size_and_position(poses):
    base = features.extract(poses["arms_up"])
    moved = features.extract(poses["arms_up"] * 1.3 + np.array([0.4, -0.2, 1.5]))
    assert np.allclose(base, moved, atol=1e-6)


@pytest.mark.parametrize("degrees", [30, 90, 170])
def test_features_ignore_viewing_angle(poses, degrees):
    base = features.extract(poses["one_leg"])
    turned = features.extract(rotate_yaw(poses["one_leg"], degrees))
    assert np.allclose(base, turned, atol=1e-6)


def test_mirror_swaps_sides_and_is_reversible(poses):
    flipped = features.mirror(poses["one_leg"])[0]
    angles = dict(zip(lm.ANGLES, features.joint_angles(flipped)[0]))
    assert angles["right_knee"] < 120 < angles["left_knee"]
    assert np.allclose(features.mirror(flipped)[0], poses["one_leg"])
