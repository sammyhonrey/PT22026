import numpy as np
import pytest

from xray import config, muscles
from xray import landmarks as lm
from xray.classifier import PoseClassifier
from xray.coach import Coach

VISIBLE = np.ones(lm.COUNT)


@pytest.fixture
def coach(toy_data):
    return Coach(PoseClassifier().fit(*toy_data))


def hold(coach, pose, start, seconds, step=0.1):
    reply = None
    for i in range(int(seconds / step)):
        reply = coach.update(pose, VISIBLE, start + i * step)
    return reply


def test_label_and_hold_time(coach, poses):
    reply = hold(coach, poses["arms_up"], 0.0, 2.0)
    assert reply["label"] == "arms_up" and reply["known"]
    assert reply["hold_seconds"] == pytest.approx(1.9, abs=0.2)


def test_single_odd_frame_does_not_change_the_label(coach, poses):
    hold(coach, poses["arms_up"], 0.0, 1.0)
    reply = coach.update(poses["standing"], VISIBLE, 1.0)
    assert reply["label"] == "arms_up"


def test_label_changes_after_a_stable_new_pose(coach, poses):
    hold(coach, poses["arms_up"], 0.0, 1.0)
    reply = hold(coach, poses["standing"], 1.0, 2.0)
    assert reply["label"] == "standing"


def test_hidden_body_gives_no_label(coach, poses):
    reply = coach.update(poses["standing"], np.full(lm.COUNT, 0.1), 0.0)
    assert reply["status"] == "low_visibility" and reply["label"] == config.UNKNOWN


def test_repetitions_are_counted(coach, poses):
    t = 0.0
    for _ in range(3):
        hold(coach, poses["standing"], t, 1.5)
        hold(coach, poses["arms_up"], t + 1.5, 1.5)
        t += 3.0
    summary = {p["label"]: p for p in coach.summary()["poses"]}
    assert summary["arms_up"]["times_entered"] == 3
    assert summary["arms_up"]["seconds"] > 2.5


def test_long_pause_starts_a_new_hold(coach, poses):
    hold(coach, poses["arms_up"], 0.0, 1.0)
    reply = coach.update(poses["arms_up"], VISIBLE, 60.0)
    assert reply["hold_seconds"] < 1.0


def test_known_pose_reports_muscles():
    assert muscles.activation("squat")["quadriceps"] == muscles.PRIMARY
    assert muscles.activation("no_such_pose") == {}
    assert muscles.group_of("right vastus lateralis") == "quadriceps"
    assert muscles.group_of("oblique head of right adductor pollicis") is None


def test_form_rules_flag_a_shallow_chair_pose(poses):
    from xray import form

    score, hints = form.assess("chair", poses["standing"])
    assert score < 60
    assert "Bend your knees more and sit back" in hints


def test_form_of_a_taught_pose_uses_its_recorded_angles(toy_data, poses):
    from xray import form

    model = PoseClassifier().fit(*toy_data)
    good, hints = form.assess("arms_up", poses["arms_up"], model.angle_reference["arms_up"])
    poor, _ = form.assess("arms_up", poses["standing"], model.angle_reference["arms_up"])
    assert good > 90 and not hints
    assert poor < good
