import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bins import BIN_COLORS, BINS, DEFAULT_THRESHOLD, advice, bin_of, decide  # noqa: E402
from data import CLASSES  # noqa: E402


def test_bins_cover_exactly_the_model_classes():
    assert set(BINS) == set(CLASSES)


def test_every_class_has_a_bin_a_chinese_name_a_colour_and_a_tip():
    for label in CLASSES:
        info = advice(label)
        assert info["known"] is True
        assert info["bin"] in BIN_COLORS
        assert info["zh"] and info["tip"]
        assert re.fullmatch(r"#[0-9A-Fa-f]{6}", info["color"])


def test_the_four_bins_are_all_used():
    assert {bin_of(label) for label in CLASSES} == set(BIN_COLORS)


def test_examples_from_the_project_plan():
    assert bin_of("plastic") == "Recyclable" and advice("plastic")["zh"] == "可回收物"
    assert bin_of("biological") == "Kitchen waste"
    assert bin_of("battery") == "Hazardous waste"
    assert bin_of("trash") == "Other waste"


def test_unknown_label_falls_back_to_other_waste_and_is_marked_unknown():
    info = advice("spaceship")
    assert info["bin"] == "Other waste" and info["known"] is False


def test_advice_returns_a_copy():
    advice("glass")["tip"] = "changed"
    assert advice("glass")["tip"] != "changed"


def test_decide_uses_the_threshold():
    assert decide("glass", 0.95)["confident"] is True
    assert decide("glass", 0.41)["confident"] is False
    assert decide("glass", 0.90)["confident"] is True          # exactly on the threshold is accepted
    assert decide("glass", 0.70, threshold=0.60)["confident"] is True
    assert decide("glass", 0.5)["threshold"] == DEFAULT_THRESHOLD
