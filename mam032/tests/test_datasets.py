"""Keep benchmark identity and file locations explicit and separate."""

from dataclasses import FrozenInstanceError

import pytest

from src.datasets import DATASETS, DEFAULT_DATASET_ID, ROOT, get_dataset


def test_recent_measured_dataset_is_default() -> None:
    assert DEFAULT_DATASET_ID == "xky_2025"
    assert next(iter(DATASETS)) == DEFAULT_DATASET_ID
    recent = get_dataset("xky_2025")
    assert recent.label == "Recent household · XKY 2025"
    assert recent.citation_url == "https://doi.org/10.57745/MA43FS"
    assert "BASE" in recent.source_note
    assert "Wh" in recent.source_note


def test_dataset_artifact_and_demo_paths_never_overlap() -> None:
    recent = get_dataset("xky_2025")
    classic = get_dataset("uci_household")
    assert recent.artifact_dir == ROOT / "artifacts" / "xky_2025"
    assert recent.processed_path == ROOT / "data" / "processed" / "xky_2025_hourly.csv"
    assert recent.demo_path == ROOT / "examples" / "xky_2025_demo_hourly.csv"
    assert len({spec.artifact_dir for spec in DATASETS.values()}) == len(DATASETS)
    assert len({spec.processed_path for spec in DATASETS.values()}) == len(DATASETS)
    assert len({spec.demo_path for spec in DATASETS.values()}) == len(DATASETS)
    assert classic.artifact_dir == ROOT / "artifacts"
    assert classic.processed_path == ROOT / "data" / "processed" / "hourly_energy.csv"
    assert classic.demo_path == ROOT / "examples" / "uci_demo_hourly.csv"


def test_unknown_dataset_fails_instead_of_silent_fallback() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        get_dataset("not_a_known_household")


def test_registered_source_configuration_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        get_dataset("xky_2025").id = "uci_household"
