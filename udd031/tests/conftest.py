"""Test configuration: make ``src`` importable and provide shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def ontology(project_root: Path):
    from mmsa.ontology import Ontology

    return Ontology.from_yaml(project_root / "config" / "ontology.yaml")


@pytest.fixture(scope="session")
def app_config(project_root: Path):
    from mmsa.config import load_config

    return load_config(project_root / "config", device="cpu")


@pytest.fixture
def sine_wave():
    """A 2 s 220 Hz tone with light noise at 16 kHz."""

    def _make(duration_s: float = 2.0, freq: float = 220.0, sr: int = 16_000) -> np.ndarray:
        t = np.linspace(0.0, duration_s, int(duration_s * sr), endpoint=False)
        rng = np.random.default_rng(1337)
        wave = 0.5 * np.sin(2 * np.pi * freq * t) + 0.01 * rng.standard_normal(t.size)
        return wave.astype(np.float32)

    return _make
