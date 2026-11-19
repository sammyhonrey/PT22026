"""Explicit dataset identities and isolated local paths for every benchmark.

This registry describes the selected measured series, never a live data feed.
Keeping paths and source labels together prevents mixing one household's model
with another household's evaluation metrics or demonstration input.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetSpec:
    """All fixed facts about one dataset: names, file paths, text and licence.

    frozen=True means the values cannot be changed after they are created.
    """

    id: str
    label: str
    short_name: str
    artifact_dir: Path  # saved model and scores
    processed_path: Path  # hourly data made by the prepare script
    demo_path: Path  # small example file for the offline demo
    source_note: str
    citation_url: str
    prepare_command: str
    train_command: str
    timezone_note: str
    target_description: str
    limitations: str
    license: str

    @property
    def input_clock_policy(self) -> str:
        """Match uploaded calendar features to the benchmark's documented clock.

        XKY data uses UTC. UCI keeps the clock times exactly as written.
        """
        return "utc" if self.id == "xky_2025" else "preserve"


# The two datasets the app knows. The key is the dataset id used everywhere.
# Note: the UCI files sit directly in artifacts/, XKY files in artifacts/xky_2025/.
DATASETS: dict[str, DatasetSpec] = {
    "xky_2025": DatasetSpec(
        id="xky_2025",
        label="Recent household · XKY 2025",
        short_name="XKY 2025",
        artifact_dir=ROOT / "artifacts" / "xky_2025",
        processed_path=ROOT / "data" / "processed" / "xky_2025_hourly.csv",
        demo_path=ROOT / "examples" / "xky_2025_demo_hourly.csv",
        source_note=(
            "XKY Study, Subject 14: real French household smart-meter records from 2025. "
            "Hourly grid-drawn energy is estimated from the BASE cumulative energy index in Wh, "
            "using past-only checkpoints within 180 seconds of each hour boundary. Released in 2026."
        ),
        citation_url="https://doi.org/10.57745/MA43FS",
        prepare_command="python scripts/prepare_xky_data.py",
        train_command="python scripts/train_models.py --dataset xky_2025",
        timezone_note="Timestamps use UTC. Retained hours are boundary estimates; missing boundary coverage is excluded.",
        target_description="hourly grid-drawn electricity energy (kWh)",
        limitations=(
            "The recent benchmark represents one measured French household in 2025. "
            "Grid-drawn energy can differ from total household demand where onsite generation is present. "
            "Accuracy on other households and later periods has not been established."
        ),
        license="Etalab Open Licence 2.0",
    ),
    "uci_household": DatasetSpec(
        id="uci_household",
        label="Classic benchmark · UCI 2006–2010",
        short_name="UCI 2006–2010",
        artifact_dir=ROOT / "artifacts",
        processed_path=ROOT / "data" / "processed" / "hourly_energy.csv",
        demo_path=ROOT / "examples" / "uci_demo_hourly.csv",
        source_note="Official UCI dataset: one household in Sceaux, France, recorded during 2006–2010.",
        citation_url="https://doi.org/10.24432/C58K54",
        prepare_command="python scripts/prepare_data.py",
        train_command="python scripts/train_models.py",
        timezone_note="Source local-clock timestamps are preserved; timezone and daylight-saving certainty is not invented.",
        target_description="hourly household electricity energy (kWh)",
        limitations=(
            "The UCI benchmark represents one French household from 2006–2010. "
            "It is not nationally representative and does not measure current consumption."
        ),
        license="CC BY 4.0",
    ),
}

# The dataset the app opens with.
DEFAULT_DATASET_ID = "xky_2025"


def get_dataset(dataset_id: str) -> DatasetSpec:
    """Resolve a known benchmark or fail explicitly instead of falling back."""
    try:
        return DATASETS[dataset_id]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset '{dataset_id}'. Choose one of: {', '.join(DATASETS)}") from exc
