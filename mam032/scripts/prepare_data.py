#!/usr/bin/env python3
"""Download the official UCI dataset and prepare auditable hourly energy data."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import download_uci_dataset, prepare_uci_hourly


def main() -> None:
    raw_path = download_uci_dataset(ROOT / "data" / "raw")
    hourly = prepare_uci_hourly(
        raw_path,
        ROOT / "data" / "processed" / "hourly_energy.csv",
        ROOT / "artifacts" / "data_audit.json",
    )
    print(f"Raw source: {raw_path}")
    print(f"Hourly rows: {len(hourly):,}")
    print(hourly["quality"].value_counts().to_string())


if __name__ == "__main__":
    main()
