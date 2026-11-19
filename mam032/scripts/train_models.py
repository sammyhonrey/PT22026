#!/usr/bin/env python3
"""Train leakage-free models and save real chronological evaluation results."""

from pathlib import Path
import argparse
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["uci_household", "xky_2025"], default="uci_household")
    args = parser.parse_args()
    from src.datasets import get_dataset
    spec = get_dataset(args.dataset)
    hourly_path = spec.processed_path
    if not hourly_path.exists():
        raise SystemExit(f"Prepared data missing. Run: {spec.prepare_command}")
    hourly = pd.read_csv(hourly_path, parse_dates=["timestamp"]).set_index("timestamp")
    metadata = train_and_evaluate(hourly, spec.artifact_dir, dataset_id=spec.id)
    print(json.dumps(metadata, indent=2))
    print("\nHeld-out test metrics:")
    print(pd.read_csv(spec.artifact_dir / "test_metrics.csv").to_string(index=False))


if __name__ == "__main__":
    main()
