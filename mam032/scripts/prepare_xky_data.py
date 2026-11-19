#!/usr/bin/env python3
"""Download and prepare one authentic 2025 XKY household, released in 2026."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recent_data import download_xky, prepare_xky_hourly


def main():
    raw = download_xky(ROOT / "data/raw/xky")
    audit = ROOT / "artifacts/xky_2025/data_audit.json"
    prepare_xky_hourly(raw, ROOT / "data/processed/xky_2025_hourly.csv", audit)
    print(audit.read_text())


if __name__ == "__main__":
    main()
