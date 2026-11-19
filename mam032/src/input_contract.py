"""Shared, versioned CSV contract used by the dashboard and validation code."""

from __future__ import annotations

import json
import re
from pathlib import Path


# The rules for uploaded CSV files (column names, row limits, and so on)
# are kept in one JSON file, so the app and the checks use the same rules.
CONTRACT_PATH = Path(__file__).resolve().parents[1] / "config" / "input_contract.json"
INPUT_CONTRACT: dict = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

# Patterns that check the text of a timestamp and of an energy value.
# re.ASCII means only plain digits 0-9 are accepted, not other scripts.
TIMESTAMP_PATTERN = re.compile(INPUT_CONTRACT["timestamp_pattern"], flags=re.ASCII)
ENERGY_PATTERN = re.compile(INPUT_CONTRACT["energy_pattern"], flags=re.ASCII)
