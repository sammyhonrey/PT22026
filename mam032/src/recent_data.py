"""Verified XKY 2025 meter acquisition and past-only hourly energy estimates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .data import DataValidationError

# Where the official XKY file comes from. XKY_MD5 is the checksum (file
# fingerprint) published with the data. We use it to check the file is exact.
XKY_FILE_URL ="https://entrepot.recherche.data.gouv.fr/api/access/datafile/768872?format=original"
XKY_SOURCE_URL = "https://doi.org/10.57745/MA43FS"
XKY_FILE_DOI = "https://doi.org/10.57745/OILZFV"
XKY_MD5 = "044353817cafd1bbac3e338d2e0e5732"
XKY_FILENAME = "Subject14_DataLinkyRAW_2025.csv"


def download_xky(destination: Path) -> Path:
    """Fetch one published household file and verify the depositor's checksum."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / XKY_FILENAME
    # If a correct copy is already there, skip the download.
    if target.exists() and hashlib.md5(target.read_bytes()).hexdigest() == XKY_MD5:
        return target
    # Download to a temporary file first. Rename it only if the checksum is right,
    # so a broken download never replaces a good file.
    temporary = target.with_suffix(".download")
    try:
        with requests.get(XKY_FILE_URL, stream=True, timeout=120) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        output.write(chunk)
        if hashlib.md5(temporary.read_bytes()).hexdigest() != XKY_MD5:
            raise DataValidationError("XKY download checksum differs from the published file.")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def cumulative_wh_to_hourly(frame: pd.DataFrame, max_age_seconds: int = 180) -> pd.DataFrame:
    """Difference past-only cumulative Wh checkpoints close to hour boundaries.

    For an interval [t,t+1h), each checkpoint is the latest meter reading at
    or before its boundary. Both must be within 180 seconds. Differences /1000
    are approximate hourly kWh, flagged estimated because endpoints may be
    slightly early. No interpolation, forward lookup or zero filling is used.
    Any counter reset in a selected checkpoint interval invalidates that hour.
    """
    if not {"timestamp", "cumulative_wh"}.issubset(frame.columns):
        raise DataValidationError("Meter data requires timestamp and cumulative_wh columns.")
    if not 0 <= max_age_seconds <= 300:
        raise ValueError("Checkpoint age must be between zero and 300 seconds.")
    data = frame[["timestamp", "cumulative_wh"]].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    data["cumulative_wh"] = pd.to_numeric(data["cumulative_wh"], errors="coerce")
    if data["timestamp"].isna().any() or not np.isfinite(data["cumulative_wh"]).all():
        raise DataValidationError("Meter timestamps and cumulative readings must be valid and finite.")
    if (data["cumulative_wh"] < 0).any() or data["timestamp"].duplicated().any():
        raise DataValidationError("Meter readings must be non-negative with unique timestamps.")
    if len(data) < 2:
        raise DataValidationError("At least two meter readings are required.")
    data = data.sort_values("timestamp").reset_index(drop=True)
    # A meter counter only grows. If a value drops, the counter was reset.
    # Each reset starts a new "reset_count" group.
    # Remove the UTC timezone only after explicit UTC parsing. The resulting
    # naive clock is documented as UTC throughout the dataset and dashboard.
    data["timestamp"] = data["timestamp"].dt.tz_localize(None)
    data["reset_count"] = data["cumulative_wh"].diff().lt(0).cumsum()
    # For each full hour (boundary), find the latest reading at or before it.
    # The reading must be no older than max_age_seconds, or it stays empty.
    boundaries = pd.date_range(data.timestamp.min().floor("h"), data.timestamp.max().ceil("h"), freq="h")
    checkpoints = pd.merge_asof(
        pd.DataFrame({"boundary": boundaries}),
        data.rename(columns={"timestamp": "reading_time"}),
        left_on="boundary", right_on="reading_time", direction="backward",
        tolerance=pd.Timedelta(seconds=max_age_seconds),
    )
    # Each hour runs from one boundary (start) to the next one (end).
    # The energy used is the end meter value minus the start meter value.
    start = checkpoints.iloc[:-1].reset_index(drop=True)
    end = checkpoints.iloc[1:].reset_index(drop=True)
    difference = end.cumulative_wh - start.cumulative_wh
    # An hour is valid only if both readings exist, no counter reset happened
    # between them, and the difference is not negative.
    valid = (start.cumulative_wh.notna() & end.cumulative_wh.notna()
             & end.reset_count.eq(start.reset_count) & difference.ge(0))
    result = pd.DataFrame({
        "timestamp": boundaries[:-1],
        "energy_kwh": (difference / 1000).where(valid).to_numpy(),
        "estimated": valid.to_numpy(),
        "quality": np.where(valid, "boundary_estimated", "insufficient"),
        "start_checkpoint_age_seconds": (start.boundary - start.reading_time).dt.total_seconds().to_numpy(),
        "end_checkpoint_age_seconds": (end.boundary - end.reading_time).dt.total_seconds().to_numpy(),
        "measured_span_seconds": (end.reading_time - start.reading_time).dt.total_seconds().where(valid).to_numpy(),
    }).set_index("timestamp")
    return result


def prepare_xky_hourly(raw_path: Path, output_path: Path, audit_path: Path) -> pd.DataFrame:
    """Read only cumulative Wh and UTC timestamp; save data quality/provenance."""
    source = Path(raw_path)
    if hashlib.md5(source.read_bytes()).hexdigest() != XKY_MD5:
        raise DataValidationError("XKY source checksum is not the verified published file.")
    raw = pd.read_csv(source, usecols=["Time(UTC)", "BASE", "HCHC", "HCHP"])
    # BASE should equal HCHC + HCHP (two tariff counters). Rows where they
    # differ by more than 1 Wh are treated as unreliable.
    comparable = raw[["BASE", "HCHC", "HCHP"]].notna().all(axis=1)
    mismatches = comparable & ~np.isclose(raw.BASE, raw.HCHC + raw.HCHP, rtol=0, atol=1)
    minute = raw.rename(columns={"Time(UTC)": "timestamp", "BASE": "cumulative_wh"})
    missing_count = int(minute.cumulative_wh.isna().sum())
    # Reject disagreeing meter counters conservatively, independent of any
    # model score. Fresh checkpoints may resume after the inconsistent block.
    minute = minute.loc[~mismatches]
    minute = minute.dropna(subset=["cumulative_wh"])
    duplicates = minute[minute.timestamp.duplicated(keep=False)]
    if duplicates.groupby("timestamp").cumulative_wh.nunique().gt(1).any():
        raise DataValidationError("XKY has conflicting meter readings at the same timestamp.")
    duplicate_count = int(minute.timestamp.duplicated().sum())
    minute = minute.drop_duplicates("timestamp")
    hourly = cumulative_wh_to_hourly(minute)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.reset_index().to_csv(output_path, index=False)
    # Save a small report about the data quality, so every number is traceable.
    audit = {
        "dataset_id": "xky_2025", "source_url": XKY_SOURCE_URL,
        "file_doi": XKY_FILE_DOI, "source_file_md5": XKY_MD5,
        "source_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_rows": len(raw), "first_timestamp": str(minute.timestamp.min()),
        "missing_counter_rows_excluded": missing_count,
        "identical_duplicate_timestamps_removed": duplicate_count,
        "tariff_counter_sum_rows_verified": int((comparable & ~mismatches).sum()),
        "conflicting_counter_rows_excluded": int(mismatches.sum()),
        "last_timestamp": str(minute.timestamp.max()), "hourly_rows": len(hourly),
        "observed_hours": 0, "estimated_hours": int(hourly.energy_kwh.notna().sum()),
        "insufficient_hours": int(hourly.energy_kwh.isna().sum()),
        "retained_target_hours": int(hourly.energy_kwh.notna().sum()),
        "timezone_note": "UTC, per the source Time(UTC) column; stored as naive UTC for compatibility.",
        "energy_conversion": "Difference of cumulative BASE (Wh) at past-only hour-boundary checkpoints /1000.",
        "counter_field": "BASE (Wh), checked against HCHC + HCHP; disagreeing rows excluded. PAPP (VA) is unused.",
        "checkpoint_age_limit_seconds": 180,
        "quality_note": "All retained hours are boundary estimates; no forward interpolation or gap zero-fill.",
        "scope": "One French household's meter consumption index, subject 14, 2025; not live 2026 data.",
        "license": "Licence Ouverte / Open Licence 2.0 (etalab-2.0)",
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    return hourly
