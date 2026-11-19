"""Dataset download, aggregation, and user CSV validation utilities."""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd
import requests

from src.input_contract import ENERGY_PATTERN, INPUT_CONTRACT, TIMESTAMP_PATTERN

UCI_DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
RAW_FILENAME = "household_power_consumption.txt"
MIN_VALID_MINUTES = 55


class DataValidationError(ValueError):
    """Raised when source or uploaded data cannot be used safely."""


def download_uci_dataset(destination: Path, timeout: int = 120) -> Path:
    """Download and extract the official UCI text file, returning its path."""
    # Make the folder, and skip the download if the file is already there.
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    raw_path = destination / RAW_FILENAME
    # A file bigger than 1 MB is treated as a real, finished download.
    if raw_path.exists() and raw_path.stat().st_size > 1_000_000:
        return raw_path

    # Download the ZIP file in 1 MB pieces, then take the text file out of it.
    zip_path = destination / "individual_household_power_consumption.zip"
    with requests.get(UCI_DATASET_URL, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with zip_path.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
    with zipfile.ZipFile(zip_path) as archive:
        member = next((m for m in archive.namelist() if m.endswith(RAW_FILENAME)), None)
        if member is None:
            raise DataValidationError(f"{RAW_FILENAME} was not found in the UCI archive.")
        with archive.open(member) as source, raw_path.open("wb") as output:
            output.write(source.read())
    zip_path.unlink(missing_ok=True)
    return raw_path


def minute_power_to_hourly(frame: pd.DataFrame, min_valid: int = MIN_VALID_MINUTES) -> pd.DataFrame:
    """Aggregate minute-average kW to hourly kWh with an auditable coverage policy.

    A complete hour uses sum(kW)/60. For 55--59 readings, mean observed kW is
    used as a one-hour energy estimate. Hours below the threshold retain their
    timestamp but receive a missing energy value and are never valid targets.
    """
    # Step 1: check that the table has the columns we need.
    required = {"timestamp", "global_active_power_kw"}
    if not required.issubset(frame.columns):
        raise DataValidationError(f"Minute data must contain columns: {sorted(required)}")
    if not 1 <= min_valid <= 60:
        raise ValueError("min_valid must be between 1 and 60.")

    # Step 2: clean the data. Bad dates or numbers become empty values.
    data = frame.loc[:, ["timestamp", "global_active_power_kw"]].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["global_active_power_kw"] = pd.to_numeric(
        data["global_active_power_kw"], errors="coerce"
    )
    data = data.dropna(subset=["timestamp"]).sort_values("timestamp")
    if data.empty:
        raise DataValidationError("No valid timestamps were found in minute data.")
    if data["timestamp"].duplicated().any():
        raise DataValidationError("Minute data contains duplicate timestamps; coverage would be overstated.")
    if not (data["timestamp"] == data["timestamp"].dt.floor("min")).all():
        raise DataValidationError("Minute timestamps must be aligned to whole minutes.")
    if np.isinf(data["global_active_power_kw"]).any():
        raise DataValidationError("Power readings must be finite or explicitly missing.")
    if (data["global_active_power_kw"].dropna() < 0).any():
        raise DataValidationError("Negative power readings are not physically valid.")

    # Step 3: group the minute readings into hours.
    # Power is in kW. One hour of 60 minute-readings: kWh = sum of kW / 60.
    indexed = data.set_index("timestamp")["global_active_power_kw"]
    grouped = indexed.resample("h")
    count = grouped.count().astype("int16")
    mean_kw = grouped.mean()
    total_kw_minutes = grouped.sum(min_count=1)

    # An hour is "complete" with 60 readings. With 55 to 59 readings we estimate it.
    # With fewer than 55 readings we leave it empty, because it is not reliable.
    complete = count == 60
    estimable = (count >= min_valid) & (count < 60)
    energy = pd.Series(np.nan, index=count.index, dtype="float64")
    energy.loc[complete] = total_kw_minutes.loc[complete] / 60.0
    energy.loc[estimable] = mean_kw.loc[estimable]

    # Step 4: return one row per hour, with a quality label for each hour.
    result = pd.DataFrame(
        {
            "energy_kwh": energy,
            "valid_minutes": count,
            "coverage": count / 60.0,
            "estimated": estimable,
            "quality": np.select(
                [complete, estimable], ["observed", "estimated"], default="insufficient"
            ),
        }
    )
    result.index.name = "timestamp"
    return result


def read_uci_minute_data(path: Path) -> pd.DataFrame:
    """Read only the required UCI columns using memory-conscious dtypes.

    Input: the path of the raw UCI text file (one row per minute).
    Output: a table with two columns, timestamp and global_active_power_kw.
    A bad date or a bad number becomes an empty value instead of an error.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    # We read only 3 columns, because the file is large. Everything is read as
    # text first, so we decide ourselves how to convert it. UCI writes "?" for
    # a missing value, so we tell pandas to treat "?" as empty.
    raw = pd.read_csv(
        path,
        sep=";",
        usecols=["Date", "Time", "Global_active_power"],
        dtype={"Date": "string", "Time": "string", "Global_active_power": "string"},
        na_values=["?"],
        low_memory=False,
    )
    # Join the date and the time into one timestamp. UCI writes dates as
    # day/month/year. errors="coerce" turns a bad value into an empty one.
    timestamps = pd.to_datetime(
        raw["Date"].str.cat(raw["Time"], sep=" "),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    power = pd.to_numeric(raw["Global_active_power"], errors="coerce").astype("float32")
    return pd.DataFrame({"timestamp": timestamps, "global_active_power_kw": power})


def prepare_uci_hourly(raw_path: Path, output_path: Path, audit_path: Path) -> pd.DataFrame:
    """Convert official minute data to hourly records and save data-quality evidence.

    Inputs: the raw file path, and two paths where we save the results.
    Output: the hourly table. It is also saved as a CSV file (output_path),
    and a small JSON report about data quality is saved (audit_path).
    The report counts how many hours are observed, estimated or insufficient.
    """
    minute = read_uci_minute_data(raw_path)
    hourly = minute_power_to_hourly(minute)
    output_path = Path(output_path)
    audit_path = Path(audit_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.reset_index().to_csv(output_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    counts = hourly["quality"].value_counts().to_dict()
    audit = {
        "source_rows": int(len(minute)),
        "first_timestamp": str(minute["timestamp"].min()),
        "last_timestamp": str(minute["timestamp"].max()),
        "hourly_rows": int(len(hourly)),
        "observed_hours": int(counts.get("observed", 0)),
        "estimated_hours": int(counts.get("estimated", 0)),
        "insufficient_hours": int(counts.get("insufficient", 0)),
        "retained_target_hours": int(hourly["energy_kwh"].notna().sum()),
        "minimum_valid_minutes": MIN_VALID_MINUTES,
        "timezone_note": "Source local-clock timestamps preserved; timezone and DST are unspecified.",
        "energy_conversion": "complete hour: sum(minute-average kW)/60; 55-59 minutes: mean observed kW x 1 hour",
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return hourly


def _read_csv_source(source: str | Path | BinaryIO) -> pd.DataFrame:
    """Read bounded UTF-8 CSV without coercing values or mangling headers.

    The source can be an uploaded file object or a file path. Every value stays
    as text here. Later functions check the text, so no value is changed by
    accident. Raises DataValidationError if the file is too big, is not UTF-8,
    has too many columns or rows, or has a row with the wrong number of fields.
    """
    maximum_bytes = INPUT_CONTRACT["maximum_bytes"]
    # Case 1: a file object (for example, a file uploaded in the app).
    if hasattr(source, "read"):
        # Streamlit reuses UploadedFile objects across reruns. Reading a
        # getvalue/seekable source must not depend on its previous cursor.
        if callable(getattr(source, "getvalue", None)):
            content = source.getvalue()
        else:
            position = None
            if callable(getattr(source, "seek", None)) and source.seekable():
                position = source.tell()
                source.seek(0)
            try:
                # Read one byte more than the limit. If we get it, the file is too big.
                content = source.read(maximum_bytes + 1)
            finally:
                if position is not None:
                    source.seek(position)
    else:
        # Case 2: a file path. Check the size before we read the file.
        path = Path(source)
        if path.stat().st_size > maximum_bytes:
            raise DataValidationError("CSV exceeds the 20 MB upload limit.")
        with path.open("rb") as handle:
            content = handle.read(maximum_bytes + 1)
    if not isinstance(content, (str, bytes)):
        raise DataValidationError("CSV source must provide UTF-8 text or bytes.")
    encoded = content.encode("utf-8") if isinstance(content, str) else content
    if len(encoded) > maximum_bytes:
        raise DataValidationError("CSV exceeds the 20 MB upload limit.")
    try:
        # "utf-8-sig" also removes the invisible start mark that Excel adds.
        decoded = encoded.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DataValidationError("CSV must be UTF-8 encoded; export it as UTF-8 CSV.") from exc
    # The stdlib's smaller default field cap must not reject an otherwise
    # permitted upload containing one long, ignored column. Total bytes remain
    # bounded above, including any multibyte UTF-8 characters.
    csv.field_size_limit(maximum_bytes)
    reader = csv.reader(io.StringIO(decoded, newline=""), strict=True)
    headers = None
    records = []
    try:
        for row in reader:
            # Skip empty lines.
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            # The first non-empty line is the header line (the column names).
            if headers is None:
                headers = row
                if len(headers) > INPUT_CONTRACT["maximum_columns"]:
                    raise DataValidationError("CSV must contain no more than 32 columns.")
                continue
            # Every data row must have the same number of fields as the header.
            if len(row) != len(headers):
                raise DataValidationError(
                    f"CSV row {reader.line_num} has {len(row)} fields; expected {len(headers)}. "
                    "Check column headers and quote values containing commas."
                )
            records.append(row)
            if len(records) > INPUT_CONTRACT["maximum_rows"]:
                raise DataValidationError("CSV must contain no more than 100000 data rows.")
    except csv.Error as exc:
        raise DataValidationError(f"CSV syntax is invalid near row {reader.line_num}: {exc}") from exc
    if headers is None:
        raise DataValidationError("CSV is empty; include timestamp and energy_kwh headers and hourly data.")
    return pd.DataFrame(records, columns=headers)


def _csv_timestamp_index(values: pd.Series, clock_policy: str) -> tuple[pd.DatetimeIndex, dict[str, str]]:
    """Validate one explicit clock before constructing any pandas datetime array.

    Input: a column of timestamp text, and the clock policy ("utc" or "preserve").
    Output: a DatetimeIndex, and a dict with three short texts that say which
    clock the data uses. The app shows these texts to the user.
    A timezone offset is the "+02:00" or "Z" at the end of a timestamp.
    All rows must use the same offset, or all rows must have none.
    """
    timestamps = values.str.strip().tolist()
    offsets: set[int | None] = set()
    # Step 1: check the text of every timestamp, and collect its offset in minutes.
    for value in timestamps:
        if TIMESTAMP_PATTERN.fullmatch(value) is None:
            raise DataValidationError(
                "CSV contains invalid timestamps. Use YYYY-MM-DD HH:mm[:ss[.fraction]] "
                "with an optional Z or +HH:mm/-HH:mm offset."
            )
        suffix = re.search(r"(Z|[+-]\d{2}:\d{2})$", value, flags=re.ASCII)
        if suffix is None:
            offsets.add(None)
            continue
        token = suffix.group(0)
        if token == "Z":
            offsets.add(0)
            continue
        hours, minutes = int(token[1:3]), int(token[4:6])
        if hours > 23 or minutes > 59:
            raise DataValidationError("CSV contains invalid timezone offsets; use an offset smaller than 24 hours.")
        offsets.add((1 if token[0] == "+" else -1) * (60 * hours + minutes))
    # Step 2: the offsets must be the same in every row. "None" means the row
    # has no offset ("naive"), so naive and offset rows cannot be mixed.
    if None in offsets and len(offsets) > 1:
        raise DataValidationError(
            "CSV mixes naive and timezone-aware timestamps. Supply an offset on every row or on none."
        )
    if len(offsets) > 1:
        raise DataValidationError(
            "CSV contains inconsistent timezone offsets. Convert every row to one consistent offset, such as UTC."
        )
    try:
        # Parse scalar strings only after checking their clock; pandas never sees
        # mixed timezone types and cannot silently fall back to object dtype.
        parsed = [pd.Timestamp(value).as_unit("ns") for value in timestamps]
        index = pd.DatetimeIndex(parsed, name="timestamp")
    except (ValueError, TypeError, OverflowError) as exc:
        raise DataValidationError(
            "CSV contains invalid timestamps or dates outside the supported nanosecond range "
            "(1677-09-21 to 2262-04-11). Check calendar dates and timezone offsets."
        ) from exc

    # Step 3: write short labels that describe the input clock.
    offset = next(iter(offsets), None)
    if offset is None:
        input_clock ="Naive timestamps (no timezone offset supplied)"
    else:
        sign = "+" if offset >= 0 else "-"
        absolute = abs(offset)
        label = f"{sign}{absolute // 60:02d}:{absolute % 60:02d}"
        input_clock = f"Explicit offset {label}"
    # Step 4: apply the clock policy. "utc" moves all times to UTC (XKY data uses
    # this). "preserve" keeps the times exactly as written (UCI data uses this).
    if clock_policy == "utc":
        if offset is None:
            normalization = "Naive input timestamps are interpreted as UTC; labels are unchanged."
        else:
            index = index.tz_convert("UTC").tz_localize(None)
            normalization = "Timezone-aware input timestamps are converted to UTC and stored without an offset."
        clock = "UTC"
    else:
        clock = (
            "Source wall clock (timezone unconfirmed)"
            if offset is None else f"Input wall clock ({label})"
        )
        normalization = (
            "Supplied wall-clock labels are preserved; the source timezone is unconfirmed."
            if offset is None else "Supplied wall-clock labels and their consistent timezone offset are preserved."
        )
    return index, {
        "timestamp_clock": clock,
        "timestamp_input_clock": input_clock,
        "timestamp_normalization": normalization,
    }


def validate_hourly_csv(
    source: str | Path | BinaryIO, min_history: int = 169, *, clock_policy: str = "preserve"
) -> pd.DataFrame:
    """Validate bounded hourly CSV and apply an explicit, auditable clock policy.

    ``preserve`` retains naive wall-clock labels or a consistent supplied offset.
    ``utc`` returns naive UTC timestamps; naive input is explicitly treated as UTC.
    Input must already be ascending and aligned to whole hours after this policy.

    Inputs: a file object or path, the minimum number of rows (default 169 =
    7 days + 1 hour), and the clock policy.
    Output: a table with one column, energy_kwh, and a time index. The clock
    labels are saved in the table's attrs, so the app can show them.
    Raises DataValidationError with a clear message for the first problem found.
    """
    if clock_policy not in {"preserve", "utc"}:
        raise ValueError("clock_policy must be 'preserve' or 'utc'.")
    if not isinstance(min_history, int) or isinstance(min_history, bool) or min_history < 1:
        raise ValueError("min_history must be a positive integer.")
    try:
        frame = _read_csv_source(source)
    except DataValidationError:
        raise
    except Exception as exc:  # pandas parser errors vary by version
        raise DataValidationError(f"Could not read CSV: {exc}") from exc

    # Find the timestamp column and the energy column. Header names are not
    # case-sensitive, and a few alternative names are allowed (see the contract).
    normalized = [column.strip().lower() for column in frame.columns]
    timestamp_cols = [i for i, column in enumerate(normalized) if column in INPUT_CONTRACT["timestamp_columns"]]
    energy_cols = [i for i, column in enumerate(normalized) if column in INPUT_CONTRACT["energy_columns"]]
    if len(timestamp_cols) > 1 or len(energy_cols) > 1:
        raise DataValidationError(
            "CSV contains duplicate or ambiguous semantic headers. Keep exactly one timestamp/datetime "
            "column and one energy_kwh/consumption_kwh column."
        )
    if not timestamp_cols or not energy_cols:
        raise DataValidationError(
            "CSV requires 'timestamp' and 'energy_kwh' columns (or datetime/consumption_kwh)."
        )
    index, clock_metadata = _csv_timestamp_index(frame.iloc[:, timestamp_cols[0]], clock_policy)
    # Check the energy values as text first. This rejects empty cells, NaN,
    # Infinity and hexadecimal numbers before we convert to numbers.
    energy_text = frame.iloc[:, energy_cols[0]].str.strip()
    if not energy_text.map(lambda value: ENERGY_PATTERN.fullmatch(value) is not None).all():
        raise DataValidationError(
            "CSV contains missing or non-numeric energy values. Use finite decimal numbers in kWh; "
            "hexadecimal, NaN and Infinity are not accepted."
        )
    energy = energy_text.astype("float64").to_numpy()
    if (~np.isfinite(energy)).any() or (energy < 0).any():
        raise DataValidationError("Energy values must be finite and non-negative.")
    # Time checks: no repeated times, rows in time order, and whole hours only.
    if index.has_duplicates:
        raise DataValidationError("CSV contains duplicate timestamps.")
    if not index.is_monotonic_increasing:
        raise DataValidationError("CSV timestamps must be sorted in ascending order; sort rows before uploading.")
    if INPUT_CONTRACT["require_whole_hour"] and (
        (index.minute != 0) | (index.second != 0) | (index.microsecond != 0) | (index.nanosecond != 0)
    ).any():
        raise DataValidationError(
            "Timestamps must align to whole hours after clock normalization; minutes, seconds and fractions must be zero."
        )
    clean = pd.DataFrame({"energy_kwh": energy}, index=index)
    if len(clean) < min_history:
        raise DataValidationError(
            f"At least {min_history} consecutive hourly rows are required; received {len(clean)}."
        )
    # No gaps allowed: the time between every two rows must be exactly 1 hour.
    deltas = clean.index.to_series().diff().dropna()
    if not (deltas == pd.Timedelta(seconds=INPUT_CONTRACT["interval_seconds"])).all():
        raise DataValidationError("Timestamps must be consecutive and exactly one hour apart.")
    clean.attrs.update(clock_metadata)
    return clean
