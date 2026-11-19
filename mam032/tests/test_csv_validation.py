import csv
import io
import json
import warnings
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.data import DataValidationError, validate_hourly_csv
from src.input_contract import INPUT_CONTRACT


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "csv_contract_cases.json"
CONTRACT_CASES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]


def _case_csv(case: dict) -> str:
    """Render the language-neutral cases also consumed by the Node verifier."""
    specification = case["rows"]
    if isinstance(specification, list):
        rows = specification
    else:
        generator = specification["generate"]
        timestamps = pd.date_range(
            generator["start"], periods=generator["count"],
            freq=pd.Timedelta(minutes=generator.get("step_minutes", 60)),
        )
        rows = [
            [timestamp.strftime("%Y-%m-%d %H:%M:%S") + generator.get("suffix", ""), generator.get("energy", "1.25")]
            for timestamp in timestamps
        ]
        for index, row in specification.get("replace", {}).items():
            rows[int(index)] = row
        rows = [row for index, row in enumerate(rows) if index not in specification.get("drop", [])]
        if specification.get("reverse"):
            rows.reverse()
    output = io.StringIO(newline="")
    if case.get("bom"):
        output.write("\ufeff")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(case["headers"])
    if case.get("blank_lines"):
        writer.writerow([])
    for row in rows:
        writer.writerow(row)
        if case.get("blank_lines"):
            writer.writerow([])
    return output.getvalue()


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=lambda case: case["name"])
def test_shared_csv_contract(case: dict) -> None:
    content = io.BytesIO(_case_csv(case).encode("utf-8"))
    kwargs = {"clock_policy": case["clock_policy"], "min_history": case.get("min_history", 169)}
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        if not case["valid"]:
            with pytest.raises(DataValidationError, match=case["error_contains"]):
                validate_hourly_csv(content, **kwargs)
            return
        result = validate_hourly_csv(content, **kwargs)
    assert list(result.columns) == ["energy_kwh"]
    assert result.index.name == "timestamp"
    assert result.index.is_unique and result.index.is_monotonic_increasing
    assert result.energy_kwh.ge(0).all()
    assert all(result.attrs.get(key) for key in (
        "timestamp_clock", "timestamp_input_clock", "timestamp_normalization"
    ))
    if "expected_rows" in case:
        assert len(result) == case["expected_rows"]
    if "expected_start" in case:
        assert result.index[0].isoformat() == case["expected_start"]
    if "expected_clock" in case:
        assert result.attrs["timestamp_clock"] == case["expected_clock"]


def test_upload_can_be_read_repeatedly_across_app_reruns() -> None:
    upload = io.BytesIO(_case_csv(CONTRACT_CASES[0]).encode("utf-8"))
    first = validate_hourly_csv(upload)
    upload.seek(0, io.SEEK_END)
    second = validate_hourly_csv(upload)
    assert_frame_equal(first, second)
    assert first.attrs == second.attrs


def test_seekable_file_cursor_is_restored(tmp_path) -> None:
    path = tmp_path / "hourly.csv"
    path.write_text(_case_csv(CONTRACT_CASES[0]), encoding="utf-8")
    with path.open("rb") as handle:
        handle.seek(13)
        result = validate_hourly_csv(handle)
        assert handle.tell() == 13
        assert_frame_equal(result, validate_hourly_csv(path))


def test_offset_policies_preserve_energy_and_instant() -> None:
    content = _case_csv(next(case for case in CONTRACT_CASES if case["name"] == "positive_offset_preserve"))
    preserved = validate_hourly_csv(io.StringIO(content))
    utc = validate_hourly_csv(io.StringIO(content), clock_policy="utc")
    assert preserved.index.tz is not None
    assert utc.index.tz is None
    assert preserved.index.tz_convert("UTC").tz_localize(None).equals(utc.index)
    assert preserved.energy_kwh.tolist() == utc.energy_kwh.tolist()
    assert preserved.attrs["timestamp_input_clock"] == "Explicit offset +06:00"
    assert utc.attrs["timestamp_input_clock"] == "Explicit offset +06:00"


@pytest.mark.parametrize("size_delta,valid", [(0, True), (1, False)])
def test_byte_limit_boundary(size_delta: int, valid: bool) -> None:
    base = "timestamp,energy_kwh,note\n2024-01-01 00:00:00,1,"
    maximum = INPUT_CONTRACT["maximum_bytes"]
    content = (base + "x" * (maximum - len(base) - 1 + size_delta) + "\n").encode("utf-8")
    assert len(content) == maximum + size_delta
    if valid:
        assert len(validate_hourly_csv(io.BytesIO(content), min_history=1)) == 1
    else:
        with pytest.raises(DataValidationError, match="20 MB"):
            validate_hourly_csv(io.BytesIO(content), min_history=1)


def test_byte_limit_counts_utf8_bytes() -> None:
    content = "timestamp,energy_kwh,note\n2024-01-01 00:00:00,1," + "é" * (INPUT_CONTRACT["maximum_bytes"] // 2)
    with pytest.raises(DataValidationError, match="20 MB"):
        validate_hourly_csv(io.StringIO(content), min_history=1)


@pytest.mark.parametrize("columns,valid", [(32, True), (33, False)])
def test_column_limit_boundary(columns: int, valid: bool) -> None:
    case = {
        "headers": ["timestamp", "energy_kwh"] + [f"extra_{index}" for index in range(columns - 2)],
        "rows": [["2024-01-01 00:00:00", "1"] + ["ignored"] * (columns - 2)],
    }
    source = io.StringIO(_case_csv(case))
    if valid:
        assert list(validate_hourly_csv(source, min_history=1).columns) == ["energy_kwh"]
    else:
        with pytest.raises(DataValidationError, match="32 columns"):
            validate_hourly_csv(source, min_history=1)


@pytest.mark.parametrize("rows,valid", [(100000, True), (100001, False)])
def test_row_limit_boundary(rows: int, valid: bool) -> None:
    case = {"headers": ["timestamp", "energy_kwh"], "rows": {"generate": {"start": "2024-01-01 00:00:00", "count": rows}}}
    source = io.StringIO(_case_csv(case))
    if valid:
        assert len(validate_hourly_csv(source)) == rows
    else:
        with pytest.raises(DataValidationError, match="100000 data rows"):
            validate_hourly_csv(source)


@pytest.mark.parametrize("content,message", [
    ('timestamp,energy_kwh\n"2024-01-01 00:00:00,1\n', "syntax"),
    ("timestamp,energy_kwh\n2024-01-01 00:00:00,1,unexpected\n", "fields"),
    ("timestamp,energy_kwh\n2024-01-01 00:00:00\n", "fields"),
    ("", "empty"),
])
def test_malformed_csv_is_actionable(content: str, message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        validate_hourly_csv(io.StringIO(content), min_history=1)


def test_non_utf8_csv_rejected() -> None:
    with pytest.raises(DataValidationError, match="UTF-8"):
        validate_hourly_csv(io.BytesIO(b"timestamp,energy_kwh\n\xff,1\n"), min_history=1)


@pytest.mark.parametrize("kwargs", [{"clock_policy": "local"}, {"min_history": 0}, {"min_history": True}, {"min_history": 1.5}])
def test_invalid_validator_options(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        validate_hourly_csv(io.StringIO(_case_csv(CONTRACT_CASES[0])), **kwargs)


def test_csv_contract_loaded_from_authoritative_json() -> None:
    contract_path = Path(__file__).resolve().parents[1] / "config" / "input_contract.json"
    assert INPUT_CONTRACT == json.loads(contract_path.read_text(encoding="utf-8"))
