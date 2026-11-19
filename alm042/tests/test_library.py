"""Tests for the persistent audio library (SQLite + API + integration)."""

from __future__ import annotations

import json
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from conftest import error_payload, http_get, http_post_json, wait_for_job
from voiceforge.app_core import AppError
from voiceforge.library import Library, validate_name


def request(method: str, url: str, payload: dict | None = None):
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=60) as response:
        return response.status, json.loads(response.read())


def library_records(server, query=""):
    with http_get(f"{server}/api/library{query}") as response:
        return json.loads(response.read())


def generate(server, text="Hello world.", engine="voiceforge", **extra):
    payload = {"text": text, "engine": engine, **extra}

    with http_post_json(server + "/api/generate", payload) as response:
        return json.loads(response.read())


# ----------------------------------------------------------------------
# Database
# ----------------------------------------------------------------------
def test_database_is_created_with_table_and_indexes(tmp_path):
    library = Library(
        db_path=tmp_path / "library.db",
        audio_dir=tmp_path / "audio",
    )

    assert library.db_path.is_file()
    assert library.audio_dir.is_dir()

    with sqlite3.connect(library.db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        version = connection.execute("PRAGMA user_version").fetchone()[0]

    assert "audio_records" in tables
    assert "idx_records_created" in indexes
    assert "idx_records_name" in indexes
    assert "idx_records_engine" in indexes
    assert "idx_records_source_type" in indexes
    assert version == 2


def test_reopening_keeps_records(tmp_path, wav_file):
    library = Library(tmp_path / "library.db", tmp_path / "audio")

    first = library.record(
        text="Hello.",
        source_filename="a.wav",
        source_path=wav_file,
        engine="voiceforge",
        sample_rate=22050,
        duration_seconds=1.0,
        source_type="studio",
    )

    reopened = Library(tmp_path / "library.db", tmp_path / "audio")
    record = reopened.get(first["id"])

    assert record["name"] == first["name"]
    assert record["filename"] == first["filename"]


def test_record_copies_audio_and_stores_metadata(tmp_path, wav_file):
    library = Library(tmp_path / "library.db", tmp_path / "audio")

    record = library.record(
        text="Hello there.",
        source_filename="source.wav",
        source_path=wav_file,
        engine="kokoro",
        voice="af_heart",
        speed=1.25,
        sample_rate=24000,
        duration_seconds=2.5,
        source_type="studio",
        name="My recording",
    )

    stored = tmp_path / "audio" / record["filename"]

    assert stored.is_file()
    assert record["name"] == "My recording"
    assert record["engine"] == "kokoro"
    assert record["voice"] == "af_heart"
    assert record["speed"] == 1.25
    assert record["sample_rate"] == 24000
    assert record["duration_seconds"] == 2.5
    assert record["file_size"] == stored.stat().st_size
    assert record["source_type"] == "studio"
    assert record["created_at"] == record["updated_at"]


def test_duplicate_source_is_not_recorded_twice(tmp_path, wav_file):
    library = Library(tmp_path / "library.db", tmp_path / "audio")

    kwargs = dict(
        text="Hello.",
        source_filename="same.wav",
        source_path=wav_file,
        engine="voiceforge",
        sample_rate=22050,
        duration_seconds=1.0,
        source_type="studio",
    )

    first = library.record(**kwargs)
    second = library.record(**kwargs)

    records, total = library.list_records()

    assert first["id"] == second["id"]
    assert total == 1
    assert len(records) == 1


def test_duplicate_job_item_is_not_recorded_twice(tmp_path, wav_file):
    library = Library(tmp_path / "library.db", tmp_path / "audio")

    base = dict(
        text="Hello.",
        engine="voiceforge",
        sample_rate=22050,
        duration_seconds=1.0,
        source_type="batch",
        job_id="job-1",
        job_item_id="item-1",
    )

    first = library.record(source_filename="one.wav", source_path=wav_file, **base)
    second = library.record(source_filename="two.wav", source_path=wav_file, **base)

    _, total = library.list_records()

    assert first["id"] == second["id"]
    assert total == 1


def test_rename_updates_name_and_timestamp(tmp_path, wav_file):
    library = Library(tmp_path / "library.db", tmp_path / "audio")
    record = library.record(
        text="Hello.",
        source_filename="a.wav",
        source_path=wav_file,
        engine="voiceforge",
        sample_rate=22050,
        duration_seconds=1.0,
        source_type="studio",
    )

    updated = library.rename(record["id"], "  New name  ")

    assert updated["name"] == "New name"
    assert updated["filename"] == record["filename"]
    assert updated["original_text"] == "Hello."
    assert updated["updated_at"] >= record["updated_at"]


@pytest.mark.parametrize("name", ["", "   ", "x" * 81, None, 42])
def test_invalid_names_are_rejected(name):
    with pytest.raises(AppError) as error:
        validate_name(name)

    assert error.value.code == "invalid_name"


def test_search_is_parameterised_and_safe(tmp_path, wav_file):
    library = Library(tmp_path / "library.db", tmp_path / "audio")

    for index, text in enumerate(["Alpha one", "Beta two", "Gamma three"]):
        library.record(
            text=text,
            source_filename=f"{index}.wav",
            source_path=wav_file,
            engine="voiceforge",
            sample_rate=22050,
            duration_seconds=1.0,
            source_type="studio",
        )

    records, total = library.list_records(query="Alpha")
    assert total == 1
    assert records[0]["original_text"] == "Alpha one"

    # SQL-ish input is treated as literal text, never executed.
    records, total = library.list_records(query="'; DROP TABLE audio_records; --")
    assert total == 0
    assert library.list_records()[1] == 3


def test_search_matches_name_and_text(tmp_path, wav_file):
    library = Library(tmp_path / "library.db", tmp_path / "audio")

    library.record(
        text="Some spoken sentence.",
        source_filename="a.wav",
        source_path=wav_file,
        engine="voiceforge",
        sample_rate=22050,
        duration_seconds=1.0,
        source_type="studio",
        name="Welcome piece",
    )

    assert library.list_records(query="Welcome")[1] == 1
    assert library.list_records(query="spoken")[1] == 1
    assert library.list_records(query="nothing")[1] == 0


def test_sorting_and_filtering(tmp_path, wav_file):
    library = Library(tmp_path / "library.db", tmp_path / "audio")

    library.record(
        text="Short.",
        source_filename="a.wav",
        source_path=wav_file,
        engine="kokoro",
        sample_rate=24000,
        duration_seconds=1.0,
        source_type="studio",
        name="Zulu",
    )
    library.record(
        text="Longer text.",
        source_filename="b.wav",
        source_path=wav_file,
        engine="voiceforge",
        sample_rate=22050,
        duration_seconds=9.0,
        source_type="batch",
        name="Alpha",
    )

    by_name = library.list_records(sort="name")[0]
    assert by_name[0]["name"] == "Alpha"

    by_duration = library.list_records(sort="duration")[0]
    assert by_duration[0]["duration_seconds"] == 9.0

    assert library.list_records(engine="kokoro")[1] == 1
    assert library.list_records(source="batch")[1] == 1
    assert library.list_records(engine="nope")[1] == 0


def test_pagination(tmp_path, wav_file):
    library = Library(tmp_path / "library.db", tmp_path / "audio")

    for index in range(5):
        library.record(
            text=f"Text {index}",
            source_filename=f"{index}.wav",
            source_path=wav_file,
            engine="voiceforge",
            sample_rate=22050,
            duration_seconds=1.0,
            source_type="studio",
        )

    page1, total = library.list_records(page=1, limit=2)
    page2, _ = library.list_records(page=2, limit=2)

    assert total == 5
    assert len(page1) == 2
    assert len(page2) == 2
    assert {record["id"] for record in page1}.isdisjoint(
        {record["id"] for record in page2}
    )


def test_invalid_sort_is_rejected(tmp_path):
    library = Library(tmp_path / "library.db", tmp_path / "audio")

    with pytest.raises(AppError) as error:
        library.list_records(sort="created_at; DROP TABLE audio_records")

    assert error.value.code == "invalid_sort"


def test_delete_removes_record_and_file(tmp_path, wav_file):
    library = Library(tmp_path / "library.db", tmp_path / "audio")
    record = library.record(
        text="Hello.",
        source_filename="a.wav",
        source_path=wav_file,
        engine="voiceforge",
        sample_rate=22050,
        duration_seconds=1.0,
        source_type="studio",
    )

    stored = tmp_path / "audio" / record["filename"]
    assert stored.is_file()

    library.delete(record["id"])

    assert not stored.exists()
    assert library.list_records()[1] == 0


def test_delete_tolerates_missing_file(tmp_path, wav_file):
    library = Library(tmp_path / "library.db", tmp_path / "audio")
    record = library.record(
        text="Hello.",
        source_filename="a.wav",
        source_path=wav_file,
        engine="voiceforge",
        sample_rate=22050,
        duration_seconds=1.0,
        source_type="studio",
    )

    (tmp_path / "audio" / record["filename"]).unlink()

    library.delete(record["id"])

    assert library.list_records()[1] == 0


def test_delete_rolls_back_when_file_cannot_be_removed(tmp_path, wav_file):
    library = Library(tmp_path / "library.db", tmp_path / "audio")
    record = library.record(
        text="Hello.",
        source_filename="a.wav",
        source_path=wav_file,
        engine="voiceforge",
        sample_rate=22050,
        duration_seconds=1.0,
        source_type="studio",
    )

    stored = tmp_path / "audio" / record["filename"]
    stored.unlink()
    stored.mkdir()  # a directory cannot be unlinked as a file

    with pytest.raises(AppError) as error:
        library.delete(record["id"])

    assert error.value.code == "delete_failed"
    # The record is still there and points at the (broken) path.
    assert library.get(record["id"])["id"] == record["id"]


def test_path_traversal_filename_is_rejected(tmp_path, wav_file):
    library = Library(tmp_path / "library.db", tmp_path / "audio")
    record = library.record(
        text="Hello.",
        source_filename="a.wav",
        source_path=wav_file,
        engine="voiceforge",
        sample_rate=22050,
        duration_seconds=1.0,
        source_type="studio",
    )

    with library._connect() as connection:  # noqa: SLF001 - deliberate test
        connection.execute(
            "UPDATE audio_records SET filename = ? WHERE id = ?",
            ("../escape.wav", record["id"]),
        )

    with pytest.raises(AppError) as error:
        library.audio_path(record["id"])

    assert error.value.code in ("invalid_filename", "audio_not_found")


# ----------------------------------------------------------------------
# Library API
# ----------------------------------------------------------------------
def test_studio_generation_creates_one_record(server):
    result = generate(server, "Library test text.", engine="voiceforge")

    payload = library_records(server)

    assert payload["ok"] is True
    assert payload["pagination"]["total"] == 1

    record = payload["records"][0]
    assert record["original_text"] == result["text"]
    assert record["engine"] == "voiceforge"
    assert record["source_type"] == "studio"
    assert record["filename"].endswith(".wav")
    assert record["file_size"] > 0


def test_legacy_generate_also_creates_a_record(server):
    with http_post_json(
        server + "/generate",
        {"text": "Legacy path.", "engine": "voiceforge"},
    ) as response:
        json.loads(response.read())

    assert library_records(server)["pagination"]["total"] == 1


def test_library_audio_playback_and_download(server):
    generate(server, "Playable audio.", engine="voiceforge")
    record = library_records(server)["records"][0]

    with http_get(f"{server}/api/library/{record['id']}/audio") as response:
        audio = response.read()
        disposition = response.headers["Content-Disposition"]

    assert response.status == 200
    assert audio[:4] == b"RIFF"
    assert disposition.startswith("inline")

    with http_get(
        f"{server}/api/library/{record['id']}/audio?download=1"
    ) as response:
        response.read()
        assert response.headers["Content-Disposition"].startswith("attachment")


def test_record_rename_via_api(server):
    generate(server, "Rename me.", engine="voiceforge")
    record = library_records(server)["records"][0]

    status, payload = request(
        "PATCH",
        f"{server}/api/library/{record['id']}",
        {"name": "Renamed audio"},
    )

    assert status == 200
    assert payload["record"]["name"] == "Renamed audio"
    assert payload["record"]["filename"] == record["filename"]

    fetched = request("GET", f"{server}/api/library/{record['id']}")[1]
    assert fetched["record"]["name"] == "Renamed audio"


@pytest.mark.parametrize("name", ["", "   ", "x" * 81])
def test_invalid_rename_is_rejected(server, name):
    generate(server, "Rename me.", engine="voiceforge")
    record = library_records(server)["records"][0]

    with pytest.raises(urllib.error.HTTPError) as error:
        request("PATCH", f"{server}/api/library/{record['id']}", {"name": name})

    payload = error_payload(error.value)

    assert error.value.code == 400
    assert payload["code"] == "invalid_name"


def test_missing_record_returns_404(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        request("GET", f"{server}/api/library/doesnotexist")

    assert error_payload(error.value)["code"] == "record_not_found"
    assert error.value.code == 404


def test_delete_via_api_removes_record_and_file(server, tmp_path):
    generate(server, "Delete me.", engine="voiceforge")
    record = library_records(server)["records"][0]

    status, payload = request("DELETE", f"{server}/api/library/{record['id']}")

    assert status == 200
    assert payload["deleted"]["id"] == record["id"]
    assert library_records(server)["pagination"]["total"] == 0

    with pytest.raises(urllib.error.HTTPError) as error:
        request("GET", f"{server}/api/library/{record['id']}")

    assert error.value.code == 404


def test_missing_wav_is_reported_but_page_still_lists(server, tmp_path):
    generate(server, "Missing file.", engine="voiceforge")
    record = library_records(server)["records"][0]

    (tmp_path / "library" / record["filename"]).unlink()

    with pytest.raises(urllib.error.HTTPError) as error:
        http_get(f"{server}/api/library/{record['id']}/audio")

    assert error_payload(error.value)["code"] == "audio_not_found"

    # The list endpoint still works.
    assert library_records(server)["pagination"]["total"] == 1


def test_library_list_search_sort_filter_pagination(server):
    generate(server, "Alpha recording.", engine="voiceforge")
    generate(server, "Beta recording.", engine="voiceforge")

    assert library_records(server, "?q=Alpha")["pagination"]["total"] == 1
    assert library_records(server, "?sort=name")["records"][0]["name"].lower() < (
        library_records(server, "?sort=name")["records"][1]["name"].lower()
    )
    assert library_records(server, "?engine=kokoro")["pagination"]["total"] == 0
    assert library_records(server, "?limit=1")["pagination"]["pages"] == 2


def test_invalid_list_parameters(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        http_get(server + "/api/library?page=abc")

    assert error_payload(error.value)["code"] == "invalid_parameter"

    with pytest.raises(urllib.error.HTTPError) as error:
        http_get(server + "/api/library?sort=bogus")

    assert error_payload(error.value)["code"] == "invalid_sort"

    with pytest.raises(urllib.error.HTTPError) as error:
        http_get(server + "/api/library?source=bogus")

    assert error_payload(error.value)["code"] == "invalid_source"


# ----------------------------------------------------------------------
# Job integration
# ----------------------------------------------------------------------
def test_long_text_job_creates_one_combined_record(server):
    text = "Hello there. " * 30

    with http_post_json(
        server + "/api/jobs",
        {
            "type": "long_text",
            "engine": "voiceforge",
            "items": [{"name": "Long one", "text": text}],
        },
    ) as response:
        job_id = json.loads(response.read())["job"]["id"]

    job = wait_for_job(server, job_id)
    assert job["status"] == "completed"

    records = library_records(server, "?source=long_text")

    assert records["pagination"]["total"] == 1
    assert records["records"][0]["job_id"] == job_id
    assert records["records"][0]["name"] == "Long one"


def test_batch_job_records_each_successful_item(server):
    with http_post_json(
        server + "/api/jobs",
        {
            "type": "batch",
            "engine": "voiceforge",
            "items": [
                {"name": "First", "text": "First batch item."},
                {"name": "Second", "text": "Second batch item."},
            ],
        },
    ) as response:
        job_id = json.loads(response.read())["job"]["id"]

    wait_for_job(server, job_id)

    records = library_records(server, "?source=batch")

    assert records["pagination"]["total"] == 2
    assert {record["name"] for record in records["records"]} == {"First", "Second"}


def test_failed_items_create_no_records(server):
    with http_post_json(
        server + "/api/jobs",
        {
            "type": "batch",
            "engine": "voiceforge",
            "items": [
                {"name": "Good", "text": "Fine text."},
                {"name": "Bad", "text": "Digits 123 fail."},
            ],
        },
    ) as response:
        job_id = json.loads(response.read())["job"]["id"]

    wait_for_job(server, job_id)

    records = library_records(server, "?source=batch")

    assert records["pagination"]["total"] == 1
    assert records["records"][0]["name"] == "Good"


def test_repeated_polling_does_not_duplicate_records(server):
    with http_post_json(
        server + "/api/jobs",
        {
            "type": "batch",
            "engine": "voiceforge",
            "items": [{"name": "Once", "text": "Only once."}],
        },
    ) as response:
        job_id = json.loads(response.read())["job"]["id"]

    wait_for_job(server, job_id)

    for _ in range(5):
        with http_get(f"{server}/api/jobs/{job_id}") as response:
            json.loads(response.read())

    assert library_records(server, "?source=batch")["pagination"]["total"] == 1


def test_records_survive_a_new_library_instance(tmp_path, wav_file):
    """Simulates restarting the server: same db, new instance."""

    library = Library(tmp_path / "library.db", tmp_path / "audio")
    library.record(
        text="Persisted.",
        source_filename="a.wav",
        source_path=wav_file,
        engine="voiceforge",
        sample_rate=22050,
        duration_seconds=1.0,
        source_type="studio",
    )

    restarted = Library(tmp_path / "library.db", tmp_path / "audio")
    records, total = restarted.list_records()

    assert total == 1
    assert records[0]["original_text"] == "Persisted."
    assert (tmp_path / "audio" / records[0]["filename"]).is_file()
