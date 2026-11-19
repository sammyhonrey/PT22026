"""Phase 5 tests: documents, pronunciation, MP3 export and presets."""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request

import numpy as np
import pytest
import soundfile as sf

from conftest import error_payload, http_get, http_post_json
from voiceforge.app_core import AppError
from voiceforge.document_reader import extract_document
from voiceforge.library import Library
from voiceforge.mp3 import (
    SUPPORTED_BITRATES,
    cached_mp3,
    encoder_available,
    export_mp3,
    validate_bitrate,
)
from voiceforge.pronunciation import apply_dictionary


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def make_pdf(pages: list[str]) -> bytes:
    """Build a small valid PDF with one text line per page."""

    objects: list[bytes] = []

    # 1 catalog, 2 pages, 3 font, then page/content pairs
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [] /Count 0 >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    kids = []

    for text in pages:
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 18 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")

        content_number = len(objects) + 2
        page_number = len(objects) + 1

        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {content_number} 0 R "
                f"/Resources << /Font << /F1 3 0 R >> >> >>"
            ).encode()
        )
        objects.append(
            f"<< /Length {len(stream)} >>\nstream\n".encode()
            + stream
            + b"\nendstream"
        )
        kids.append(f"{page_number} 0 R")

    objects[1] = (
        f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    ).encode()

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]

    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"

    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"

    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode()

    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()

    return bytes(out)


def make_docx(paragraphs: list[str], table: list[list[str]] | None = None) -> bytes:
    import docx

    document = docx.Document()
    document.add_heading(paragraphs[0], level=1)

    for text in paragraphs[1:]:
        document.add_paragraph(text)

    if table:
        created = document.add_table(rows=len(table), cols=len(table[0]))

        for row_index, row in enumerate(table):
            for column_index, cell in enumerate(row):
                created.cell(row_index, column_index).text = cell

    buffer = io.BytesIO()
    document.save(buffer)

    return buffer.getvalue()


def write_wav(path, seconds=1.0, sample_rate=22050, channels=1):
    frames = int(sample_rate * seconds)
    time = np.arange(frames) / sample_rate
    mono = (0.4 * np.sin(2 * np.pi * 220 * time)).astype("float32")
    data = mono.reshape(-1, 1)

    if channels == 2:
        data = np.repeat(data, 2, axis=1)

    sf.write(str(path), data, sample_rate, subtype="PCM_16")
    return path


def request(method: str, url: str, payload: dict | None = None):
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=120) as response:
        raw = response.read()

        try:
            return response.status, json.loads(raw)
        except json.JSONDecodeError:
            return response.status, raw


# ----------------------------------------------------------------------
# Document extraction
# ----------------------------------------------------------------------
def test_pdf_extraction_single_page():
    document = extract_document("notes.pdf", make_pdf(["Hello from a PDF."]))

    assert document["format"] == "pdf"
    assert document["pages"] == 1
    assert "Hello from a PDF." in document["text"]
    assert document["characters"] == len(document["text"])


def test_pdf_multi_page_keeps_order():
    document = extract_document(
        "multi.pdf",
        make_pdf(["First page text.", "Second page text."]),
    )

    assert document["pages"] == 2
    assert document["text"].index("First") < document["text"].index("Second")


def test_pdf_without_text_explains_ocr_limitation():
    with pytest.raises(AppError) as error:
        extract_document("scan.pdf", make_pdf([""]))

    assert error.value.code == "no_extractable_text"
    assert "OCR" in error.value.message


def test_encrypted_pdf_is_rejected():
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    writer.append(PdfReader(io.BytesIO(make_pdf(["Secret."]))))
    writer.encrypt("hunter2")

    buffer = io.BytesIO()
    writer.write(buffer)

    with pytest.raises(AppError) as error:
        extract_document("locked.pdf", buffer.getvalue())

    assert error.value.code == "encrypted_document"


def test_docx_extraction_with_headings_and_tables():
    data = make_docx(
        ["Report title", "First paragraph.", "Second paragraph."],
        table=[["Name", "Value"], ["Alpha", "1"]],
    )

    document = extract_document("report.docx", data)

    assert document["format"] == "docx"
    assert "Report title" in document["text"]
    assert "First paragraph." in document["text"]
    assert "Name | Value" in document["text"]
    assert "Alpha | 1" in document["text"]


def test_docx_table_order_is_preserved():
    data = make_docx(["Title", "Before table."], table=[["Cell"]])
    text = extract_document("order.docx", data)["text"]

    assert text.index("Before table.") < text.index("Cell")


def test_empty_docx_is_rejected():
    data = make_docx([" "])

    with pytest.raises(AppError) as error:
        extract_document("empty.docx", data)

    assert error.value.code == "empty_document"


def test_unsupported_extension_is_rejected():
    with pytest.raises(AppError) as error:
        extract_document("notes.txt", b"hello")

    assert error.value.code == "unsupported_document"


def test_invalid_pdf_is_rejected():
    with pytest.raises(AppError) as error:
        extract_document("broken.pdf", b"%PDF-1.4 this is not a pdf")

    assert error.value.code == "invalid_document"


def test_invalid_docx_is_rejected():
    with pytest.raises(AppError) as error:
        extract_document("broken.docx", b"PK not really a zip")

    assert error.value.code == "invalid_document"


def test_oversized_document_is_rejected():
    with pytest.raises(AppError) as error:
        extract_document("big.pdf", b"%PDF" + b"0" * (10 * 1024 * 1024 + 10))

    assert error.value.code == "document_too_large"


def test_empty_document_is_rejected():
    with pytest.raises(AppError) as error:
        extract_document("empty.pdf", b"")

    assert error.value.code == "empty_document"


# ----------------------------------------------------------------------
# Pronunciation
# ----------------------------------------------------------------------
ENTRY = {"word": "Abdulrahman", "pronunciation": "Ab-dool-rah-man"}


def test_replacement_is_case_insensitive():
    text, count = apply_dictionary("abdulrahman and ABDULRAHMAN", [ENTRY])

    assert text == "Ab-dool-rah-man and Ab-dool-rah-man"
    assert count == 2


def test_whole_word_matching_only():
    text, count = apply_dictionary("Abdulrahmanic and Abdulrahman.", [ENTRY])

    assert text == "Abdulrahmanic and Ab-dool-rah-man."
    assert count == 1


def test_punctuation_is_preserved():
    text, _ = apply_dictionary("Hi, Abdulrahman! (Abdulrahman?)", [ENTRY])

    assert text == "Hi, Ab-dool-rah-man! (Ab-dool-rah-man?)"


def test_phrases_are_supported():
    entries = [{"word": "New York", "pronunciation": "New York City"}]

    text, count = apply_dictionary("I live in New York.", entries)

    assert text == "I live in New York City."
    assert count == 1


def test_longest_match_wins():
    entries = [
        {"word": "New", "pronunciation": "Noo"},
        {"word": "New York", "pronunciation": "NY"},
    ]

    text, _ = apply_dictionary("New York and New Jersey", entries)

    assert text == "NY and Noo Jersey"


def test_replacements_are_not_recursive():
    entries = [
        {"word": "alpha", "pronunciation": "beta"},
        {"word": "beta", "pronunciation": "gamma"},
    ]

    text, _ = apply_dictionary("alpha", entries)

    assert text == "beta"


def test_unicode_words_are_matched():
    entries = [{"word": "café", "pronunciation": "ka-fay"}]

    text, count = apply_dictionary("The café is open.", entries)

    assert text == "The ka-fay is open."
    assert count == 1


def test_no_entries_returns_text_unchanged():
    text, count = apply_dictionary("Hello.", [])

    assert text == "Hello."
    assert count == 0


# ----------------------------------------------------------------------
# MP3
# ----------------------------------------------------------------------
def test_wav_to_mp3_conversion(tmp_path):
    wav = write_wav(tmp_path / "a.wav", seconds=1.0)
    result = export_mp3(wav, tmp_path / "a.mp3", 192)

    assert result["bitrate"] == 192
    assert result["duration_seconds"] == pytest.approx(1.0, abs=0.1)
    assert result["file_size"] > 1000
    assert result["encoder"] == "lameenc"

    data = (tmp_path / "a.mp3").read_bytes()
    assert data[:3] == b"ID3" or (data[0] == 0xFF and (data[1] & 0xE0) == 0xE0)


def test_original_wav_is_not_modified(tmp_path):
    wav = write_wav(tmp_path / "a.wav")
    before = wav.read_bytes()

    export_mp3(wav, tmp_path / "a.mp3", 128)

    assert wav.read_bytes() == before


def test_all_bitrates_work_and_follow_size_order(tmp_path):
    wav = write_wav(tmp_path / "a.wav", seconds=1.0)
    sizes = {}

    for bitrate in SUPPORTED_BITRATES:
        result = export_mp3(wav, tmp_path / f"a{bitrate}.mp3", bitrate)
        sizes[bitrate] = result["file_size"]

    assert sizes[128] < sizes[320]


@pytest.mark.parametrize("bitrate", [96, 256, 999, "fast", None])
def test_invalid_bitrate_is_rejected(bitrate):
    with pytest.raises(AppError) as error:
        validate_bitrate(bitrate)

    assert error.value.code == "invalid_bitrate"


def test_invalid_wav_is_rejected(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")

    with pytest.raises(AppError) as error:
        export_mp3(bad, tmp_path / "out.mp3")

    assert error.value.code in ("invalid_audio", "audio_not_found")


def test_missing_encoder_gives_clear_error(tmp_path, monkeypatch):
    import voiceforge.mp3 as mp3

    wav = write_wav(tmp_path / "a.wav")

    monkeypatch.setattr(mp3, "_lameenc", lambda: None)
    monkeypatch.setattr(mp3, "_ffmpeg_path", lambda: None)

    with pytest.raises(AppError) as error:
        export_mp3(wav, tmp_path / "out.mp3")

    assert error.value.code == "mp3_unavailable"
    assert "lameenc" in error.value.message


def test_cached_mp3_is_reused(tmp_path):
    wav = write_wav(tmp_path / "a.wav")
    cache = tmp_path / "cache"

    first, _ = cached_mp3(wav, cache, 192)
    second, _ = cached_mp3(wav, cache, 192)

    assert first == second
    assert first.is_file()


def test_encoder_is_available_in_this_environment():
    assert encoder_available() is True


# ----------------------------------------------------------------------
# Library API: documents, dictionary, presets, MP3
# ----------------------------------------------------------------------
def test_document_extract_endpoint(server):
    data = base64.b64encode(
        make_docx(["Doc title", "Some text for speech."])
    ).decode()

    status, payload = request(
        "POST",
        server + "/api/documents/extract",
        {"name": "speech.docx", "data_base64": data},
    )

    assert status == 200
    assert payload["document"]["format"] == "docx"
    assert "Some text for speech." in payload["document"]["text"]


def test_document_extract_rejects_unknown_extension(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        request(
            "POST",
            server + "/api/documents/extract",
            {
                "name": "notes.txt",
                "data_base64": base64.b64encode(b"hello").decode(),
            },
        )

    assert error_payload(error.value)["code"] == "unsupported_document"


def test_pronunciation_crud(server):
    status, created = request(
        "POST",
        server + "/api/pronunciation",
        {"word": "Abdulrahman", "pronunciation": "Ab-dool-rah-man"},
    )

    assert status == 201
    entry_id = created["entry"]["id"]

    status, listed = request("GET", server + "/api/pronunciation")
    assert len(listed["entries"]) == 1

    status, searched = request("GET", server + "/api/pronunciation?q=abdul")
    assert len(searched["entries"]) == 1

    status, updated = request(
        "PATCH",
        f"{server}/api/pronunciation/{entry_id}",
        {
            "word": "Abdulrahman",
            "pronunciation": "Ab-dul-rah-man",
            "description": "name",
        },
    )
    assert updated["entry"]["pronunciation"] == "Ab-dul-rah-man"

    status, deleted = request(
        "DELETE", f"{server}/api/pronunciation/{entry_id}"
    )
    assert deleted["deleted"]["id"] == entry_id
    assert request("GET", server + "/api/pronunciation")[1]["entries"] == []


def test_duplicate_pronunciation_is_rejected(server):
    payload = {"word": "Test", "pronunciation": "T-est"}

    request("POST", server + "/api/pronunciation", payload)

    with pytest.raises(urllib.error.HTTPError) as error:
        request("POST", server + "/api/pronunciation", payload)

    assert error_payload(error.value)["code"] == "duplicate_word"
    assert error.value.code == 409


def test_pronunciation_validation(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        request(
            "POST",
            server + "/api/pronunciation",
            {"word": "   ", "pronunciation": "x"},
        )

    assert error_payload(error.value)["code"] == "invalid_word"


def test_preset_crud_and_validation(server):
    status, created = request(
        "POST",
        server + "/api/presets",
        {
            "name": "My Natural Voice",
            "engine": "kokoro",
            "voice": "af_heart",
            "speed": 1.0,
        },
    )

    assert status == 201
    preset = created["preset"]
    assert preset["engine"] == "kokoro"
    assert preset["voice"] == "af_heart"

    listing = request("GET", server + "/api/presets")[1]
    assert len(listing["presets"]) == 1

    updated = request(
        "PATCH",
        f"{server}/api/presets/{preset['id']}",
        {
            "name": "Fast Narration",
            "engine": "kokoro",
            "voice": "am_michael",
            "speed": 1.25,
        },
    )[1]
    assert updated["preset"]["name"] == "Fast Narration"
    assert updated["preset"]["speed"] == 1.25

    deleted = request("DELETE", f"{server}/api/presets/{preset['id']}")[1]
    assert deleted["deleted"]["id"] == preset["id"]


def test_preset_for_custom_engine_has_no_speed(server):
    created = request(
        "POST",
        server + "/api/presets",
        {"name": "Custom", "engine": "voiceforge", "speed": 1.5},
    )[1]

    assert created["preset"]["voice"] is None
    assert created["preset"]["speed"] is None


@pytest.mark.parametrize(
    "payload,code",
    [
        ({"name": "", "engine": "kokoro"}, "invalid_preset_name"),
        ({"name": "X", "engine": "nope"}, "invalid_engine"),
        ({"name": "X", "engine": "kokoro", "voice": "nope"}, "invalid_voice"),
        ({"name": "X", "engine": "kokoro", "speed": 5}, "invalid_speed"),
    ],
)
def test_invalid_presets_are_rejected(server, payload, code):
    with pytest.raises(urllib.error.HTTPError) as error:
        request("POST", server + "/api/presets", payload)

    assert error_payload(error.value)["code"] == code


def test_duplicate_preset_name_is_rejected(server):
    payload = {"name": "Same", "engine": "voiceforge"}

    request("POST", server + "/api/presets", payload)

    with pytest.raises(urllib.error.HTTPError) as error:
        request("POST", server + "/api/presets", payload)

    assert error_payload(error.value)["code"] == "duplicate_preset"


def test_library_mp3_export(server):
    result = request(
        "POST",
        server + "/api/generate",
        {"text": "Export this audio as em pee three.", "engine": "voiceforge"},
    )[1]

    record_id = result["library_id"]
    assert record_id

    with http_get(f"{server}/api/library/{record_id}/mp3?bitrate=192") as response:
        data = response.read()

    assert response.status == 200
    assert response.headers["Content-Type"] == "audio/mpeg"
    assert data[:3] == b"ID3" or (data[0] == 0xFF and (data[1] & 0xE0) == 0xE0)

    # WAV still works and is untouched.
    with http_get(f"{server}/api/library/{record_id}/audio") as response:
        assert response.read()[:4] == b"RIFF"


def test_library_mp3_invalid_bitrate(server):
    result = request(
        "POST",
        server + "/api/generate",
        {"text": "Bitrate check.", "engine": "voiceforge"},
    )[1]

    with pytest.raises(urllib.error.HTTPError) as error:
        http_get(
            f"{server}/api/library/{result['library_id']}/mp3?bitrate=96"
        )

    assert error_payload(error.value)["code"] == "invalid_bitrate"


def test_job_mp3_archive(server):
    from conftest import wait_for_job

    with http_post_json(
        server + "/api/jobs",
        {
            "type": "batch",
            "engine": "voiceforge",
            "items": [{"name": "One", "text": "Batch export item."}],
        },
    ) as response:
        job_id = json.loads(response.read())["job"]["id"]

    wait_for_job(server, job_id)

    with http_get(f"{server}/api/jobs/{job_id}/download?format=mp3") as response:
        archive = response.read()

    import zipfile

    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        names = bundle.namelist()
        manifest = json.loads(bundle.read("manifest.json"))

    assert any(name.endswith(".mp3") for name in names)
    assert manifest["format"] == "mp3"
    assert manifest["bitrate"] == 192


def test_pronunciation_applied_in_generation(server):
    request(
        "POST",
        server + "/api/pronunciation",
        {"word": "Abdulrahman", "pronunciation": "Ab-dool-rah-man"},
    )

    payload = request(
        "POST",
        server + "/api/generate",
        {
            "text": "Hello Abdulrahman.",
            "engine": "voiceforge",
            "pronunciation": True,
        },
    )[1]

    assert "Ab-dool-rah-man" in payload["text"]


def test_pronunciation_can_be_disabled(server):
    request(
        "POST",
        server + "/api/pronunciation",
        {"word": "Abdulrahman", "pronunciation": "Ab-dool-rah-man"},
    )

    payload = request(
        "POST",
        server + "/api/generate",
        {
            "text": "Hello Abdulrahman.",
            "engine": "voiceforge",
            "pronunciation": False,
        },
    )[1]

    assert "Abdulrahman" in payload["text"]


def test_dictionary_and_presets_survive_restart(tmp_path):
    library = Library(tmp_path / "lib.db", tmp_path / "audio")
    library.add_pronunciation("Kokoro", "Ko-ko-ro")
    library.create_preset("Saved", "kokoro", "af_heart", 1.0)

    reopened = Library(tmp_path / "lib.db", tmp_path / "audio")

    assert reopened.list_pronunciation()[0]["word"] == "Kokoro"
    assert reopened.list_presets()[0]["name"] == "Saved"


def test_migration_keeps_existing_audio_records(tmp_path):
    import sqlite3

    db = tmp_path / "old.db"
    connection = sqlite3.connect(db)
    connection.executescript(
        """
        CREATE TABLE audio_records (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, original_text TEXT NOT NULL,
            filename TEXT NOT NULL, source_filename TEXT NOT NULL,
            engine TEXT NOT NULL, voice TEXT, speed REAL,
            sample_rate INTEGER NOT NULL, duration_seconds REAL NOT NULL,
            file_size INTEGER NOT NULL, source_type TEXT NOT NULL,
            job_id TEXT, job_item_id TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        PRAGMA user_version = 1;
        """
    )
    connection.execute(
        "INSERT INTO audio_records VALUES "
        "('old','Old','Hello.','old.wav','old.wav','voiceforge',NULL,NULL,"
        "22050,1.0,100,'studio',NULL,NULL,'2024-01-01','2024-01-01')"
    )
    connection.commit()
    connection.close()

    library = Library(db, tmp_path / "audio")

    assert library.get("old")["name"] == "Old"

    with sqlite3.connect(db) as check:
        version = check.execute("PRAGMA user_version").fetchone()[0]
        tables = {row[0] for row in check.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}

    assert version == 2
    assert {"pronunciation_entries", "voice_presets"} <= tables
