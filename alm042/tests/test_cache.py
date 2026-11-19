"""Tests for the MP3 cache cleanup policy and audit fixes."""

from __future__ import annotations

import os
import time

from voiceforge.cache import cleanup_mp3_cache


def make_file(directory, name, size=1024, age_days=0):
    path = directory / name
    path.write_bytes(b"x" * size)
    stamp = time.time() - age_days * 86400
    os.utime(path, (stamp, stamp))
    return path


def test_stale_mp3_files_are_removed_and_fresh_kept(tmp_path):
    cache = tmp_path / "mp3"
    cache.mkdir()

    old = make_file(cache, "old.mp3", age_days=30)
    fresh = make_file(cache, "fresh.mp3", age_days=1)

    summary = cleanup_mp3_cache(cache, max_age_days=7)

    assert not old.exists()
    assert fresh.exists()
    assert "old.mp3" in summary["removed"]
    assert summary["kept"] == 1


def test_wav_files_are_never_deleted(tmp_path):
    cache = tmp_path / "mp3"
    cache.mkdir()

    wav = make_file(cache, "important.wav", age_days=365)
    mp3 = make_file(cache, "old.mp3", age_days=30)

    cleanup_mp3_cache(cache, max_age_days=7)

    assert wav.exists()
    assert not mp3.exists()


def test_non_mp3_files_are_ignored(tmp_path):
    cache = tmp_path / "mp3"
    cache.mkdir()

    keep = make_file(cache, "notes.txt", age_days=365)

    cleanup_mp3_cache(cache, max_age_days=7)

    assert keep.exists()


def test_size_limit_removes_oldest_first(tmp_path):
    cache = tmp_path / "mp3"
    cache.mkdir()

    oldest = make_file(cache, "a.mp3", size=1024, age_days=3)
    middle = make_file(cache, "b.mp3", size=1024, age_days=2)
    newest = make_file(cache, "c.mp3", size=1024, age_days=1)

    # ~2 KB budget: one file must go, the oldest first.
    cleanup_mp3_cache(cache, max_age_days=30, max_total_mb=0.002)

    assert not oldest.exists()
    assert middle.exists()
    assert newest.exists()


def test_missing_cache_directory_is_safe(tmp_path):
    summary = cleanup_mp3_cache(tmp_path / "missing")

    assert summary["removed"] == []
    assert summary["kept"] == 0


def test_request_body_fits_a_ten_megabyte_document():
    from voiceforge.web_app import MAX_BODY_BYTES

    # Base64 inflates 10 MB to ~13.34 MB, plus JSON overhead.
    assert MAX_BODY_BYTES >= 14 * 1024 * 1024


def test_docx_reports_no_page_count():
    import io

    import docx

    from voiceforge.document_reader import extract_document

    document = docx.Document()
    document.add_paragraph("Hello from Word.")
    buffer = io.BytesIO()
    document.save(buffer)

    result = extract_document("a.docx", buffer.getvalue())

    assert result["pages"] is None
