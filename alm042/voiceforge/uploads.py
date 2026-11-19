"""
Validation for uploaded ``.txt`` files.

The browser sends the raw bytes base64-encoded, so the server - not the
client - decides whether the file is acceptable: extension, size, encoding
and filename safety are all checked here.

Uploaded names are only used as display labels and as safe archive names;
they are never joined to a filesystem path.
"""

from __future__ import annotations

import base64
import binascii
import re
from pathlib import Path

from voiceforge.app_core import AppError

#: Limits (also documented in the README).
MAX_UPLOAD_FILES = 20
MAX_FILE_BYTES = 1024 * 1024          # 1 MB per file
MAX_TOTAL_BYTES = 5 * 1024 * 1024     # 5 MB per request
MAX_NAME_LENGTH = 80

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]+")
_WHITESPACE = re.compile(r"\s+")


def sanitize_filename(name: object, fallback: str = "upload") -> str:
    """
    Turn an uploaded name into a safe display/archive name.

    Path components are dropped, unsafe characters replaced and the result is
    always a plain ``.txt`` filename.
    """

    if not isinstance(name, str) or not name.strip():
        return f"{fallback}.txt"

    # Never trust a path: keep only the last component.
    base = Path(name.replace("\\", "/")).name.strip()

    base = _SAFE_NAME.sub("_", base)
    base = _WHITESPACE.sub("_", base).strip("._ ")

    if not base:
        return f"{fallback}.txt"

    if not base.lower().endswith(".txt"):
        base = f"{base}.txt"

    if len(base) > MAX_NAME_LENGTH:
        base = base[: MAX_NAME_LENGTH - 4].rstrip("._ ") + ".txt"

    return base


def _decode_payload(entry: dict, index: int) -> tuple[str, bytes]:
    name = entry.get("name")
    data = entry.get("data_base64")

    if not isinstance(name, str):
        raise AppError(
            f"File {index + 1} is missing a name.",
            code="invalid_upload",
        )

    if not name.lower().endswith(".txt"):
        raise AppError(
            f"'{sanitize_filename(name)}' is not a .txt file. "
            "Only plain text files are accepted.",
            code="unsupported_file",
        )

    if not isinstance(data, str) or not data:
        raise AppError(
            f"File {index + 1} is empty.",
            code="invalid_upload",
        )

    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as error:
        raise AppError(
            f"File {index + 1} could not be decoded.",
            code="invalid_upload",
        ) from error

    if len(raw) > MAX_FILE_BYTES:
        raise AppError(
            f"'{sanitize_filename(name)}' is larger than "
            f"{MAX_FILE_BYTES // 1024} KB.",
            code="upload_too_large",
            status=413,
        )

    return name, raw


def validate_uploads(payload: object) -> list[dict]:
    """
    Validate a ``/api/uploads`` payload and return decoded text files.

    Raises :class:`AppError` with a specific code for each rejection.
    """

    if not isinstance(payload, dict):
        raise AppError("Upload payload must be a JSON object.", code="invalid_upload")

    files = payload.get("files")

    if not isinstance(files, list) or not files:
        raise AppError("Select at least one .txt file.", code="invalid_upload")

    if len(files) > MAX_UPLOAD_FILES:
        raise AppError(
            f"Too many files: {len(files)}. The limit is {MAX_UPLOAD_FILES}.",
            code="upload_too_large",
            status=413,
        )

    results: list[dict] = []
    total_bytes = 0

    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise AppError(
                f"File {index + 1} is not a valid entry.",
                code="invalid_upload",
            )

        raw_name, raw = _decode_payload(entry, index)

        total_bytes += len(raw)

        if total_bytes > MAX_TOTAL_BYTES:
            raise AppError(
                "The upload is too large in total. "
                f"The limit is {MAX_TOTAL_BYTES // (1024 * 1024)} MB.",
                code="upload_too_large",
                status=413,
            )

        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise AppError(
                f"'{sanitize_filename(raw_name)}' is not valid UTF-8 text.",
                code="invalid_encoding",
            ) from error

        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

        if not text:
            raise AppError(
                f"'{sanitize_filename(raw_name)}' contains no text.",
                code="empty_file",
            )

        results.append(
            {
                "name": sanitize_filename(raw_name),
                "text": text,
                "characters": len(text),
            }
        )

    return results
