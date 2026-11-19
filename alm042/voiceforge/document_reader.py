"""
Extract readable text from uploaded PDF and DOCX documents.

Only text is extracted; macros, embedded scripts and external content are
never executed. Scanned PDFs contain images rather than text and are not
supported (no OCR in this phase).
"""

from __future__ import annotations

import io
import re
from pathlib import Path

from voiceforge.app_core import AppError

#: Limits for one uploaded document.
MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
SUPPORTED_EXTENSIONS = (".pdf", ".docx")

MAX_FILENAME_LENGTH = 80

_WHITESPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def _clean_name(name: object) -> str:
    if not isinstance(name, str) or not name.strip():
        raise AppError(
            "The document has no filename.",
            code="invalid_document",
        )

    # Only the last path component is ever used, and only for display.
    base = Path(name.replace("\\", "/")).name.strip()
    base = "".join(
        character for character in base if ord(character) >= 32
    )

    if not base:
        raise AppError(
            "The document has no filename.",
            code="invalid_document",
        )

    if len(base) > MAX_FILENAME_LENGTH:
        stem, suffix = Path(base).stem, Path(base).suffix
        base = stem[: MAX_FILENAME_LENGTH - len(suffix)] + suffix

    return base


def _normalise(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def _extract_pdf(data: bytes) -> tuple[str, int]:
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
    except Exception as error:  # noqa: BLE001 - wrapped for the client
        raise AppError(
            "This PDF could not be read. It may be corrupted or invalid.",
            code="invalid_document",
        ) from error

    if reader.is_encrypted:
        try:
            # Many PDFs are "encrypted" with an empty owner password.
            if not reader.decrypt(""):
                raise AppError(
                    "This PDF is password-protected and cannot be read.",
                    code="encrypted_document",
                )
        except AppError:
            raise
        except Exception as error:  # noqa: BLE001
            raise AppError(
                "This PDF is password-protected and cannot be read.",
                code="encrypted_document",
            ) from error

    try:
        pages = [
            page.extract_text() or ""
            for page in reader.pages
        ]
    except Exception as error:  # noqa: BLE001
        raise AppError(
            "The text of this PDF could not be extracted.",
            code="invalid_document",
        ) from error

    text = _normalise("\n\n".join(pages))

    if not text:
        raise AppError(
            "This PDF contains no extractable text. Scanned PDFs are images "
            "and need OCR, which is not supported in this version.",
            code="no_extractable_text",
        )

    return text, len(pages)


def _extract_docx(data: bytes) -> tuple[str, int | None]:
    try:
        import docx
        from docx.oxml.ns import qn
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        document = docx.Document(io.BytesIO(data))
    except Exception as error:  # noqa: BLE001 - wrapped for the client
        raise AppError(
            "This DOCX file could not be read. It may be corrupted or not a "
            "Word document.",
            code="invalid_document",
        ) from error

    blocks: list[str] = []

    try:
        # Walk the body in document order so paragraphs and tables keep
        # their real sequence.
        for child in document.element.body.iterchildren():
            if child.tag == qn("w:p"):
                text = Paragraph(child, document).text.strip()

                if text:
                    blocks.append(text)
            elif child.tag == qn("w:tbl"):
                table = Table(child, document)

                for row in table.rows:
                    cells = [
                        cell.text.strip().replace("\n", " ")
                        for cell in row.cells
                    ]

                    if any(cells):
                        blocks.append(" | ".join(cells))
    except Exception as error:  # noqa: BLE001
        raise AppError(
            "The text of this DOCX file could not be extracted.",
            code="invalid_document",
        ) from error

    text = _normalise("\n".join(blocks))

    if not text:
        raise AppError(
            "This document contains no text to read.",
            code="empty_document",
        )

    # DOCX files have no fixed page count, so none is reported.
    return text, None


def extract_document(name: object, data: bytes) -> dict:
    """
    Validate and extract an uploaded PDF/DOCX document.

    Returns metadata plus the extracted text; nothing is written to disk.
    """

    filename = _clean_name(name)
    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise AppError(
            f"'{filename}' is not supported. Upload a PDF or DOCX file.",
            code="unsupported_document",
        )

    if not isinstance(data, (bytes, bytearray)) or not data:
        raise AppError(
            "The document is empty.",
            code="empty_document",
        )

    if len(data) > MAX_DOCUMENT_BYTES:
        raise AppError(
            f"The document is larger than {MAX_DOCUMENT_BYTES // (1024 * 1024)} MB.",
            code="document_too_large",
            status=413,
        )

    if suffix == ".pdf":
        text, pages = _extract_pdf(bytes(data))
        document_format = "pdf"
    else:
        text, pages = _extract_docx(bytes(data))
        document_format = "docx"

    return {
        "filename": filename,
        "format": document_format,
        "text": text,
        "characters": len(text),
        "pages": pages,
    }
