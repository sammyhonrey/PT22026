# Load a PDF or TXT file and return its text page by page.

from pathlib import Path

from pypdf import PdfReader


def load_document(file_path):
    # Return a list of (page_number, text) pairs for the given file
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix == ".txt":
        return _load_txt(path)
    raise ValueError("Unsupported file type. Please upload a PDF or TXT file.")


def _load_pdf(path):
    # Extract text from each PDF page, skipping pages with no text
    reader = PdfReader(str(path))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((number, text))
    if not pages:
        raise ValueError("No readable text found in this PDF (it may be scanned).")
    return pages


def _load_txt(path):
    # Read a plain text file and treat it as a single page
    # errors="ignore" stops the app from crashing on unknown characters
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        raise ValueError("The text file is empty.")
    return [(1, text)]
