import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path


logger = logging.getLogger(__name__)
PROJECT_FOLDER = Path(__file__).resolve().parents[2]
DATA_FOLDER = PROJECT_FOLDER / "data"
HISTORY_FILE = DATA_FOLDER / "history.json"


def load_history():
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    if not HISTORY_FILE.exists():
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            records = json.load(file)
    except (json.JSONDecodeError, OSError, UnicodeError) as error:
        logger.exception("Could not read analysis history at %s.", HISTORY_FILE)
        raise RuntimeError(
            f"Could not read analysis history at {HISTORY_FILE}; the file was left unchanged."
        ) from error

    if not isinstance(records, list):
        raise ValueError(f"Analysis history at {HISTORY_FILE} must contain a JSON list.")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError(f"Analysis history at {HISTORY_FILE} contains an invalid record.")
    return records


def save_history(records):
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=DATA_FOLDER,
            prefix=f".{HISTORY_FILE.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(records, file, indent=4, ensure_ascii=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, HISTORY_FILE)
    except OSError:
        logger.exception("Could not save analysis history at %s.", HISTORY_FILE)
        raise
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def add_history_record(source, text, result, metadata=None):
    records = load_history()
    record = {
        "source": source,
        "text": text,
        "sentiment": result["sentiment"],
        "confidence": result["confidence"],
        "scores": result["scores"],
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if metadata:
        record.update(metadata)
    records.insert(0, record)
    save_history(records)
    return record


def clear_history():
    save_history([])
    return []


def get_history():
    return load_history()
