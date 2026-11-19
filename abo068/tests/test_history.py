import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.src import history


class HistoryPersistenceTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.data_folder = Path(temporary_directory.name)
        self.history_file = self.data_folder / "history.json"
        for name, value in (
            ("DATA_FOLDER", self.data_folder),
            ("HISTORY_FILE", self.history_file),
        ):
            patcher = patch.object(history, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_corrupt_history_is_reported_without_replacing_file(self):
        corrupt_content = '{"unfinished":'
        self.history_file.write_text(corrupt_content, encoding="utf-8")

        with self.assertLogs("backend.src.history", level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "left unchanged"):
                history.load_history()

        self.assertEqual(self.history_file.read_text(encoding="utf-8"), corrupt_content)

    def test_non_list_history_is_rejected(self):
        self.history_file.write_text('{"record": {}}', encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "JSON list"):
            history.load_history()

    def test_save_replaces_history_with_complete_json(self):
        records = [{"source": "AI Playground", "text": "Great service"}]

        history.save_history(records)

        self.assertEqual(json.loads(self.history_file.read_text(encoding="utf-8")), records)
        self.assertEqual(list(self.data_folder.glob("*.tmp")), [])

    def test_failed_save_keeps_existing_history(self):
        original_content = '[{"source":"ReviewAI"}]'
        self.history_file.write_text(original_content, encoding="utf-8")

        with self.assertRaises(TypeError):
            history.save_history([object()])

        self.assertEqual(self.history_file.read_text(encoding="utf-8"), original_content)
        self.assertEqual(list(self.data_folder.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
