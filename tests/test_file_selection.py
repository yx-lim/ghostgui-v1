import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QWidget

from gui.file_selection import SynchronousFileSelectionStage


class FileSelectionStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.parent = QWidget()
        self.context = {"value": "first"}
        self.stage = SynchronousFileSelectionStage(
            self.parent,
            context_provider=lambda: self.context["value"],
        )

    def tearDown(self):
        self.stage.cancel()
        self.parent.close()
        self.app.processEvents()

    def open_dialog(self, path, **callbacks):
        opened = self.stage.select_file(
            mode="open",
            title="Choose a file",
            directory=path.parent,
            name_filter="Text files (*.txt)",
            selected=callbacks.get("selected", lambda _path: None),
            cancelled=callbacks.get("cancelled"),
        )
        self.assertTrue(opened)
        return self.stage._dialog

    def flush_events(self):
        # A zero-duration timer scheduled from a dialog signal may run on the
        # following dispatcher turn rather than the signal's current turn.
        for _ in range(3):
            self.app.processEvents()

    def test_dialog_is_parented_nonblocking_and_window_modal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.txt"
            path.write_text("example", encoding="utf-8")
            dialog = self.open_dialog(path)

            self.assertIs(dialog.parentWidget(), self.parent)
            self.assertEqual(
                dialog.windowModality(),
                Qt.WindowModality.WindowModal,
            )
            self.assertEqual(
                dialog.fileMode(),
                QFileDialog.FileMode.ExistingFile,
            )
            self.assertTrue(self.stage.is_active())

    def test_selection_is_delivered_once_after_accept(self):
        selected = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.txt"
            path.write_text("example", encoding="utf-8")
            dialog = self.open_dialog(path, selected=selected.append)
            dialog.selectFile(str(path))
            dialog.accept()
            self.flush_events()

            self.assertEqual(selected, [str(path)])
            self.assertFalse(self.stage.is_active())

    def test_changed_context_drops_stale_selection(self):
        selected = []
        cancelled = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.txt"
            path.write_text("example", encoding="utf-8")
            dialog = self.open_dialog(
                path,
                selected=selected.append,
                cancelled=lambda: cancelled.append(True),
            )
            self.context["value"] = "second"
            dialog.selectFile(str(path))
            dialog.accept()
            self.flush_events()

            self.assertEqual(selected, [])
            self.assertEqual(cancelled, [True])
            self.assertFalse(self.stage.is_active())

    def test_cancel_invalidates_late_dialog_signals(self):
        selected = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.txt"
            path.write_text("example", encoding="utf-8")
            dialog = self.open_dialog(path, selected=selected.append)
            self.stage.cancel()
            dialog.selectFile(str(path))
            dialog.accept()
            self.flush_events()

            self.assertEqual(selected, [])
            self.assertFalse(self.stage.is_active())


if __name__ == "__main__":
    unittest.main()
