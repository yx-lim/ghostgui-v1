"""Lifecycle-managed, non-blocking file selection for GUI workflows."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QCoreApplication, QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import QFileDialog, QWidget


class SynchronousFileSelectionStage(QObject):
    """Own one parented, window-modal ``QFileDialog`` at a time.

    The historical class name is retained for callers, but selection no longer
    launches a detached helper process.  ``QFileDialog.open()`` keeps the main
    event loop responsive and allows Qt to use the platform's native picker.
    """

    active_changed = Signal(bool)

    def __init__(
        self,
        parent=None,
        *,
        context_provider: Callable[[], object] | None = None,
    ):
        super().__init__(parent)
        self._dialog = None
        self._selected_callback = None
        self._failed_callback = None
        self._cancelled_callback = None
        self._context_provider = context_provider
        self._selection_context = None
        self._callback_pending = False
        self._request_id = 0
        self._active = False
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.cancel)

    def is_active(self):
        return self._dialog is not None or self._callback_pending

    def select_file(
        self,
        *,
        mode,
        title,
        directory,
        name_filter,
        selected,
        failed=None,
        cancelled=None,
        filename=None,
    ):
        if self.is_active():
            return False

        parent = self.parent()
        dialog_parent = parent if isinstance(parent, QWidget) else None
        dialog = QFileDialog(dialog_parent)
        dialog.setWindowTitle(str(title))
        dialog.setDirectory(str(directory))
        dialog.setNameFilter(str(name_filter))
        dialog.setWindowModality(Qt.WindowModality.WindowModal)

        if mode == "open":
            dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
            dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        elif mode == "save":
            dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
            dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        elif mode == "directory":
            dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
            dialog.setFileMode(QFileDialog.FileMode.Directory)
            dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        else:
            dialog.deleteLater()
            message = f"Unsupported file-selector mode: {mode}"
            if failed is not None:
                QTimer.singleShot(0, lambda: failed(message))
            return False

        if filename:
            dialog.selectFile(str(filename))

        self._request_id += 1
        request_id = self._request_id
        self._dialog = dialog
        self._selected_callback = selected
        self._failed_callback = failed
        self._cancelled_callback = cancelled
        self._selection_context = self._current_context()
        dialog.accepted.connect(
            lambda candidate=dialog, token=request_id: self._on_accepted(
                candidate, token
            )
        )
        dialog.rejected.connect(
            lambda candidate=dialog, token=request_id: self._on_rejected(
                candidate, token
            )
        )
        self._update_active_signal()
        dialog.open()
        return True

    def cancel(self):
        dialog = self._dialog
        self._request_id += 1
        self._dialog = None
        self._clear_callbacks()
        self._callback_pending = False
        if dialog is not None:
            dialog.close()
            dialog.deleteLater()
        self._update_active_signal()

    def _on_accepted(self, dialog, request_id):
        if dialog is not self._dialog or request_id != self._request_id:
            return
        paths = dialog.selectedFiles()
        selected_path = str(paths[0]) if paths else ""
        selected_callback = self._selected_callback
        cancelled_callback = self._cancelled_callback
        context_is_current = self._context_is_current()
        self._dialog = None
        self._clear_callbacks()
        dialog.deleteLater()

        if selected_path and selected_callback is not None and context_is_current:
            self._callback_pending = True
            QTimer.singleShot(
                0,
                lambda: self._deliver_selection(
                    request_id,
                    selected_callback,
                    selected_path,
                ),
            )
            return

        self._update_active_signal()
        if cancelled_callback is not None:
            cancelled_callback()

    def _on_rejected(self, dialog, request_id):
        if dialog is not self._dialog or request_id != self._request_id:
            return
        cancelled_callback = self._cancelled_callback
        self._dialog = None
        self._clear_callbacks()
        dialog.deleteLater()
        self._update_active_signal()
        if cancelled_callback is not None:
            cancelled_callback()

    def _deliver_selection(self, request_id, callback, path):
        if not self._callback_pending or request_id != self._request_id:
            return
        try:
            callback(path)
        finally:
            self._callback_pending = False
            self._update_active_signal()

    def _current_context(self):
        if self._context_provider is None:
            return None
        return self._context_provider()

    def _context_is_current(self):
        if self._context_provider is None:
            return True
        try:
            return self._selection_context == self._context_provider()
        except RuntimeError:
            # A deleted Qt owner is a stale selection context.
            return False

    def _clear_callbacks(self):
        self._selected_callback = None
        self._failed_callback = None
        self._cancelled_callback = None
        self._selection_context = None

    def _update_active_signal(self):
        active = self.is_active()
        if active == self._active:
            return
        self._active = active
        self.active_changed.emit(active)
