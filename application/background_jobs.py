"""Serialized background work with GUI-thread result delivery."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Queue
from threading import Lock, Thread

from PySide6.QtCore import QCoreApplication, QObject, Qt, Signal


@dataclass(frozen=True)
class _BackgroundJob:
    identifier: int
    name: str
    work: object


class SerializedBackgroundJobs(QObject):
    """Run jobs one at a time and deliver their results on the Qt GUI thread."""

    busy_changed = Signal(bool)
    _result_ready = Signal(int, object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._jobs = Queue()
        self._callbacks = {}
        self._callback_lock = Lock()
        self._next_identifier = 1
        self._pending_count = 0
        self._closing = False
        self._result_ready.connect(
            self._deliver_result,
            Qt.ConnectionType.QueuedConnection,
        )
        self._thread = Thread(
            target=self._processing_loop,
            name="ghostgui-background-jobs",
            daemon=True,
        )
        self._thread.start()
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

    def is_busy(self):
        return self._pending_count > 0

    def submit(self, name, work, succeeded, failed):
        if self._closing:
            return False
        identifier = self._next_identifier
        self._next_identifier += 1
        with self._callback_lock:
            self._callbacks[identifier] = (succeeded, failed)
        was_idle = self._pending_count == 0
        self._pending_count += 1
        self._jobs.put(_BackgroundJob(identifier, str(name), work))
        if was_idle:
            self.busy_changed.emit(True)
        return True

    def clear_pending(self):
        removed_identifiers = []
        while True:
            try:
                job = self._jobs.get_nowait()
            except Empty:
                break
            if job is None:
                self._jobs.put(None)
                break
            removed_identifiers.append(job.identifier)
            self._jobs.task_done()

        if not removed_identifiers:
            return
        with self._callback_lock:
            for identifier in removed_identifiers:
                self._callbacks.pop(identifier, None)
        self._pending_count = max(
            0, self._pending_count - len(removed_identifiers)
        )
        if self._pending_count == 0:
            self.busy_changed.emit(False)

    def shutdown(self):
        if self._closing:
            return
        self._closing = True
        self.clear_pending()
        with self._callback_lock:
            self._callbacks.clear()
        self._pending_count = 0
        self._jobs.put(None)
        self._thread.join(timeout=0.25)
        self.busy_changed.emit(False)

    def _processing_loop(self):
        while True:
            job = self._jobs.get()
            if job is None:
                self._jobs.task_done()
                return
            result = None
            error = None
            try:
                result = job.work()
            except Exception as exc:
                error = exc
            self._jobs.task_done()
            if self._closing:
                continue
            try:
                self._result_ready.emit(job.identifier, result, error)
            except RuntimeError:
                return

    def _deliver_result(self, identifier, result, error):
        with self._callback_lock:
            callbacks = self._callbacks.pop(identifier, None)
        if callbacks is None:
            return

        self._pending_count = max(0, self._pending_count - 1)
        if self._pending_count == 0:
            self.busy_changed.emit(False)
        succeeded, failed = callbacks
        if error is None:
            succeeded(result)
        else:
            failed(error)
