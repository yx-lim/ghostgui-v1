"""Asynchronous model loading and per-model editor-session caching."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, QThread, Signal

from core.models.adapter import MuJoCoRobotAdapter
from .project import ProjectDocument


@dataclass
class RobotModelSession:
    adapter: object
    backend: object
    reference: object
    viewer_3d: object
    viewer_2d_skeleton: object
    project: ProjectDocument
    active_index: int = -1

    @property
    def trajectory(self):
        return self.project.target_trajectory

    @trajectory.setter
    def trajectory(self, value):
        self.project.target_trajectory = value


class ModelLoadThread(QThread):
    loaded = Signal(str, object)
    failed = Signal(str, str)

    def __init__(self, model_key, parent=None):
        super().__init__(parent)
        self.model_key = model_key

    def run(self):
        try:
            adapter = MuJoCoRobotAdapter(self.model_key)
        except Exception as exc:
            self.failed.emit(self.model_key, str(exc))
            return
        self.loaded.emit(self.model_key, adapter)


class ModelSessionManager(QObject):
    """Own model-load threads and cache completed model-specific sessions."""

    loaded = Signal(str, object)
    failed = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sessions = {}
        self.loaders = {}

    def register(self, model_key, session):
        self.sessions[model_key] = session

    def get(self, model_key):
        return self.sessions.get(model_key)

    def is_loading(self, model_key):
        return model_key in self.loaders

    def load(self, model_key):
        if self.is_loading(model_key):
            return
        loader = ModelLoadThread(model_key, self)
        loader.loaded.connect(self._loaded)
        loader.failed.connect(self._failed)
        loader.finished.connect(loader.deleteLater)
        self.loaders[model_key] = loader
        loader.start()

    def _loaded(self, model_key, adapter):
        self.loaders.pop(model_key, None)
        self.loaded.emit(model_key, adapter)

    def _failed(self, model_key, error):
        self.loaders.pop(model_key, None)
        self.failed.emit(model_key, error)
