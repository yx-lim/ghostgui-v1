"""Shared immutable robot-model resources keyed by their source path."""

from __future__ import annotations

from pathlib import Path


class ModelResourcePool:
    """Own one adapter/MjModel for each canonical robot model source."""

    def __init__(self):
        self._adapters = {}

    @staticmethod
    def canonical_key(model_info=None, model_path=None):
        path = model_path
        if path is None and model_info is not None:
            path = getattr(model_info, "model_path", None)
        if path is None:
            return None
        return str(Path(path).expanduser().resolve())

    def register(self, adapter):
        if adapter is None:
            return None
        key = self.canonical_key(model_path=getattr(adapter, "model_path", None))
        if key is None:
            return adapter
        existing = self._adapters.get(key)
        if existing is not None:
            return existing
        self._adapters[key] = adapter
        return adapter

    def get(self, model_info=None, model_path=None):
        key = self.canonical_key(model_info=model_info, model_path=model_path)
        return None if key is None else self._adapters.get(key)

    def discard(self, model_info=None, model_path=None):
        key = self.canonical_key(model_info=model_info, model_path=model_path)
        return None if key is None else self._adapters.pop(key, None)

    def __len__(self):
        return len(self._adapters)
