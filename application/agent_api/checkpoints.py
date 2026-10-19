"""A bounded checkpoint ring owned by one active candidate session."""

from collections import OrderedDict
from uuid import uuid4


class CheckpointRing:
    def __init__(self, capacity=32):
        if capacity < 1:
            raise ValueError("checkpoint capacity must be positive")
        self.capacity = capacity
        self.entries = OrderedDict()

    def add(self, checkpoint, generated):
        key = uuid4().hex
        self.entries[key] = (checkpoint, generated)
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return key

    def get(self, key=None):
        if not self.entries:
            raise ValueError("no candidate checkpoint is available")
        key = key or next(reversed(self.entries))
        if key not in self.entries:
            raise ValueError("checkpoint is expired or belongs to another session")
        return self.entries[key]

    def clear(self):
        self.entries.clear()

    def discard_from(self, key=None):
        """Undo consumes the chosen boundary and discards the newer branch."""
        self.get(key)
        key = key or next(reversed(self.entries))
        keys = list(self.entries)
        for item in keys[keys.index(key):]:
            del self.entries[item]
