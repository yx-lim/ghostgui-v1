"""Command-bar prompt history: Up/Down recall, persisted at 0600 in the config directory.

Qt-free so it can be tested directly. Prompts that look like they contain a secret are
never stored, and stored text is scrubbed again before it is written.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from application.ai.secrets import contains_secret, scrub_secrets

MAX_ENTRIES = 200
MAX_CHARS = 2000


def default_history_path() -> Path:
    from application.project_manager import ghostgui_config_dir

    return ghostgui_config_dir() / "agent" / "prompt_history.json"


class PromptHistory:
    def __init__(self, path: Path | None = None, *, enabled: bool = True):
        self.path = Path(path) if path is not None else default_history_path()
        self.enabled = enabled
        self.entries: list[str] = self._load() if enabled else []
        self.reset_cursor()

    # ----------------------------------------------------------- persistence
    def _load(self) -> list[str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        items = data.get("prompts", []) if isinstance(data, dict) else []
        return [str(item)[:MAX_CHARS] for item in items if isinstance(item, str) and item.strip()][-MAX_ENTRIES:]

    def _save(self):
        if not self.enabled:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            body = json.dumps({"schema_version": 1, "prompts": self.entries}, ensure_ascii=False)
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(body)
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
        except OSError:
            pass  # history is a convenience; never fail a prompt over it

    def set_enabled(self, enabled: bool):
        self.enabled = bool(enabled)
        if self.enabled:
            self.entries = self._load() or self.entries
            self._save()
        self.reset_cursor()

    def clear(self):
        self.entries = []
        self.reset_cursor()
        try:
            self.path.unlink()
        except OSError:
            pass

    # ---------------------------------------------------------------- recall
    def add(self, prompt: str):
        text = str(prompt).strip()[:MAX_CHARS]
        if not text or contains_secret(text):
            self.reset_cursor()
            return
        text = scrub_secrets(text)
        if self.entries and self.entries[-1] == text:
            self.reset_cursor()
            return
        self.entries.append(text)
        del self.entries[:-MAX_ENTRIES]
        self._save()
        self.reset_cursor()

    def reset_cursor(self):
        self._index = len(self.entries)
        self._draft = ""

    def previous(self, current: str) -> str | None:
        """Older entry (Up). Remembers the unsent draft on first use."""
        if not self.entries or self._index == 0:
            return None
        if self._index == len(self.entries):
            self._draft = current
        self._index -= 1
        return self.entries[self._index]

    def next(self) -> str | None:
        """Newer entry (Down); returns the draft after the newest entry."""
        if self._index >= len(self.entries):
            return None
        self._index += 1
        return self._draft if self._index == len(self.entries) else self.entries[self._index]
