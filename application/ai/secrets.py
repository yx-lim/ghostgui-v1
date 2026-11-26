"""Secret hygiene for bring-your-own API keys."""

from __future__ import annotations

import logging
import re

KEY_PATTERN = re.compile(r"sk-ant-[A-Za-z0-9_\-]{6,}|AIza[0-9A-Za-z_\-]{20,}")
ADMIN_KEY_PREFIX = "sk-ant-admin"
_REDACTED = "[redacted]"


class SecretStr:
    """Holds a key without exposing it through repr/str/format."""

    __slots__ = ("_value",)

    def __init__(self, value: str):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("secret must be a non-empty string")
        self._value = value.strip()

    def reveal(self) -> str:
        return self._value

    @property
    def is_admin_key(self) -> bool:
        return self._value.startswith(ADMIN_KEY_PREFIX)

    def __repr__(self) -> str:
        return "SecretStr('[redacted]')"

    __str__ = __repr__

    def __format__(self, spec) -> str:
        return repr(self)

    def __eq__(self, other) -> bool:
        return isinstance(other, SecretStr) and other._value == self._value

    def __hash__(self) -> int:
        return hash(("SecretStr", self._value))

    def __reduce__(self):
        raise TypeError("SecretStr cannot be pickled")


def scrub_secrets(text: str) -> str:
    return KEY_PATTERN.sub(_REDACTED, str(text))


def contains_secret(text: str) -> bool:
    return bool(KEY_PATTERN.search(str(text)))


class SecretScrubFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        if contains_secret(message):
            record.msg, record.args = scrub_secrets(message), ()
        return True


_INSTALLED = False


def install_log_scrubbing(names=("anthropic", "httpx2", "httpx", "ghostgui")) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    for name in names:
        logging.getLogger(name).addFilter(SecretScrubFilter())
    _INSTALLED = True


def child_environment(environment: dict) -> dict:
    """Copy of an environment with provider keys removed, for child processes."""
    blocked = {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GOOGLE_API_KEY", "GEMINI_API_KEY"}
    return {key: value for key, value in environment.items() if key not in blocked}
