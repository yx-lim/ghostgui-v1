"""Sanitise untrusted strings (project, model, joint, file names) before they reach a model."""

from __future__ import annotations

from application.ai.secrets import scrub_secrets

MAX_UNTRUSTED = 64


def untrusted(value, limit: int = MAX_UNTRUSTED):
    """Printable, secret-scrubbed, length-capped text; None passes through."""
    if value is None:
        return None
    text = "".join(character for character in scrub_secrets(str(value)) if character.isprintable())
    return text if len(text) <= limit else text[: limit - 1] + "…"
