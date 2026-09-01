"""Credential sources for optional AI providers.

Secrets are read on demand and passed directly to provider SDK constructors.
Optional persistence uses the OS credential store, never project or preference
files.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping, Protocol


GHOSTGUI_KEYRING_SERVICE = "GhostGUI AI"


class CredentialSource(Protocol):
    """Read a provider secret without exposing a persistence implementation."""

    def get_secret(self, provider_name: str) -> str | None:
        ...


class CredentialStore(CredentialSource, Protocol):
    """Read, write, and explicitly remove provider secrets."""

    def set_secret(self, provider_name: str, secret: str) -> None:
        ...

    def delete_secret(self, provider_name: str) -> bool:
        ...


class CredentialStorageError(RuntimeError):
    """The requested secure credential-store operation could not complete."""


@dataclass(frozen=True)
class EnvironmentCredentialSource:
    """Read Gemini's official environment variables without mutating them."""

    environ: Mapping[str, str] | None = None

    def get_secret(self, provider_name: str) -> str | None:
        if provider_name != "gemini":
            return None
        values = os.environ if self.environ is None else self.environ
        # The Gemini SDK gives GOOGLE_API_KEY precedence when both are present.
        return values.get("GOOGLE_API_KEY") or values.get("GEMINI_API_KEY") or None


@dataclass(frozen=True)
class SystemKeyringCredentialSource:
    """Read provider keys from the OS credential store when keyring is installed."""

    service_name: str = GHOSTGUI_KEYRING_SERVICE

    def get_secret(self, provider_name: str) -> str | None:
        try:
            import keyring
        except ImportError:
            return None
        try:
            return keyring.get_password(self.service_name, provider_name)
        except keyring.errors.KeyringError:
            # A missing/locked OS backend must not prevent the environment
            # fallback or normal non-AI GhostGUI startup.
            return None


@dataclass(frozen=True)
class SystemKeyringCredentialStore(SystemKeyringCredentialSource):
    """Persist provider keys only in the operating system credential store."""

    def set_secret(self, provider_name: str, secret: str) -> None:
        secret = str(secret)
        if not provider_name.strip() or not secret:
            raise ValueError("provider name and secret must not be empty")
        keyring = self._required_keyring()
        try:
            keyring.set_password(self.service_name, provider_name, secret)
        except keyring.errors.KeyringError as error:
            raise CredentialStorageError(
                "The system credential store could not save the API key"
            ) from error

    def delete_secret(self, provider_name: str) -> bool:
        keyring = self._required_keyring()
        try:
            keyring.delete_password(self.service_name, provider_name)
        except keyring.errors.PasswordDeleteError:
            return False
        except keyring.errors.KeyringError as error:
            raise CredentialStorageError(
                "The system credential store could not remove the API key"
            ) from error
        return True

    @staticmethod
    def _required_keyring():
        try:
            import keyring
        except ImportError as error:
            raise CredentialStorageError(
                "Secure API-key storage is unavailable; install GhostGUI with "
                "the ai extra or use an environment variable"
            ) from error
        return keyring


@dataclass(frozen=True)
class ChainedCredentialSource:
    """Return the first available secret from an ordered source list."""

    sources: tuple[CredentialSource, ...]

    def get_secret(self, provider_name: str) -> str | None:
        for source in self.sources:
            secret = source.get_secret(provider_name)
            if secret:
                return secret
        return None


def default_credential_source() -> CredentialSource:
    """Prefer OS credential storage, with the official environment fallback."""

    return ChainedCredentialSource(
        (SystemKeyringCredentialSource(), EnvironmentCredentialSource())
    )
