"""Session-only cache identity for explicit provider connection tests."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class ConnectionTestIdentity:
    provider: str
    model: str
    credential_fingerprint: str

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.model.strip():
            raise ValueError("connection-test provider and model must not be empty")
        if len(self.credential_fingerprint) != 64:
            raise ValueError("connection-test credential fingerprint is invalid")


class ConnectionTestCache:
    """Remember only the current successful test, never a plaintext credential."""

    def __init__(self) -> None:
        self._successful_identity: ConnectionTestIdentity | None = None

    def has_success(self, identity: ConnectionTestIdentity) -> bool:
        if self._successful_identity != identity:
            self._successful_identity = None
            return False
        return True

    def record_success(self, identity: ConnectionTestIdentity) -> None:
        if not isinstance(identity, ConnectionTestIdentity):
            raise TypeError("connection-test identity has an invalid type")
        self._successful_identity = identity

    def invalidate(self) -> None:
        self._successful_identity = None


def connection_test_identity(
    provider: str,
    model: str,
    credential: str | None,
    *,
    configuration: str,
) -> ConnectionTestIdentity:
    """Hash the effective credential and its source without retaining either."""

    provider_name = provider.strip().lower()
    model_name = model.strip()
    source = configuration.strip()
    if not provider_name or not model_name or not source:
        raise ValueError("connection-test identity fields must not be empty")
    secret = "" if credential is None else str(credential)
    digest = hashlib.sha256(
        source.encode("utf-8") + b"\0" + secret.encode("utf-8")
    ).hexdigest()
    return ConnectionTestIdentity(provider_name, model_name, digest)
