"""Typed failures shared by AI providers, agents, and motion tools."""

from __future__ import annotations

from typing import Any, Mapping


class AIError(RuntimeError):
    """Base class for expected AI workflow failures."""


class ProviderError(AIError):
    """A provider request could not produce a usable response."""


class ProviderCapabilityError(ProviderError):
    """A request needs a feature unsupported by the selected provider/model."""


class ProviderCancelledError(ProviderError):
    """A provider request was cooperatively cancelled."""


class ProviderConfigurationError(ProviderError):
    """A provider cannot start because required local configuration is absent."""


class ProviderAuthenticationError(ProviderError):
    """A provider rejected its credentials."""


class ProviderRateLimitError(ProviderError):
    """A provider rejected a request because its usage limit was reached."""

    retry_after: float | None = None


class ProviderResponseError(ProviderError):
    """A provider returned a response that cannot satisfy the common contract."""

    def __init__(
        self,
        message: str,
        *,
        diagnostic_details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostic_details = dict(diagnostic_details or {})


class ToolRegistrationError(AIError, ValueError):
    """A tool definition is invalid or conflicts with an existing tool."""


class ToolNotFoundError(AIError, LookupError):
    """A requested tool is not present in the explicit registry."""


class ToolValidationError(AIError, ValueError):
    """Tool arguments do not satisfy the registered input schema."""


class ToolExecutionError(AIError):
    """A registered tool failed while executing validated arguments."""


class ProviderOfflineError(ProviderError):
    """The provider could not be reached (connection or timeout)."""


class ProviderOverloadedError(ProviderError):
    """The provider is overloaded or temporarily unavailable after SDK retries."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ProviderPermissionError(ProviderError):
    """The key is valid but lacks permission for the request."""


class ProviderModelUnavailableError(ProviderError):
    """The selected model does not exist or is unavailable for this key."""


class ProviderBadRequestError(ProviderError):
    """The provider rejected the request shape."""

    def __init__(self, message: str, *, request_id: str = "") -> None:
        super().__init__(message)
        self.request_id = request_id


class ProviderRefusalError(ProviderError):
    """The model (and any fallback) declined the request."""

    def __init__(self, message: str, *, category: str | None = None) -> None:
        super().__init__(message)
        self.category = category


class ProviderContextLimitError(ProviderError):
    """The conversation no longer fits the model's context window."""


class ProviderMalformedStreamError(ProviderError):
    """A streamed tool input could not be parsed; the turn may be re-issued."""
