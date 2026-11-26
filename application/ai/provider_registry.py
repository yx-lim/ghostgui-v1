"""Small registry for the live AI providers assembled by GhostGUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from application.ai.providers.base import LLMProvider
from application.ai.providers.anthropic import (
    DEFAULT_ANTHROPIC_CAPABILITIES,
    DEFAULT_CLAUDE_MODEL,
)
from application.ai.providers.gemini import DEFAULT_GEMINI_CAPABILITIES
from application.ai.schemas import ProviderCapabilities


ProviderFactory = Callable[..., LLMProvider]
DEFAULT_GEMINI_MODEL = "gemini-3.7-flash"


@dataclass(frozen=True)
class ProviderRegistration:
    """Composition data needed by controllers and provider settings UI."""

    name: str
    display_name: str
    factory: ProviderFactory
    models: tuple[str, ...]
    capabilities: ProviderCapabilities
    credential_identifier: str
    environment_variables: tuple[str, ...]
    sdk_distribution: str
    agent_models: tuple[str, ...] = ()
    supports_agent_loop: bool = False
    supports_fallbacks: bool = False
    supports_mid_conversation_system: bool = False
    agent_factory: Callable[..., object] | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or self.name != self.name.strip().lower():
            raise ValueError("provider name must be a lowercase identifier")
        if not self.display_name.strip():
            raise ValueError("provider display name must not be empty")
        if not callable(self.factory):
            raise TypeError("provider factory must be callable")
        if not self.models or any(not model.strip() for model in self.models):
            raise ValueError("provider models must contain non-empty identifiers")
        if len(set(self.models)) != len(self.models):
            raise ValueError("provider models must be unique")
        if not isinstance(self.capabilities, ProviderCapabilities):
            raise TypeError("provider capabilities must use ProviderCapabilities")
        if not self.credential_identifier.strip():
            raise ValueError("credential identifier must not be empty")
        if any(not variable.strip() for variable in self.environment_variables):
            raise ValueError("credential environment variables must not be empty")
        if not self.sdk_distribution.strip():
            raise ValueError("SDK distribution must not be empty")
        if self.supports_agent_loop and (not self.agent_models or self.agent_factory is None):
            raise ValueError("agent-capable providers need agent models and an agent factory")

    @property
    def default_model(self) -> str:
        return self.models[0]

    def create(self, *, api_key: str | None = None) -> LLMProvider:
        return self.factory(api_key=api_key)

    def create_agent(self, *, api_key=None):
        if not self.supports_agent_loop or self.agent_factory is None:
            raise ValueError(f"{self.display_name} does not support the command bar agent loop")
        return self.agent_factory(api_key=api_key)


class ProviderRegistry:
    """Validated immutable-by-contract lookup for a few built-in providers."""

    def __init__(
        self,
        registrations: tuple[ProviderRegistration, ...],
        *,
        default_name: str,
    ) -> None:
        if not registrations:
            raise ValueError("provider registry must not be empty")
        entries = {
            registration.name: registration
            for registration in registrations
        }
        if len(entries) != len(registrations):
            raise ValueError("provider names must be unique")
        credential_identifiers = {
            registration.credential_identifier
            for registration in registrations
        }
        if len(credential_identifiers) != len(registrations):
            raise ValueError("provider credential identifiers must be unique")
        if default_name not in entries:
            raise ValueError("default provider must be registered")
        self._registrations = tuple(registrations)
        self._entries = entries
        self.default_name = default_name

    @property
    def registrations(self) -> tuple[ProviderRegistration, ...]:
        return self._registrations

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(registration.name for registration in self._registrations)

    @property
    def default(self) -> ProviderRegistration:
        return self._entries[self.default_name]

    def get(self, provider_name: str) -> ProviderRegistration:
        name = str(provider_name).strip().lower()
        try:
            return self._entries[name]
        except KeyError as error:
            raise ValueError(f"Unsupported AI provider: {provider_name}") from error

    def create(
        self,
        provider_name: str,
        *,
        api_key: str | None = None,
    ) -> LLMProvider:
        return self.get(provider_name).create(api_key=api_key)


def _gemini_factory(*, api_key: str | None = None) -> LLMProvider:
    from application.ai.providers.gemini import GeminiProvider

    return GeminiProvider(api_key=api_key)


def _anthropic_agent_factory(*, api_key=None):
    from application.ai.providers.anthropic_agent import AnthropicAgentProvider

    return AnthropicAgentProvider(api_key=api_key)


def _anthropic_factory(*, api_key: str | None = None) -> LLMProvider:
    from application.ai.providers.anthropic import AnthropicProvider

    return AnthropicProvider(api_key=api_key)


DEFAULT_PROVIDER_REGISTRY = ProviderRegistry(
    (
        ProviderRegistration(
            name="gemini",
            display_name="Gemini",
            factory=_gemini_factory,
            models=(DEFAULT_GEMINI_MODEL, "gemini-3.6-flash"),
            capabilities=DEFAULT_GEMINI_CAPABILITIES,
            credential_identifier="gemini",
            environment_variables=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
            sdk_distribution="google-genai",
        ),
        ProviderRegistration(
            name="anthropic",
            display_name="Anthropic",
            factory=_anthropic_factory,
            models=(
                DEFAULT_CLAUDE_MODEL,
                "claude-sonnet-4-6",
                "claude-haiku-4-5-20251001",
            ),
            capabilities=DEFAULT_ANTHROPIC_CAPABILITIES,
            credential_identifier="anthropic",
            environment_variables=("ANTHROPIC_API_KEY",),
            sdk_distribution="anthropic",
            agent_models=("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"),
            supports_agent_loop=True,
            supports_fallbacks=True,
            supports_mid_conversation_system=True,
            agent_factory=_anthropic_agent_factory,
        ),
    ),
    default_name="gemini",
)
