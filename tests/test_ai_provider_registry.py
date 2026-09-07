"""Tests for the centralized live-provider composition registry."""

from __future__ import annotations

from dataclasses import replace
import unittest

from application.ai.provider_registry import (
    DEFAULT_PROVIDER_REGISTRY,
    ProviderRegistration,
    ProviderRegistry,
)
from application.ai.schemas import ProviderCapabilities


class _Provider:
    def __init__(self, api_key=None):
        self.api_key = api_key


def _registration(name="future", factory=None):
    return ProviderRegistration(
        name=name,
        display_name=name.title(),
        factory=factory or (lambda *, api_key=None: _Provider(api_key)),
        models=(f"{name}-default", f"{name}-small"),
        capabilities=ProviderCapabilities(
            supports_tools=True,
            supports_vision=False,
        ),
        credential_identifier=f"{name}-credential",
        environment_variables=(f"{name.upper()}_API_KEY",),
        sdk_distribution=f"{name}-sdk",
    )


class ProviderRegistryTests(unittest.TestCase):
    def test_default_registry_centralizes_live_provider_composition(self):
        self.assertEqual(DEFAULT_PROVIDER_REGISTRY.names, ("gemini", "anthropic"))
        gemini = DEFAULT_PROVIDER_REGISTRY.default
        self.assertEqual(gemini.name, "gemini")
        self.assertEqual(gemini.default_model, "gemini-3.7-flash")
        self.assertEqual(gemini.environment_variables[0], "GOOGLE_API_KEY")
        self.assertTrue(gemini.capabilities.supports_vision)

    def test_registration_factory_receives_ephemeral_key(self):
        registry = ProviderRegistry((_registration(),), default_name="future")

        provider = registry.create("future", api_key="temporary-secret")

        self.assertEqual(provider.api_key, "temporary-secret")
        self.assertEqual(registry.get(" FUTURE ").default_model, "future-default")

    def test_registry_rejects_duplicate_and_unknown_providers(self):
        registration = _registration()
        with self.assertRaisesRegex(ValueError, "unique"):
            ProviderRegistry(
                (registration, registration),
                default_name="future",
            )
        other = replace(registration, name="other", display_name="Other")
        with self.assertRaisesRegex(ValueError, "credential identifiers"):
            ProviderRegistry(
                (registration, other),
                default_name="future",
            )
        registry = ProviderRegistry((registration,), default_name="future")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            registry.get("missing")


if __name__ == "__main__":
    unittest.main()
