"""Tests for session-only provider connection-test caching."""

from __future__ import annotations

import unittest

from application.ai.connection_cache import (
    ConnectionTestCache,
    connection_test_identity,
)


class ConnectionTestCacheTests(unittest.TestCase):
    def test_identity_hashes_credentials_without_retaining_plaintext(self):
        identity = connection_test_identity(
            "Gemini",
            "gemini-test",
            "secret-api-key",
            configuration="settings-dialog",
        )

        self.assertEqual(identity.provider, "gemini")
        self.assertEqual(len(identity.credential_fingerprint), 64)
        self.assertNotIn("secret-api-key", repr(identity))

    def test_provider_model_key_and_configuration_changes_invalidate(self):
        cache = ConnectionTestCache()
        original = connection_test_identity(
            "gemini",
            "model-a",
            "key-a",
            configuration="session-memory",
        )
        cache.record_success(original)
        self.assertTrue(cache.has_success(original))

        changes = (
            connection_test_identity(
                "anthropic", "model-a", "key-a", configuration="session-memory"
            ),
            connection_test_identity(
                "gemini", "model-b", "key-a", configuration="session-memory"
            ),
            connection_test_identity(
                "gemini", "model-a", "key-b", configuration="session-memory"
            ),
            connection_test_identity(
                "gemini", "model-a", "key-a", configuration="settings-dialog"
            ),
        )
        for changed in changes:
            with self.subTest(changed=changed):
                cache.record_success(original)
                self.assertFalse(cache.has_success(changed))
                self.assertFalse(cache.has_success(original))

    def test_explicit_invalidation_clears_success(self):
        cache = ConnectionTestCache()
        identity = connection_test_identity(
            "gemini",
            "model-a",
            None,
            configuration="keyring-or-environment",
        )
        cache.record_success(identity)
        cache.invalidate()
        self.assertFalse(cache.has_success(identity))


if __name__ == "__main__":
    unittest.main()
