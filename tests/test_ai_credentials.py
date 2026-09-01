"""Credential storage tests that never access a real system keyring."""

from __future__ import annotations

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from application.ai.credentials import (
    CredentialStorageError,
    SystemKeyringCredentialSource,
    SystemKeyringCredentialStore,
)


class _KeyringError(Exception):
    pass


class _PasswordDeleteError(_KeyringError):
    pass


class _FakeKeyring:
    errors = SimpleNamespace(
        KeyringError=_KeyringError,
        PasswordDeleteError=_PasswordDeleteError,
    )

    def __init__(self):
        self.values = {}

    def get_password(self, service, account):
        return self.values.get((service, account))

    def set_password(self, service, account, secret):
        self.values[(service, account)] = secret

    def delete_password(self, service, account):
        try:
            del self.values[(service, account)]
        except KeyError as error:
            raise _PasswordDeleteError() from error


class CredentialStoreTests(unittest.TestCase):
    def test_system_store_round_trip_and_explicit_removal(self):
        keyring = _FakeKeyring()
        with patch.dict(sys.modules, {"keyring": keyring}):
            store = SystemKeyringCredentialStore()
            store.set_secret("gemini", "secret-value")
            self.assertEqual(store.get_secret("gemini"), "secret-value")
            self.assertTrue(store.delete_secret("gemini"))
            self.assertFalse(store.delete_secret("gemini"))

    def test_read_failure_is_optional_but_write_failure_is_explicit(self):
        class _FailingKeyring(_FakeKeyring):
            def get_password(self, service, account):
                raise _KeyringError()

            def set_password(self, service, account, secret):
                raise _KeyringError()

        with patch.dict(sys.modules, {"keyring": _FailingKeyring()}):
            self.assertIsNone(SystemKeyringCredentialSource().get_secret("gemini"))
            with self.assertRaises(CredentialStorageError):
                SystemKeyringCredentialStore().set_secret("gemini", "secret")

    def test_empty_secret_is_never_written(self):
        with patch.dict(sys.modules, {"keyring": _FakeKeyring()}):
            with self.assertRaises(ValueError):
                SystemKeyringCredentialStore().set_secret("gemini", "")


if __name__ == "__main__":
    unittest.main()
