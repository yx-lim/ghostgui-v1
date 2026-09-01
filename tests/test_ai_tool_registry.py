"""Tests for strict AI tool registration, validation, and dispatch."""

from __future__ import annotations

import unittest

from application.ai.errors import (
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistrationError,
    ToolValidationError,
)
from application.ai.tool_registry import ToolCategory, ToolRegistry, ToolSpec


def _target_schema():
    return {
        "type": "object",
        "properties": {
            "logical_frame": {"type": "string", "minLength": 1},
            "time_seconds": {"type": "number", "minimum": 0.0},
            "position": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3,
            },
            "space": {
                "type": "string",
                "enum": ["world", "relative"],
            },
        },
        "required": ["logical_frame", "time_seconds", "position", "space"],
        "additionalProperties": False,
    }


class ToolRegistryTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.registry = ToolRegistry()
        self.registry.register(
            ToolSpec(
                name="set_logical_frame_target",
                description="Set a semantic body or End Effector target.",
                input_schema=_target_schema(),
                handler=self._handle,
                category=ToolCategory.EDIT,
                mutates_working_copy=True,
            )
        )

    def _handle(self, arguments, context):
        self.calls.append((arguments, context))
        return {"ok": True}

    def test_registered_definition_is_provider_safe(self):
        definition = self.registry.definitions()[0]
        self.assertEqual(definition.name, "set_logical_frame_target")
        self.assertNotIn("handler", definition.__dict__)

    def test_executes_only_after_strict_validation(self):
        arguments = {
            "logical_frame": "pelvis",
            "time_seconds": 1.25,
            "position": [0.0, 0.0, 0.7],
            "space": "world",
        }
        result = self.registry.execute("set_logical_frame_target", arguments, context="session")
        self.assertEqual(result, {"ok": True})
        self.assertEqual(self.calls, [(arguments, "session")])

    def test_rejects_unknown_tool_and_extra_argument(self):
        with self.assertRaises(ToolNotFoundError):
            self.registry.execute("run_python", {})
        with self.assertRaisesRegex(ToolValidationError, "unknown property"):
            self.registry.execute(
                "set_logical_frame_target",
                {
                    "logical_frame": "pelvis",
                    "time_seconds": 1.0,
                    "position": [0.0, 0.0, 0.7],
                    "space": "world",
                    "code": "arbitrary()",
                },
            )
        self.assertEqual(self.calls, [])

    def test_rejects_missing_wrong_length_and_non_finite_values(self):
        base = {
            "logical_frame": "pelvis",
            "time_seconds": 1.0,
            "position": [0.0, 0.0, 0.7],
            "space": "world",
        }
        for replacement, pattern in (
            ({"space": None}, "must be string"),
            ({"position": [0.0, 0.0]}, "below minItems"),
            ({"time_seconds": float("inf")}, "must be finite"),
        ):
            arguments = dict(base)
            arguments.update(replacement)
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ToolValidationError, pattern):
                    self.registry.execute("set_logical_frame_target", arguments)

    def test_registration_requires_closed_object_schema(self):
        with self.assertRaisesRegex(ToolRegistrationError, "additionalProperties"):
            ToolRegistry().register(
                ToolSpec(
                    name="unsafe_tool",
                    description="Unsafe open schema.",
                    input_schema={"type": "object", "properties": {}},
                    handler=self._handle,
                    category=ToolCategory.TEST,
                )
            )

    def test_handler_failure_is_wrapped(self):
        registry = ToolRegistry()

        def fail(arguments, context):
            raise RuntimeError("solver exploded")

        registry.register(
            ToolSpec(
                name="validate_motion",
                description="Validate the candidate motion.",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=fail,
                category=ToolCategory.TEST,
            )
        )
        with self.assertRaisesRegex(ToolExecutionError, "solver exploded"):
            registry.execute("validate_motion", {})


if __name__ == "__main__":
    unittest.main()
