"""Contracts for compact provider-neutral trajectory edit specifications."""

from __future__ import annotations

import json
import unittest

from application.ai.trajectory_edit_spec import (
    MAX_TRAJECTORY_OPERATIONS,
    TrajectoryEditMode,
    TrajectoryEditSpecError,
    TrajectoryOperationType,
    parse_trajectory_edit_spec,
    trajectory_edit_spec_response_schema,
    trajectory_operation_argument_contracts,
)


def _wire(mode, operation_type, arguments):
    return json.dumps({
        "mode": mode,
        "summary": "Compact motion intent.",
        "operations": [{
            "type": operation_type,
            "arguments": json.dumps(arguments, separators=(",", ":")),
        }],
    })


def _sparse_keyframes():
    values = []
    for time, height in ((0.0, 0.85), (1.0, 0.5), (3.0, 0.3), (5.0, 0.85)):
        values.append({
            "time_seconds": time,
            "root_position_m": [0.0, 0.0, height],
            "torso_rpy_rad": [0.0, 0.0, 0.0],
            "end_effector_targets": [],
            "joint_targets": [],
        })
    return {"duration_seconds": 5.0, "keyframes": values}


class TrajectoryEditSpecTests(unittest.TestCase):
    def test_wire_schema_is_compact_bounded_and_contains_no_dense_qpos(self):
        schema = trajectory_edit_spec_response_schema()
        operation = schema["properties"]["operations"]

        self.assertEqual(operation["maxItems"], MAX_TRAJECTORY_OPERATIONS)
        self.assertEqual(
            set(operation["items"]["properties"]["type"]["enum"]),
            {item.value for item in TrajectoryOperationType},
        )
        self.assertEqual(
            operation["items"]["properties"]["arguments"]["type"],
            "string",
        )
        self.assertNotIn("qpos", json.dumps(schema).lower())
        self.assertEqual(
            set(trajectory_operation_argument_contracts()),
            {item.value for item in TrajectoryOperationType},
        )

    def test_acceptance_operations_parse_into_strict_local_values(self):
        cases = (
            (
                "edit",
                "root_offset",
                {"start_time": 0.0, "end_time": 8.7, "translation_m": [0, 0, 0.05]},
            ),
            (
                "edit",
                "hold_pose",
                {
                    "source_time": 0.5,
                    "start_time": 0.5,
                    "end_time": 2.0,
                    "body_scope": "joint_group",
                    "body_name": "both_arms",
                },
            ),
            (
                "edit",
                "retime_interval",
                {"start_time": 2.0, "end_time": 3.0, "scale": 1.5},
            ),
            (
                "edit",
                "set_logical_frame_target",
                {
                    "logical_frame": "torso",
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "mode": "relative",
                    "position_m": None,
                    "orientation_rpy_rad": [0.0, 0.1, 0.0],
                },
            ),
            ("generate", "sparse_keyframes", _sparse_keyframes()),
        )

        for mode, operation_type, arguments in cases:
            with self.subTest(operation_type=operation_type):
                spec = parse_trajectory_edit_spec(
                    _wire(mode, operation_type, arguments)
                )
                self.assertEqual(spec.mode, TrajectoryEditMode(mode))
                self.assertEqual(
                    spec.operations[0].operation_type,
                    TrajectoryOperationType(operation_type),
                )
                self.assertEqual(dict(spec.operations[0].arguments), arguments)

    def test_malformed_or_unsafe_operation_arguments_fail_closed(self):
        invalid = (
            "not json",
            json.dumps({"mode": "edit", "summary": "Missing operations"}),
            _wire(
                "edit",
                "root_offset",
                {"start_time": 1.0, "end_time": 0.0, "translation_m": [0, 0, 0.05]},
            ),
            _wire(
                "edit",
                "root_offset",
                {"start_time": 0.0, "end_time": 1.0, "translation_m": [0, 0, float("nan")]},
            ),
            _wire("edit", "sparse_keyframes", _sparse_keyframes()),
            _wire(
                "edit",
                "set_logical_frame_target",
                {
                    "logical_frame": "torso",
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "mode": "relative",
                    "position_m": None,
                    "orientation_rpy_rad": None,
                },
            ),
            _wire(
                "generate",
                "sparse_keyframes",
                {
                    **_sparse_keyframes(),
                    "duration_seconds": 121.0,
                },
            ),
        )

        for text in invalid:
            with self.subTest(text=text), self.assertRaises(TrajectoryEditSpecError):
                parse_trajectory_edit_spec(text)

    def test_sparse_generation_requires_strictly_ordered_full_duration(self):
        arguments = _sparse_keyframes()
        arguments["keyframes"][2]["time_seconds"] = 0.5

        with self.assertRaisesRegex(TrajectoryEditSpecError, "strictly increasing"):
            parse_trajectory_edit_spec(
                _wire("generate", "sparse_keyframes", arguments)
            )


if __name__ == "__main__":
    unittest.main()
