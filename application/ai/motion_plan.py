"""Provider-neutral semantic motion-plan values and JSON contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

from application.ai.schemas import ToolDefinition
from application.ai.tool_registry import ToolRegistry


MAX_PLANNED_OPERATIONS = 16
MAX_PLAN_SUMMARY_CHARACTERS = 4_000
MAX_CLARIFICATION_CHARACTERS = 2_000


class MotionPlanError(RuntimeError):
    """A provider plan is malformed or outside the semantic contract."""


@dataclass(frozen=True)
class PlannedOperation:
    tool: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.tool.strip():
            raise ValueError("planned operation tool must not be empty")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("planned operation arguments must be an object")
        object.__setattr__(self, "arguments", dict(self.arguments))


@dataclass(frozen=True)
class MotionEditPlan:
    summary: str
    needs_clarification: bool
    clarification_question: str | None
    operations: tuple[PlannedOperation, ...]

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ValueError("motion edit plan summary must not be empty")
        if len(self.summary) > MAX_PLAN_SUMMARY_CHARACTERS:
            raise ValueError("motion edit plan summary exceeds the local size limit")
        question = self.clarification_question
        if question is not None:
            question = question.strip() or None
            object.__setattr__(self, "clarification_question", question)
        if question is not None and len(question) > MAX_CLARIFICATION_CHARACTERS:
            raise ValueError("clarification question exceeds the local size limit")
        if self.needs_clarification:
            if question is None:
                raise ValueError("clarification plan requires a question")
            if self.operations:
                raise ValueError("clarification plan cannot contain operations")
        elif question is not None:
            raise ValueError("non-clarification plan cannot contain a question")
        if len(self.operations) > MAX_PLANNED_OPERATIONS:
            raise ValueError("motion edit plan exceeds the operation limit")


def editable_tool_definitions(
    registry: ToolRegistry,
) -> tuple[ToolDefinition, ...]:
    """Return only registered operations allowed to mutate the working copy."""

    return tuple(
        definition
        for definition in registry.definitions()
        if registry.get(definition.name).mutates_working_copy
    )


def motion_edit_plan_response_schema(registry: ToolRegistry) -> dict[str, Any]:
    """Build a portable provider schema from the active semantic allowlist.

    Gemini rejects realistic nested argument-schema envelopes on its current
    generateContent endpoint. Arguments therefore cross the provider boundary
    as a JSON object string, are decoded into PlannedOperation.arguments, and
    receive complete ToolRegistry validation before execution.
    """

    operation_schema = planned_operation_response_schema(registry)
    return {
        "type": "object",
        "properties": {
            # Text bounds are enforced again while parsing. Keeping them out of
            # the provider schema preserves the common Gemini/Claude subset.
            "summary": {"type": "string"},
            "needs_clarification": {"type": "boolean"},
            # An empty string is normalized to None locally. This avoids the
            # nullable type union rejected by older generateContent endpoints.
            "clarification_question": {"type": "string"},
            "operations": {
                "type": "array",
                "items": operation_schema,
                "maxItems": MAX_PLANNED_OPERATIONS,
            },
        },
        "required": [
            "summary",
            "needs_clarification",
            "clarification_question",
            "operations",
        ],
        "additionalProperties": False,
    }


def motion_repair_response_schema(registry: ToolRegistry) -> dict[str, Any]:
    """Return the replacement-operations-only repair response contract."""

    return {
        "type": "object",
        "properties": {
            "operations": {
                "type": "array",
                "items": planned_operation_response_schema(registry),
                "minItems": 1,
                "maxItems": MAX_PLANNED_OPERATIONS,
            },
        },
        "required": ["operations"],
        "additionalProperties": False,
    }


def parse_motion_edit_plan(text: str) -> MotionEditPlan:
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError) as error:
        raise MotionPlanError("provider returned malformed motion-plan JSON") from error
    required = {
        "summary",
        "needs_clarification",
        "clarification_question",
        "operations",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise MotionPlanError("motion edit plan has invalid fields")
    if not isinstance(payload["summary"], str):
        raise MotionPlanError("motion edit plan summary must be text")
    if not isinstance(payload["needs_clarification"], bool):
        raise MotionPlanError("needs_clarification must be boolean")
    if payload["clarification_question"] is not None and not isinstance(
        payload["clarification_question"],
        str,
    ):
        raise MotionPlanError("clarification question must be text or null")
    if not isinstance(payload["operations"], list):
        raise MotionPlanError("motion edit plan operations must be a list")

    operations = _parse_wire_operations(payload["operations"])
    try:
        return MotionEditPlan(
            summary=payload["summary"],
            needs_clarification=payload["needs_clarification"],
            clarification_question=payload["clarification_question"],
            operations=tuple(operations),
        )
    except (TypeError, ValueError) as error:
        raise MotionPlanError(str(error)) from error


def parse_motion_repair_plan(text: str) -> MotionEditPlan:
    """Normalize a replacement-only provider response for PlanExecutor."""

    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError) as error:
        raise MotionPlanError("provider returned malformed motion-repair JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"operations"}:
        raise MotionPlanError("motion repair has invalid fields")
    values = payload["operations"]
    if not isinstance(values, list) or not values:
        raise MotionPlanError("motion repair requires replacement operations")
    operations = _parse_wire_operations(values)
    try:
        return MotionEditPlan(
            summary="Replacement operations",
            needs_clarification=False,
            clarification_question=None,
            operations=operations,
        )
    except (TypeError, ValueError) as error:
        raise MotionPlanError(str(error)) from error


def planned_operation_response_schema(registry: ToolRegistry) -> dict[str, Any]:
    """Return the strict wire schema for one allowlisted semantic operation."""

    definitions = editable_tool_definitions(registry)
    if not definitions:
        raise MotionPlanError("no semantic edit tools are registered")
    return {
        "type": "object",
        "properties": {
            "tool": {
                "type": "string",
                "enum": [definition.name for definition in definitions],
            },
            "arguments": {
                "type": "string",
                "description": (
                    "Compact JSON object matching the selected semantic "
                    "operation's argument schema."
                ),
            },
        },
        "required": ["tool", "arguments"],
        "additionalProperties": False,
    }


def _parse_wire_operations(values: list) -> tuple[PlannedOperation, ...]:
    operations = []
    for value in values:
        if not isinstance(value, dict) or set(value) != {"tool", "arguments"}:
            raise MotionPlanError("planned operation has invalid fields")
        if not isinstance(value["tool"], str):
            raise MotionPlanError("planned operation tool or arguments are invalid")
        raw_arguments = value["arguments"]
        if not isinstance(raw_arguments, str):
            raise MotionPlanError("planned operation arguments must be JSON text")
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as error:
            raise MotionPlanError(
                "planned operation arguments contain malformed JSON"
            ) from error
        if not isinstance(arguments, Mapping):
            raise MotionPlanError(
                "planned operation arguments must decode to an object"
            )
        operations.append(PlannedOperation(value["tool"], arguments))
    if len(operations) > MAX_PLANNED_OPERATIONS:
        raise MotionPlanError("motion plan exceeds the operation limit")
    return tuple(operations)
