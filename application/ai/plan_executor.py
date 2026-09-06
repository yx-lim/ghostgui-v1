"""Deterministic local execution of normalized semantic motion plans."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from typing import Any

from application.ai.edit_session import AIEditSessionState
from application.ai.errors import ProviderCancelledError
from application.ai.limits import MAX_TOOL_RESULT_CHARACTERS
from application.ai.motion_plan import MotionEditPlan, PlannedOperation
from application.ai.providers.base import CancellationSignal
from application.ai.semantic_tools import SemanticToolContext
from application.ai.tool_registry import ToolRegistry


class PlanExecutionError(RuntimeError):
    """Local plan execution or validation could not complete safely."""


@dataclass(frozen=True)
class PlannedOperationResult:
    index: int
    operation: PlannedOperation
    succeeded: bool
    changed: bool
    output: Any = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("planned operation index must not be negative")
        if self.succeeded and self.error is not None:
            raise ValueError("successful planned operation cannot contain an error")
        if not self.succeeded and not self.error:
            raise ValueError("failed planned operation requires an error")
        if self.changed and not self.succeeded:
            raise ValueError("failed planned operation cannot remain changed")


@dataclass(frozen=True)
class PlanExecutionResult:
    plan: MotionEditPlan
    operations: tuple[PlannedOperationResult, ...]
    validation: dict | None

    @property
    def failed_operations(self) -> tuple[PlannedOperationResult, ...]:
        return tuple(result for result in self.operations if not result.succeeded)

    @property
    def changed_operations(self) -> tuple[PlannedOperationResult, ...]:
        return tuple(result for result in self.operations if result.changed)

    @property
    def validation_passed(self) -> bool | None:
        if self.validation is None:
            return None
        return self.validation.get("valid") is True


class PlanExecutor:
    """Validate and execute allowlisted operations against an AI working copy."""

    def __init__(
        self,
        tools: ToolRegistry,
        *,
        max_result_characters: int = MAX_TOOL_RESULT_CHARACTERS,
    ) -> None:
        if max_result_characters <= 0:
            raise ValueError("plan result limit must be positive")
        self.tools = tools
        self.max_result_characters = int(max_result_characters)

    def execute(
        self,
        plan: MotionEditPlan,
        *,
        context: SemanticToolContext,
        cancellation_token: CancellationSignal | None = None,
    ) -> PlanExecutionResult:
        if plan.needs_clarification:
            return PlanExecutionResult(plan, (), None)
        if context.session.state not in {
            AIEditSessionState.READY,
            AIEditSessionState.STAGED,
        }:
            raise PlanExecutionError("plan requires an editable AI working copy")

        results = []
        for index, operation in enumerate(plan.operations):
            _raise_if_cancelled(cancellation_token)
            checkpoint = context.session.checkpoint()
            edit_count = len(context.session.edits)
            try:
                tool = self.tools.get(operation.tool)
                if not tool.mutates_working_copy:
                    raise PlanExecutionError(
                        f"planned tool is not a semantic edit: {operation.tool}"
                    )
                self.tools.validate_arguments(operation.tool, operation.arguments)
                output = self.tools.execute(
                    operation.tool,
                    operation.arguments,
                    context=context,
                )
                output = _json_value(
                    output,
                    max_characters=self.max_result_characters,
                )
            except Exception as error:
                context.session.restore_checkpoint(checkpoint)
                results.append(PlannedOperationResult(
                    index=index,
                    operation=operation,
                    succeeded=False,
                    changed=False,
                    error=_bounded_error(error, self.max_result_characters),
                ))
                continue
            results.append(PlannedOperationResult(
                index=index,
                operation=operation,
                succeeded=True,
                changed=len(context.session.edits) > edit_count,
                output=output,
            ))

        _raise_if_cancelled(cancellation_token)
        validation = None
        if context.session.has_changes:
            validation = _json_value(
                self.tools.execute("validate_motion", {}, context=context),
                max_characters=self.max_result_characters,
            )
            if not isinstance(validation, dict) or "valid" not in validation:
                raise PlanExecutionError("motion validation returned an invalid result")
        return PlanExecutionResult(plan, tuple(results), validation)


def local_proposal(result: PlanExecutionResult) -> tuple[str, tuple[str, ...]]:
    """Summarize known local effects without another provider request."""

    changed = result.changed_operations
    failed = result.failed_operations
    if result.plan.needs_clarification:
        question = result.plan.clarification_question or "Please clarify the request."
        return question, ("No motion changes were attempted",)

    groups = Counter(_proposal_group(item.operation) for item in changed)
    lines = tuple(
        _proposal_line(group, count)
        for group, count in groups.items()
    )
    if failed:
        count = len(failed)
        lines += (f"{count} planned operation{'s' if count != 1 else ''} failed locally",)
    if result.validation_passed is True:
        lines += ("Basic validation passed",)
    elif result.validation_passed is False:
        lines += ("Basic validation failed",)

    changed_count = len(changed)
    if changed_count:
        summary = (
            f"Proposed {changed_count} local motion "
            f"change{'s' if changed_count != 1 else ''}."
        )
    elif failed:
        summary = "No motion changes were staged because local execution failed."
    else:
        summary = "The plan required no additional motion changes."
    return summary, lines or ("No semantic motion change was reported",)


def _proposal_group(operation: PlannedOperation) -> tuple[str, str | None]:
    arguments = operation.arguments
    target_keys = {
        "set_logical_frame_target": "logical_frame",
        "move_end_effector": "end_effector",
        "set_joint_angle": "joint",
        "set_joint_group_angles": "joint_group",
        "protect_keyframe": "logical_frame",
    }
    key = target_keys.get(operation.tool)
    target = None if key is None else str(arguments.get(key, "")).strip() or None
    return operation.tool, target


def _proposal_line(group: tuple[str, str | None], count: int) -> str:
    tool, target = group
    target_text = "" if target is None else f" {target.replace('_', ' ')}"
    labels = {
        "ensure_keyframe": "Added or confirmed Keyframes",
        "set_logical_frame_target": "Modified",
        "move_end_effector": "Moved",
        "set_joint_angle": "Set Joint Angle for",
        "set_joint_group_angles": "Set Joint Angles for",
        "retime_segment": "Retimed Keyframe intervals",
        "protect_keyframe": "Updated protection for",
    }
    label = labels.get(tool, tool.replace("_", " ").title())
    if tool in {"ensure_keyframe", "retime_segment"}:
        return f"{label}: {count}"
    unit = "Keyframe" if count == 1 else "Keyframes"
    return f"{label}{target_text} at {count} {unit}"


def _json_value(value: Any, *, max_characters: int) -> Any:
    try:
        encoded = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise PlanExecutionError("tool returned a non-JSON result") from error
    if len(encoded) > max_characters:
        raise PlanExecutionError("tool result exceeds the local size limit")
    return json.loads(encoded)


def _bounded_error(error: Exception, max_characters: int) -> str:
    message = str(error).strip() or type(error).__name__
    if len(message) > max_characters:
        return "planned operation failed with an oversized error message"
    return message


def _raise_if_cancelled(token: CancellationSignal | None) -> None:
    if token is not None and token.cancellation_requested:
        raise ProviderCancelledError("motion plan execution was cancelled")
