"""Provider-neutral comparison of bounded GhostGUI agent outcomes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from time import monotonic
from typing import Any

from application.ai.agent import (
    AgentError,
    AgentLimitError,
    AgentRunResult,
    AgentTimeoutError,
    AgentValidationError,
)
from application.ai.edit_session import AIEditSession, AIEditSessionState
from application.ai.errors import AIError, ProviderCancelledError
from application.ai.motion_state import capture_motion_state
from application.ai.schemas import EditAuthor, MessageRole
from application.ai.tool_registry import ToolRegistry
from application.project_document import ProjectDocument


class ProviderRunStatus(str, Enum):
    """Stable outcome categories that do not expose provider SDK details."""

    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    LIMITED = "limited"
    INVALID_MOTION = "invalid_motion"
    FAILED = "failed"


@dataclass(frozen=True)
class SemanticToolObservation:
    """One tool request, excluding provider-native identifiers and prose."""

    name: str
    arguments_digest: str
    arguments_valid: bool
    succeeded: bool


@dataclass(frozen=True)
class ProviderBehaviorSnapshot:
    """Normalized evidence from one provider run over a detached document."""

    provider_name: str
    model: str
    status: ProviderRunStatus
    baseline_motion_digest: str
    resulting_motion_digest: str
    motion_changed: bool
    semantic_tools: tuple[SemanticToolObservation, ...]
    edit_operations: tuple[str, ...]
    edit_authors: tuple[EditAuthor, ...]
    validation_passed: bool | None
    provider_turns: int
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float


@dataclass(frozen=True)
class ProviderComparisonCase:
    """One independently prepared agent run for a comparison scenario."""

    provider_name: str
    model: str
    baseline_document: ProjectDocument
    session: AIEditSession
    tools: ToolRegistry
    run: Callable[[], Awaitable[AgentRunResult]]

    def __post_init__(self) -> None:
        if not self.provider_name.strip() or not self.model.strip():
            raise ValueError("comparison provider and model must not be empty")
        if not callable(self.run):
            raise TypeError("comparison run must be callable")


@dataclass(frozen=True)
class ProviderBehaviorComparison:
    """Semantic and operational differences for one shared scenario."""

    scenario: str
    runs: tuple[ProviderBehaviorSnapshot, ...]
    semantic_differences: tuple[str, ...]
    operational_differences: tuple[str, ...]

    @property
    def semantically_equivalent(self) -> bool:
        return not self.semantic_differences

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe report without provider prose or raw motion state."""

        return {
            "scenario": self.scenario,
            "semantically_equivalent": self.semantically_equivalent,
            "semantic_differences": list(self.semantic_differences),
            "operational_differences": list(self.operational_differences),
            "runs": [
                {
                    "provider": run.provider_name,
                    "model": run.model,
                    "status": run.status.value,
                    "motion_changed": run.motion_changed,
                    "baseline_motion_digest": run.baseline_motion_digest,
                    "resulting_motion_digest": run.resulting_motion_digest,
                    "semantic_tools": [
                        {
                            "name": tool.name,
                            "arguments_digest": tool.arguments_digest,
                            "arguments_valid": tool.arguments_valid,
                            "succeeded": tool.succeeded,
                        }
                        for tool in run.semantic_tools
                    ],
                    "edit_operations": list(run.edit_operations),
                    "edit_authors": [author.value for author in run.edit_authors],
                    "validation_passed": run.validation_passed,
                    "provider_turns": run.provider_turns,
                    "usage": {
                        "input_tokens": run.input_tokens,
                        "output_tokens": run.output_tokens,
                    },
                    "elapsed_seconds": run.elapsed_seconds,
                }
                for run in self.runs
            ],
        }


async def compare_provider_agents(
    scenario: str,
    cases: Sequence[ProviderComparisonCase],
) -> ProviderBehaviorComparison:
    """Run 2--4 prepared agents sequentially and compare semantic outcomes.

    Each case must own a separate ``AIEditSession`` based on the same committed
    motion. Sequential execution avoids cross-provider rate races and makes this
    suitable for explicit, opt-in live comparisons as well as MockProvider CI.
    """

    if not scenario.strip():
        raise ValueError("comparison scenario must not be empty")
    if not 2 <= len(cases) <= 4:
        raise ValueError("provider comparison requires between 2 and 4 cases")

    _validate_cases(cases)

    snapshots = []
    for case in cases:
        started = monotonic()
        try:
            result = await case.run()
            error: AIError | AgentError | None = None
        except (AIError, AgentError) as caught:
            result = None
            error = caught
        snapshots.append(_capture_behavior(
            case,
            result=result,
            error=error,
            elapsed_seconds=monotonic() - started,
        ))

    return _compare(scenario.strip(), tuple(snapshots))


def _validate_cases(cases: Sequence[ProviderComparisonCase]) -> None:
    baseline_digests = {_motion_digest(case.baseline_document) for case in cases}
    if len(baseline_digests) != 1:
        raise ValueError("provider comparison cases must use identical baseline motion")
    if len({id(case.session) for case in cases}) != len(cases):
        raise ValueError("provider comparison cases must use separate edit sessions")
    labels = {(case.provider_name, case.model) for case in cases}
    if len(labels) != len(cases):
        raise ValueError("provider comparison cases require unique provider/model labels")
    for case in cases:
        if case.session.state is not AIEditSessionState.READY or case.session.edits:
            raise ValueError("provider comparison cases must start with fresh edit sessions")
        if _motion_digest(case.session.working_document) != _motion_digest(
            case.baseline_document
        ):
            raise ValueError(
                "provider comparison working copies must match their baseline motion"
            )


def _capture_behavior(
    case: ProviderComparisonCase,
    *,
    result: AgentRunResult | None,
    error: AIError | AgentError | None,
    elapsed_seconds: float,
) -> ProviderBehaviorSnapshot:
    baseline_digest = _motion_digest(case.baseline_document)
    resulting_digest = _motion_digest(case.session.working_document)
    if result is None:
        status = _error_status(error)
        tools = ()
        validation_passed = None
        provider_turns = 0
        input_tokens = 0
        output_tokens = 0
    else:
        status = ProviderRunStatus.COMPLETED
        tools = _semantic_tools(result, case.tools)
        validation_passed = (
            None if result.validation is None else result.validation.get("valid") is True
        )
        provider_turns = result.provider_turns
        input_tokens = result.usage.input_tokens
        output_tokens = result.usage.output_tokens

    return ProviderBehaviorSnapshot(
        provider_name=case.provider_name,
        model=case.model,
        status=status,
        baseline_motion_digest=baseline_digest,
        resulting_motion_digest=resulting_digest,
        motion_changed=baseline_digest != resulting_digest,
        semantic_tools=tools,
        edit_operations=tuple(edit.operation for edit in case.session.edits),
        edit_authors=tuple(edit.author for edit in case.session.edits),
        validation_passed=validation_passed,
        provider_turns=provider_turns,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_seconds=round(float(elapsed_seconds), 6),
    )


def _semantic_tools(
    result: AgentRunResult,
    registry: ToolRegistry,
) -> tuple[SemanticToolObservation, ...]:
    requested = [
        call
        for message in result.transcript
        if message.role is MessageRole.ASSISTANT
        for call in message.tool_calls
    ]
    if len(requested) != len(result.tool_executions):
        raise ValueError("agent result has inconsistent tool-call evidence")
    observations = []
    for call, execution in zip(requested, result.tool_executions):
        if call.name != execution.name:
            raise ValueError("agent result tool execution does not match its transcript")
        try:
            registry.validate_arguments(call.name, call.arguments)
            arguments_valid = True
        except AIError:
            arguments_valid = False
        observations.append(SemanticToolObservation(
            name=call.name,
            arguments_digest=_value_digest(dict(call.arguments)),
            arguments_valid=arguments_valid,
            succeeded=execution.succeeded,
        ))
    return tuple(observations)


def _compare(
    scenario: str,
    runs: tuple[ProviderBehaviorSnapshot, ...],
) -> ProviderBehaviorComparison:
    semantic_fields = {
        "status": lambda run: run.status,
        "semantic_tool_sequence": lambda run: tuple(
            (
                tool.name,
                tool.arguments_digest,
                tool.arguments_valid,
                tool.succeeded,
            )
            for tool in run.semantic_tools
        ),
        "edit_sequence": lambda run: tuple(zip(run.edit_operations, run.edit_authors)),
        "validation": lambda run: run.validation_passed,
        "resulting_motion": lambda run: run.resulting_motion_digest,
    }
    operational_fields = {
        "provider_turns": lambda run: run.provider_turns,
        "token_usage": lambda run: (run.input_tokens, run.output_tokens),
    }
    semantic = tuple(
        name for name, getter in semantic_fields.items()
        if len({getter(run) for run in runs}) > 1
    )
    operational = tuple(
        name for name, getter in operational_fields.items()
        if len({getter(run) for run in runs}) > 1
    )
    return ProviderBehaviorComparison(scenario, runs, semantic, operational)


def _error_status(error: AIError | AgentError | None) -> ProviderRunStatus:
    if isinstance(error, ProviderCancelledError):
        return ProviderRunStatus.CANCELLED
    if isinstance(error, AgentTimeoutError):
        return ProviderRunStatus.TIMED_OUT
    if isinstance(error, AgentLimitError):
        return ProviderRunStatus.LIMITED
    if isinstance(error, AgentValidationError):
        return ProviderRunStatus.INVALID_MOTION
    return ProviderRunStatus.FAILED


def _motion_digest(document: ProjectDocument) -> str:
    state = capture_motion_state(document)
    return _value_digest({
        "model_key": state.model_key,
        "trajectory": state.trajectory,
        "active_index": state.active_index,
        "current_time": state.current_time,
        "timeline_duration": state.timeline_duration,
        "qpos_states": state.qpos_states,
    })


def _value_digest(value: Any) -> str:
    payload = json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("comparison evidence contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        return _canonical_value(to_list())
    item = getattr(value, "item", None)
    if callable(item):
        return _canonical_value(item())
    raise TypeError(f"comparison evidence is not JSON-compatible: {type(value).__name__}")
