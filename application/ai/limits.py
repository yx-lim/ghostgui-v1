"""Shared local budgets for optional AI requests and responses."""

DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_MOTION_PLANNER_OUTPUT_TOKENS = 8192
MAX_AI_OUTPUT_TOKENS = 16_384
MAX_AI_INSTRUCTION_CHARACTERS = 16_000
MAX_AI_RESPONSE_CHARACTERS = 65_536
MAX_TOOL_RESULT_CHARACTERS = 65_536
MAX_MOTION_FRAME_BYTES = 8 * 1024 * 1024

# Compact Motion Assistant request budgets. These are deliberately separate
# from the legacy semantic agent loop.
MAX_MOTION_CONTEXT_CHARACTERS = 120_000
MAX_MOTION_NUMERICAL_SAMPLES = 20
MAX_MOTION_IMAGES = 8
MAX_MOTION_IMAGE_DIMENSION = 512
DEFAULT_MOTION_REQUEST_TIMEOUT_SECONDS = 180.0
MAX_MOTION_REQUEST_TIMEOUT_SECONDS = 180.0
MAX_GENERATED_MOTION_DURATION_SECONDS = 120.0


from dataclasses import dataclass as _dataclass


@_dataclass(frozen=True)
class AgentRunLimits:
    """Per-run and per-session bounds for the in-app command bar loop."""

    max_turns: int = 30
    max_tool_calls: int = 100
    max_images: int = 12
    max_wall_seconds: float = 900.0
    max_turn_seconds: float = 300.0
    max_run_cost_usd: float = 1.00
    max_session_cost_usd: float = 10.00
    max_consecutive_tool_errors: int = 4
    max_identical_failures: int = 2
    max_confirmations: int = 5
    max_tool_result_bytes: int = 65_536
    new_conversation_input_tokens: int = 200_000
    budget_notice_fraction: float = 0.8
    max_stream_reissues: int = 2

    def __post_init__(self):
        for name, value in vars(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")


AGENT_MAX_OUTPUT_TOKENS = 64_000
AGENT_MAX_DIGEST_BYTES = 1_536
