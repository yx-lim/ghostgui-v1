"""Provider-neutral contracts for GhostGUI's bounded AI editing workflow."""

from application.ai.errors import (
    AIError,
    ProviderCancelledError,
    ProviderCapabilityError,
    ProviderError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistrationError,
    ToolValidationError,
)
from application.ai.schemas import (
    EditAuthor,
    ImageVariant,
    MessageRole,
    MotionEntityRef,
    MotionFrameImage,
    ProviderCapabilities,
    ProviderMessage,
    ProviderRequest,
    ProviderResponse,
    StopReason,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Usage,
)
from application.ai.tool_registry import ToolCategory, ToolRegistry, ToolSpec
from application.ai.providers import LLMProvider, MockProvider, MockStep
from application.ai.edit_session import (
    AIEditSession,
    AIEditSessionError,
    AIEditSessionState,
    SessionEditRecord,
)
from application.ai.metadata import (
    InMemoryMotionMetadataStore,
    MotionEditMetadata,
    MotionMetadataService,
    TimestampMotionIdentityResolver,
)
from application.ai.motion_state import MotionStateSnapshot, ReplaceMotionState
from application.ai.context import (
    AIContext,
    ContextBuilder,
    EditorSelectionContext,
    RobotCapabilityContext,
)

__all__ = [
    "AIError",
    "AIEditSession",
    "AIEditSessionError",
    "AIEditSessionState",
    "AIContext",
    "ContextBuilder",
    "EditAuthor",
    "EditorSelectionContext",
    "ImageVariant",
    "InMemoryMotionMetadataStore",
    "MessageRole",
    "LLMProvider",
    "MockProvider",
    "MockStep",
    "MotionEditMetadata",
    "MotionEntityRef",
    "MotionFrameImage",
    "MotionMetadataService",
    "MotionStateSnapshot",
    "ProviderCancelledError",
    "ProviderCapabilityError",
    "ProviderCapabilities",
    "ProviderError",
    "ProviderMessage",
    "ProviderRequest",
    "ProviderResponse",
    "ReplaceMotionState",
    "RobotCapabilityContext",
    "SessionEditRecord",
    "StopReason",
    "ToolCall",
    "ToolCategory",
    "ToolDefinition",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolRegistrationError",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "ToolValidationError",
    "TimestampMotionIdentityResolver",
    "Usage",
]
