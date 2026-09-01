"""Typed failures shared by AI providers, agents, and motion tools."""


class AIError(RuntimeError):
    """Base class for expected AI workflow failures."""


class ProviderError(AIError):
    """A provider request could not produce a usable response."""


class ProviderCapabilityError(ProviderError):
    """A request needs a feature unsupported by the selected provider/model."""


class ProviderCancelledError(ProviderError):
    """A provider request was cooperatively cancelled."""


class ToolRegistrationError(AIError, ValueError):
    """A tool definition is invalid or conflicts with an existing tool."""


class ToolNotFoundError(AIError, LookupError):
    """A requested tool is not present in the explicit registry."""


class ToolValidationError(AIError, ValueError):
    """Tool arguments do not satisfy the registered input schema."""


class ToolExecutionError(AIError):
    """A registered tool failed while executing validated arguments."""
