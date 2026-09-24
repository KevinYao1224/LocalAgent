from dataclasses import dataclass
from datetime import datetime

from llm.base import Message, ModelResponse, ToolCall
from observability.metrics import RunMetrics
from runtime.result import ToolResult


@dataclass(frozen=True, slots=True, kw_only=True)
class EventMetadata:
    """Runtime-supplied correlation and timing fields on every event.

    Optional defaults allow small hand-built events in existing experiments.
    AgentLoop always populates run_id and an aware UTC timestamp.
    """

    run_id: str | None = None
    step_id: str | None = None
    timestamp: datetime | None = None
    duration_ms: float | None = None


@dataclass(frozen=True, slots=True)
class AgentStarted(EventMetadata):
    max_steps: int
    initial_messages: tuple[Message, ...]


@dataclass(frozen=True, slots=True)
class ModelCallStarted(EventMetadata):
    step: int
    message_count: int
    tool_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StepPreparationFailed(EventMetadata):
    """Tool discovery or context construction failed before the model call."""

    step: int
    error_type: str
    error: str


@dataclass(frozen=True, slots=True)
class ModelResponseReceived(EventMetadata):
    step: int
    response: ModelResponse


@dataclass(frozen=True, slots=True)
class ModelCallFailed(EventMetadata):
    step: int
    error_type: str
    error: str


@dataclass(frozen=True, slots=True)
class ToolExecutionStarted(EventMetadata):
    step: int
    call: ToolCall


@dataclass(frozen=True, slots=True)
class ToolExecutionFinished(EventMetadata):
    step: int
    call: ToolCall
    result: ToolResult


@dataclass(frozen=True, slots=True)
class ToolExecutionFailed(EventMetadata):
    step: int
    call: ToolCall
    error_type: str
    error: str


@dataclass(frozen=True, slots=True)
class EmptyModelResponse(EventMetadata):
    step: int


@dataclass(frozen=True, slots=True)
class AgentFinished(EventMetadata):
    steps: int
    stop_reason: str
    final_content: str
    metrics: RunMetrics | None = None


AgentEvent = (
    AgentStarted
    | ModelCallStarted
    | StepPreparationFailed
    | ModelResponseReceived
    | ModelCallFailed
    | ToolExecutionStarted
    | ToolExecutionFinished
    | ToolExecutionFailed
    | EmptyModelResponse
    | AgentFinished
)
