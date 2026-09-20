from dataclasses import dataclass

from llm.base import Message, ModelResponse, ToolCall
from runtime.result import ToolResult


@dataclass(frozen=True, slots=True)
class AgentStarted:
    max_steps: int
    initial_messages: tuple[Message, ...]


@dataclass(frozen=True, slots=True)
class ModelCallStarted:
    step: int
    message_count: int
    tool_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelResponseReceived:
    step: int
    response: ModelResponse


@dataclass(frozen=True, slots=True)
class ModelCallFailed:
    step: int
    error_type: str
    error: str


@dataclass(frozen=True, slots=True)
class ToolExecutionStarted:
    step: int
    call: ToolCall


@dataclass(frozen=True, slots=True)
class ToolExecutionFinished:
    step: int
    call: ToolCall
    result: ToolResult


@dataclass(frozen=True, slots=True)
class ToolExecutionFailed:
    step: int
    call: ToolCall
    error_type: str
    error: str


@dataclass(frozen=True, slots=True)
class EmptyModelResponse:
    step: int


@dataclass(frozen=True, slots=True)
class AgentFinished:
    steps: int
    stop_reason: str
    final_content: str


AgentEvent = (
    AgentStarted
    | ModelCallStarted
    | ModelResponseReceived
    | ModelCallFailed
    | ToolExecutionStarted
    | ToolExecutionFinished
    | ToolExecutionFailed
    | EmptyModelResponse
    | AgentFinished
)
