from dataclasses import dataclass
from datetime import datetime

from llm.base import Message, ModelResponse, ToolCall
from observability.metrics import RunMetrics
from runtime.result import ToolResult


@dataclass(frozen=True, slots=True, kw_only=True)
class EventMetadata:
    """由 Runtime 为事件提供的关联标识和计时字段。

    可选默认值使现有实验能够简便地手动构造事件。AgentLoop 总会填入 run_id 和
    带时区的 UTC 时间戳。
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
    """模型调用开始前，工具发现或上下文构造失败。"""

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
class RecoveryDecision(EventMetadata):
    """整批工具调用结束后，对失败结果作出的有界恢复决定。"""

    step: int
    error_types: tuple[str, ...]
    decision: str
    corrections_used: int
    self_critique_next_step: bool = False


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
    | RecoveryDecision
    | AgentFinished
)
