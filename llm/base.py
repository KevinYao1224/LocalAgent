from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from tools.base import Tool


Role = Literal["system", "user", "assistant", "tool"]


class ReasoningReplaySupport(str, Enum):
    """声明 provider 适配器能否将 reasoning 再发送给对应 API。"""

    UNSUPPORTED = "unsupported"
    SUPPORTED = "supported"


@dataclass(frozen=True, slots=True)
class LLMCapabilities:
    """供 Agent Runtime 使用的 provider 传输能力声明。"""

    reasoning_replay: ReasoningReplaySupport = (
        ReasoningReplaySupport.UNSUPPORTED
    )


@dataclass(frozen=True, slots=True)
class ReasoningBlock:
    """保存带来源信息的 provider reasoning，以便在满足条件时安全回放。"""

    provider: str
    model: str | None
    raw_thinking: str
    replayable: bool
    provider_state: Any = None


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class Message:
    role: Role
    content: str
    tool_name: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str = ""


@dataclass(slots=True)
class ModelResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str = ""
    done_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLM(ABC):
    @property
    def provider_name(self) -> str:
        """返回稳定的 provider 名称，供 reasoning 来源标记使用。"""

        return type(self).__name__

    @property
    def model_name(self) -> str | None:
        """适配器能够提供时，返回模型标识。"""

        return None

    @property
    def capabilities(self) -> LLMCapabilities:
        """未声明能力的适配器使用保守的默认值。"""

        return LLMCapabilities()

    def create_reasoning_block(
        self,
        response: ModelResponse,
    ) -> ReasoningBlock | None:
        """为原始 thinking 响应附加 provider 来源信息。"""

        if not response.thinking:
            return None

        replayable = (
            self.capabilities.reasoning_replay
            is ReasoningReplaySupport.SUPPORTED
        )
        return ReasoningBlock(
            provider=self.provider_name,
            model=self.model_name,
            raw_thinking=response.thinking,
            replayable=replayable,
        )

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
    ) -> ModelResponse:
        pass
