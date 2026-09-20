from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from tools.base import Tool


Role = Literal["system", "user", "assistant", "tool"]


class ReasoningReplaySupport(str, Enum):
    """Whether a provider adapter can send reasoning back to its API."""

    UNSUPPORTED = "unsupported"
    SUPPORTED = "supported"


@dataclass(frozen=True, slots=True)
class LLMCapabilities:
    """Provider transport capabilities used by the agent runtime."""

    reasoning_replay: ReasoningReplaySupport = (
        ReasoningReplaySupport.UNSUPPORTED
    )


@dataclass(frozen=True, slots=True)
class ReasoningBlock:
    """Provider reasoning captured with provenance for safe replay."""

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
        """Stable provider name used in reasoning provenance."""

        return type(self).__name__

    @property
    def model_name(self) -> str | None:
        """Model identifier when the adapter exposes one."""

        return None

    @property
    def capabilities(self) -> LLMCapabilities:
        """Use conservative defaults for adapters without declarations."""

        return LLMCapabilities()

    def create_reasoning_block(
        self,
        response: ModelResponse,
    ) -> ReasoningBlock | None:
        """Attach provider provenance to a raw thinking response."""

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
