from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class Message:
    role: Role
    content: str
    tool_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }

        if self.tool_name is not None:
            data["tool_name"] = self.tool_name

        return data


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ModelResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)

    done_reason: str | None = None

    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLM(ABC):

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        pass