import json

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The structured outcome of one tool execution."""

    tool_name: str
    success: bool
    value: Any = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.success and self.error is not None:
            raise ValueError("A successful ToolResult cannot contain an error.")
        if not self.success and not self.error:
            raise ValueError("A failed ToolResult must contain an error.")

    @classmethod
    def succeeded(cls, tool_name: str, value: Any) -> "ToolResult":
        return cls(
            tool_name=tool_name,
            success=True,
            value=value,
        )

    @classmethod
    def failed(cls, tool_name: str, error: str) -> "ToolResult":
        return cls(
            tool_name=tool_name,
            success=False,
            error=error,
        )

    def to_message_content(self) -> str:
        """Convert this result to the text sent back to the model."""

        if not self.success:
            return f"Error: {self.error}"

        if isinstance(self.value, str):
            return self.value

        try:
            return json.dumps(self.value, ensure_ascii=False)
        except TypeError:
            return str(self.value)
