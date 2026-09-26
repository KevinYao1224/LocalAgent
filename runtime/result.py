import json

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ToolErrorType(str, Enum):
    """预期内工具调用失败的分类。"""

    TOOL_NOT_FOUND = "tool_not_found"
    TOOL_NOT_AVAILABLE = "tool_not_available"
    PERMISSION_DENIED = "permission_denied"
    VALIDATION_ERROR = "validation_error"
    EXECUTION_ERROR = "execution_error"


@dataclass(frozen=True, slots=True)
class ToolResult:
    """一次工具执行的结构化结果。"""

    tool_name: str
    success: bool
    value: Any = None
    error: str | None = None
    error_type: ToolErrorType | None = None

    def __post_init__(self) -> None:
        if self.success and (
            self.error is not None
            or self.error_type is not None
        ):
            raise ValueError(
                "A successful ToolResult cannot contain error information."
            )
        if not self.success and (
            not self.error
            or self.error_type is None
        ):
            raise ValueError(
                "A failed ToolResult must contain an error and error_type."
            )

    @classmethod
    def succeeded(cls, tool_name: str, value: Any) -> "ToolResult":
        return cls(
            tool_name=tool_name,
            success=True,
            value=value,
        )

    @classmethod
    def failed(
        cls,
        tool_name: str,
        error: str,
        error_type: ToolErrorType,
    ) -> "ToolResult":
        return cls(
            tool_name=tool_name,
            success=False,
            error=error,
            error_type=error_type,
        )

    def to_message_content(self) -> str:
        """将结果转换为返回给模型的文本。"""

        if not self.success:
            assert self.error_type is not None
            return f"Error [{self.error_type.value}]: {self.error}"

        if isinstance(self.value, str):
            return self.value

        try:
            return json.dumps(self.value, ensure_ascii=False)
        except TypeError:
            return str(self.value)
