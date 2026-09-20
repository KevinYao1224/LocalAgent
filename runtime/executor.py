from llm.base import ToolCall
from runtime.result import ToolErrorType, ToolResult
from runtime.validation import (
    ToolArgumentsValidationError,
    ToolArgumentsValidator,
)
from tools.base import Tool, ToolExecutionError
from tools.registry import ToolNotFoundError, ToolRegistry


class ToolExecutor:
    """Execute untrusted model tool calls through a ToolRegistry."""

    def __init__(
        self,
        registry: ToolRegistry,
        validator: ToolArgumentsValidator | None = None,
    ) -> None:
        self._registry = registry
        self._validator = (
            validator
            if validator is not None
            else ToolArgumentsValidator()
        )

    def available_tools(self) -> list[Tool]:
        """Return the tools that may be advertised to the model."""

        return self._registry.all()

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute one call and turn expected failures into a ToolResult."""

        try:
            tool = self._registry.get(call.name)
        except ToolNotFoundError as exc:
            return ToolResult.failed(
                tool_name=call.name,
                error=str(exc),
                error_type=ToolErrorType.TOOL_NOT_FOUND,
            )

        try:
            self._validator.validate(tool, call.arguments)
        except ToolArgumentsValidationError as exc:
            return ToolResult.failed(
                tool_name=call.name,
                error=str(exc),
                error_type=ToolErrorType.VALIDATION_ERROR,
            )

        try:
            value = tool.execute(call.arguments)
        except ToolExecutionError as exc:
            return ToolResult.failed(
                tool_name=call.name,
                error=str(exc),
                error_type=ToolErrorType.EXECUTION_ERROR,
            )

        return ToolResult.succeeded(
            tool_name=call.name,
            value=value,
        )
