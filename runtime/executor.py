from llm.base import ToolCall
from runtime.result import ToolResult
from tools.base import Tool, ToolExecutionError
from tools.registry import ToolNotFoundError, ToolRegistry


class ToolExecutor:
    """Execute untrusted model tool calls through a ToolRegistry."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def available_tools(self) -> list[Tool]:
        """Return the tools that may be advertised to the model."""

        return self._registry.all()

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute one call and turn expected failures into a ToolResult."""

        try:
            value = self._registry.execute(
                name=call.name,
                arguments=call.arguments,
            )
        except (ToolNotFoundError, ToolExecutionError) as exc:
            return ToolResult.failed(
                tool_name=call.name,
                error=str(exc),
            )

        return ToolResult.succeeded(
            tool_name=call.name,
            value=value,
        )
