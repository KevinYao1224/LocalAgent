from collections.abc import Iterable

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
        enabled_tool_names: Iterable[str] | None = None,
    ) -> None:
        self._registry = registry
        self._validator = (
            validator
            if validator is not None
            else ToolArgumentsValidator()
        )
        self._enabled_tool_names: set[str] | None = None
        self.set_enabled_tools(enabled_tool_names)

    def available_tools(self) -> list[Tool]:
        """Return exactly the tools that are currently executable."""

        tools = self._registry.all()
        if self._enabled_tool_names is None:
            return tools
        return [
            tool for tool in tools
            if tool.name in self._enabled_tool_names
        ]

    def set_enabled_tools(
        self,
        tool_names: Iterable[str] | None,
    ) -> None:
        """Replace the active capability set; ``None`` enables all tools.

        Restriction affects both model advertisement and execution. Registered
        tools outside this set remain in the registry but cannot be invoked.
        """

        if tool_names is None:
            self._enabled_tool_names = None
            return
        if isinstance(tool_names, (str, bytes)):
            raise ValueError("enabled tool names must be an iterable of names.")

        names = set(tool_names)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("enabled tool names must be nonempty strings.")

        registered_names = {tool.name for tool in self._registry.all()}
        unknown_names = names - registered_names
        if unknown_names:
            unknown = ", ".join(sorted(unknown_names))
            raise ValueError(f"Enabled tools are not registered: {unknown}.")

        self._enabled_tool_names = names

    def disable_tool(self, tool_name: str) -> None:
        """Remove one registered tool from advertisement and execution."""

        self._require_registered(tool_name)
        if self._enabled_tool_names is None:
            self._enabled_tool_names = {
                tool.name for tool in self._registry.all()
            }
        self._enabled_tool_names.discard(tool_name)

    def enable_tool(self, tool_name: str) -> None:
        """Add one registered tool to a restricted capability set."""

        self._require_registered(tool_name)
        if self._enabled_tool_names is not None:
            self._enabled_tool_names.add(tool_name)

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

        # This deliberately checks the same method used for model
        # advertisement. Future permission-aware executors may override or
        # dynamically filter available_tools(); execution must honor that view.
        if not any(
            available.name == call.name
            for available in self.available_tools()
        ):
            return ToolResult.failed(
                tool_name=call.name,
                error=f"Tool '{call.name}' is not available in this context.",
                error_type=ToolErrorType.TOOL_NOT_AVAILABLE,
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

    def _require_registered(self, tool_name: str) -> None:
        try:
            self._registry.get(tool_name)
        except ToolNotFoundError as exc:
            raise ValueError(str(exc)) from exc
