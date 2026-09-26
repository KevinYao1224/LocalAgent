from collections.abc import Iterable

from llm.base import ToolCall
from runtime.permissions import PermissionDeniedError
from runtime.result import ToolErrorType, ToolResult
from runtime.validation import (
    ToolArgumentsValidationError,
    ToolArgumentsValidator,
)
from tools.base import Tool, ToolExecutionError
from tools.registry import ToolNotFoundError, ToolRegistry


class ToolExecutor:
    """通过 ToolRegistry 执行不可信的模型工具调用。"""

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
        """返回当前确实允许执行的工具。"""

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
        """替换当前 capability 集合；传入 ``None`` 表示启用所有工具。

        限制同时作用于提供给模型的工具列表和实际执行。集合之外的工具仍保留在注册表中，
        但不能被调用。
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
        """从可提供给模型和可执行集合中移除一个已注册工具。"""

        self._require_registered(tool_name)
        if self._enabled_tool_names is None:
            self._enabled_tool_names = {
                tool.name for tool in self._registry.all()
            }
        self._enabled_tool_names.discard(tool_name)

    def enable_tool(self, tool_name: str) -> None:
        """将一个已注册工具加入受限的 capability 集合。"""

        self._require_registered(tool_name)
        if self._enabled_tool_names is not None:
            self._enabled_tool_names.add(tool_name)

    def execute(self, call: ToolCall) -> ToolResult:
        """执行一次调用，并将预期内的失败转换为 ToolResult。"""

        try:
            tool = self._registry.get(call.name)
        except ToolNotFoundError as exc:
            return ToolResult.failed(
                tool_name=call.name,
                error=str(exc),
                error_type=ToolErrorType.TOOL_NOT_FOUND,
            )

        # 这里有意复用提供给模型的工具列表。未来带权限策略的 executor
        # 可能重写或动态过滤 available_tools()；实际执行必须遵守该结果。
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
        except PermissionDeniedError as exc:
            return ToolResult.failed(
                tool_name=call.name,
                error=str(exc),
                error_type=ToolErrorType.PERMISSION_DENIED,
            )
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
