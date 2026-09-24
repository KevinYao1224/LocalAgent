"""ToolExecutor 和 ToolResult 的确定性实验。

从项目根目录运行：

    python experiments/tool_executor_experiment.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.base import ToolCall
from runtime import ToolErrorType, ToolExecutor, ToolResult
from tools.base import Tool
from tools.calculator import multiply_tool
from tools.registry import ToolRegistry


def fail_deliberately() -> None:
    raise RuntimeError("deliberate handler failure")


failing_tool = Tool(
    name="fail_deliberately",
    description="Always fail so the runtime error path can be inspected.",
    parameters={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    handler=fail_deliberately,
)


class FilteredExecutor(ToolExecutor):
    """模拟未来由权限层过滤可提供工具的 executor。"""

    def available_tools(self):
        return []


def create_executor() -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(multiply_tool)
    registry.register(failing_tool)
    return ToolExecutor(registry)


def show(label: str, result: ToolResult) -> None:
    print(f"\n=== {label} ===")
    print(f"tool_name: {result.tool_name}")
    print(f"success:   {result.success}")
    print(f"value:     {result.value!r}")
    print(f"error:     {result.error!r}")
    print(f"message:   {result.to_message_content()!r}")


def main() -> None:
    executor = create_executor()

    success = executor.execute(ToolCall(
        name="multiply",
        arguments={"a": 23, "b": 47},
    ))
    assert success.success
    assert success.value == 1081
    assert success.error is None
    show("successful execution", success)

    unknown = executor.execute(ToolCall(
        name="missing_tool",
        arguments={},
    ))
    assert not unknown.success
    assert unknown.error is not None
    assert "not registered" in unknown.error
    assert unknown.error_type is ToolErrorType.TOOL_NOT_FOUND
    show("unknown tool", unknown)

    invalid_arguments = executor.execute(ToolCall(
        name="multiply",
        arguments={"a": 23},
    ))
    assert not invalid_arguments.success
    assert invalid_arguments.error is not None
    assert "Invalid arguments" in invalid_arguments.error
    assert invalid_arguments.error_type is ToolErrorType.VALIDATION_ERROR
    show("invalid arguments", invalid_arguments)

    handler_failure = executor.execute(ToolCall(
        name="fail_deliberately",
        arguments={},
    ))
    assert not handler_failure.success
    assert handler_failure.error is not None
    assert "deliberate handler failure" in handler_failure.error
    assert handler_failure.error_type is ToolErrorType.EXECUTION_ERROR
    show("handler failure", handler_failure)

    restricted_registry = ToolRegistry()
    restricted_registry.register(multiply_tool)
    restricted = ToolExecutor(
        restricted_registry,
        enabled_tool_names=(),
    )
    assert restricted.available_tools() == []
    unavailable = restricted.execute(ToolCall(
        name="multiply",
        arguments={"a": 23, "b": 47},
    ))
    assert not unavailable.success
    assert unavailable.error_type is ToolErrorType.TOOL_NOT_AVAILABLE
    assert "not available" in unavailable.error
    show("registered but unavailable", unavailable)

    restricted.enable_tool("multiply")
    assert [tool.name for tool in restricted.available_tools()] == [
        "multiply"
    ]
    assert restricted.execute(ToolCall(
        name="multiply",
        arguments={"a": 2, "b": 3},
    )).value == 6
    restricted.disable_tool("multiply")
    assert restricted.available_tools() == []

    try:
        restricted.set_enabled_tools(["missing_tool"])
    except ValueError as exc:
        assert "not registered" in str(exc)
    else:
        raise AssertionError("An unregistered enabled tool was accepted.")

    try:
        restricted.set_enabled_tools("multiply")
    except ValueError as exc:
        assert "iterable of names" in str(exc)
    else:
        raise AssertionError("A single string was accepted as a name iterable.")

    filtered_registry = ToolRegistry()
    filtered_registry.register(multiply_tool)
    filtered = FilteredExecutor(filtered_registry)
    filtered_result = filtered.execute(ToolCall(
        name="multiply",
        arguments={"a": 2, "b": 4},
    ))
    assert filtered_result.error_type is ToolErrorType.TOOL_NOT_AVAILABLE

    print("\nAll ToolExecutor experiments passed.")


if __name__ == "__main__":
    main()
