"""Deterministic experiments for ToolExecutor and ToolResult.

Run from the project root:

    python experiments/tool_executor_experiment.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.base import ToolCall
from runtime.executor import ToolExecutor, ToolResult
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
    assert "not registered" in unknown.error
    show("unknown tool", unknown)

    invalid_arguments = executor.execute(ToolCall(
        name="multiply",
        arguments={"a": 23},
    ))
    assert not invalid_arguments.success
    assert "Invalid arguments" in invalid_arguments.error
    show("invalid arguments", invalid_arguments)

    handler_failure = executor.execute(ToolCall(
        name="fail_deliberately",
        arguments={},
    ))
    assert not handler_failure.success
    assert "deliberate handler failure" in handler_failure.error
    show("handler failure", handler_failure)

    print("\nAll ToolExecutor experiments passed.")


if __name__ == "__main__":
    main()
