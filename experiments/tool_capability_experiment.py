"""Verify that remembered tool calls cannot bypass current capabilities.

Run from the project root:

    .venv/bin/python experiments/tool_capability_experiment.py
"""

from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import AgentLoop
from llm.base import LLM, Message, ModelResponse, ToolCall
from runtime import ToolErrorType, ToolExecutor
from tools import Tool, ToolRegistry


class RememberedCallLLM(LLM):
    """Request a historical tool even though it is no longer advertised."""

    def __init__(self) -> None:
        self._responses = iter([
            ModelResponse(tool_calls=[ToolCall(
                name="read_secret",
                arguments={"key": "ALPHA"},
            )]),
            ModelResponse(content="The runtime denied the remembered call."),
        ])
        self.advertised_tool_names: list[list[str]] = []
        self.calls: list[list[Message]] = []

    def chat(self, messages, tools=None):
        self.calls.append(deepcopy(messages))
        self.advertised_tool_names.append([
            tool.name for tool in (tools or [])
        ])
        return next(self._responses)


def main() -> None:
    executions: list[str] = []

    def read_secret(key: str) -> str:
        executions.append(key)
        return f"secret-for-{key}"

    secret_tool = Tool(
        name="read_secret",
        description="Read a secret by key.",
        parameters={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
        handler=read_secret,
    )
    registry = ToolRegistry()
    registry.register(secret_tool)
    executor = ToolExecutor(registry)

    # Simulate a previous turn in which the tool was advertised and executed.
    previous = executor.execute(ToolCall(
        name="read_secret",
        arguments={"key": "ALPHA"},
    ))
    assert previous.success
    assert executions == ["ALPHA"]

    remembered_call = ToolCall(
        name="read_secret",
        arguments={"key": "ALPHA"},
    )
    history = [
        Message(role="user", content="Read secret ALPHA."),
        Message(role="assistant", content="", tool_calls=[remembered_call]),
        Message(
            role="tool",
            content=previous.to_message_content(),
            tool_name="read_secret",
        ),
        Message(role="assistant", content="The secret was read."),
        Message(role="user", content="Use that tool again."),
    ]

    # The registry still contains read_secret, but current capabilities do not.
    executor.disable_tool("read_secret")
    llm = RememberedCallLLM()
    result = AgentLoop(llm=llm, executor=executor, max_steps=3).run(history)

    denied = result.tool_results[0]
    assert result.completed
    assert llm.advertised_tool_names == [[], []]
    assert denied.error_type is ToolErrorType.TOOL_NOT_AVAILABLE
    assert executions == ["ALPHA"]
    assert registry.get("read_secret") is secret_tool

    print("=== Remembered tool capability check ===")
    print("registry_contains=read_secret")
    print("advertised_tools=[]")
    print("model_requested=read_secret(key='ALPHA')")
    print(f"runtime_result={denied.to_message_content()!r}")
    print(f"handler_execution_count={len(executions)}")
    print("verdict=PASS")


if __name__ == "__main__":
    main()
