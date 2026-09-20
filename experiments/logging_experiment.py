"""Capture and inspect the human-readable AgentLoop log.

Run from the project root:

    python experiments/logging_experiment.py
"""

from collections.abc import Iterable
from io import StringIO
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import AgentLoop
from llm.base import LLM, Message, ModelResponse, ToolCall
from observability import HumanReadableLogger
from runtime import ToolExecutor, ToolSchemaError
from tools import Tool, ToolRegistry, calculator_tools


class ScriptedLLM(LLM):
    def __init__(self, responses: Iterable[ModelResponse]) -> None:
        self._responses = iter(responses)

    def chat(self, messages, tools=None) -> ModelResponse:
        return next(self._responses)


def create_executor() -> ToolExecutor:
    registry = ToolRegistry()

    for tool in calculator_tools:
        registry.register(tool)

    return ToolExecutor(registry)


def experiment_successful_trace() -> str:
    stream = StringIO()
    logger = HumanReadableLogger(stream=stream)
    llm = ScriptedLLM([
        ModelResponse(tool_calls=[
            ToolCall(name="add", arguments={"a": 12, "b": 7}),
        ]),
        ModelResponse(tool_calls=[
            ToolCall(name="multiply", arguments={"a": 19, "b": 5}),
        ]),
        ModelResponse(
            content="The final answer is 95.",
            thinking=(
                "Both tool results are available.\n"
                "I should now answer the user."
            ),
            prompt_tokens=120,
            completion_tokens=12,
            done_reason="stop",
        ),
    ])

    AgentLoop(
        llm=llm,
        executor=create_executor(),
        max_steps=5,
        logger=logger,
    ).run([
        Message(role="user", content="Calculate (12 + 7) * 5."),
    ])

    output = stream.getvalue()
    assert "=== Agent run started ===" in output
    assert "--- Step 1: model call ---" in output
    assert '1. add({"a": 12, "b": 7})' in output
    assert "Tool result: add -> 19" in output
    assert "Tool result: multiply -> 95" in output
    assert "Model thinking:" in output
    assert "  Both tool results are available." in output
    assert "  I should now answer the user." in output
    assert "Model metadata: prompt=120, completion=12" in output
    assert "Stop reason: completed" in output
    assert "Final content: The final answer is 95." in output
    return output


def experiment_error_trace() -> str:
    stream = StringIO()
    llm = ScriptedLLM([
        ModelResponse(tool_calls=[
            ToolCall(name="divide", arguments={"a": 10, "b": 0}),
        ]),
        ModelResponse(content="Division by zero is undefined."),
    ])

    AgentLoop(
        llm=llm,
        executor=create_executor(),
        logger=HumanReadableLogger(stream=stream),
    ).run([
        Message(role="user", content="Calculate 10 / 0."),
    ])

    output = stream.getvalue()
    assert "Tool error: divide [execution_error]" in output
    assert "Cannot divide by zero" in output
    return output


def experiment_model_failure_trace() -> str:
    class FailingLLM(LLM):
        def chat(self, messages, tools=None) -> ModelResponse:
            raise RuntimeError("provider unavailable")

    stream = StringIO()

    try:
        AgentLoop(
            llm=FailingLLM(),
            executor=create_executor(),
            logger=HumanReadableLogger(stream=stream),
        ).run([
            Message(role="user", content="Calculate 1 + 1."),
        ])
    except RuntimeError as exc:
        assert str(exc) == "provider unavailable"
    else:
        raise AssertionError("The provider error must be re-raised.")

    output = stream.getvalue()
    assert "Model call failed: RuntimeError: provider unavailable" in output
    assert "Stop reason: error" in output
    return output


def experiment_tool_runtime_failure_trace() -> str:
    invalid_tool = Tool(
        name="invalid_tool",
        description="A tool with an invalid schema for this experiment.",
        parameters={"type": "not-a-valid-type"},
        handler=lambda: None,
    )
    registry = ToolRegistry()
    registry.register(invalid_tool)
    stream = StringIO()
    llm = ScriptedLLM([
        ModelResponse(tool_calls=[
            ToolCall(name="invalid_tool", arguments={}),
        ]),
    ])

    try:
        AgentLoop(
            llm=llm,
            executor=ToolExecutor(registry),
            logger=HumanReadableLogger(stream=stream),
        ).run([
            Message(role="user", content="Call the invalid tool."),
        ])
    except ToolSchemaError:
        pass
    else:
        raise AssertionError("The invalid tool schema must be re-raised.")

    output = stream.getvalue()
    assert "Tool execution failed: invalid_tool: ToolSchemaError" in output
    assert "Stop reason: error" in output
    return output


if __name__ == "__main__":
    print(experiment_successful_trace())
    print(experiment_error_trace())
    print(experiment_model_failure_trace())
    print(experiment_tool_runtime_failure_trace())
    print("All logging experiments passed.")
