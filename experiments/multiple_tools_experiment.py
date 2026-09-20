"""Experiments for multiple calculator tools and multi-step AgentLoop runs.

Run from the project root:

    python experiments/multiple_tools_experiment.py
"""

from collections.abc import Iterable
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import AgentLoop, AgentRunResult
from llm.base import LLM, Message, ModelResponse, ToolCall
from runtime import ToolErrorType, ToolExecutor
from tools import ToolRegistry, calculator_tools


class ScriptedLLM(LLM):
    """Return prepared responses so only Runtime behavior is tested."""

    def __init__(self, responses: Iterable[ModelResponse]) -> None:
        self._responses = iter(responses)

    def chat(self, messages, tools=None) -> ModelResponse:
        return next(self._responses)


def create_executor() -> ToolExecutor:
    registry = ToolRegistry()

    for tool in calculator_tools:
        registry.register(tool)

    return ToolExecutor(registry)


def print_trace(title: str, result: AgentRunResult) -> None:
    print(f"\n=== {title} ===")
    print(f"stop_reason={result.stop_reason.value}, steps={result.steps}")

    for message in result.messages:
        calls = [call.name for call in message.tool_calls]
        extra = f", tool_calls={calls}" if calls else ""
        print(f"{message.role:>9}: {message.content!r}{extra}")


def experiment_dependent_calls() -> None:
    llm = ScriptedLLM([
        ModelResponse(tool_calls=[
            ToolCall(name="add", arguments={"a": 12, "b": 7}),
        ]),
        ModelResponse(tool_calls=[
            ToolCall(name="multiply", arguments={"a": 19, "b": 5}),
        ]),
        ModelResponse(content="(12 + 7) multiplied by 5 is 95."),
    ])
    result = AgentLoop(llm, create_executor()).run([
        Message(role="user", content="Calculate (12 + 7) * 5."),
    ])

    assert result.completed
    assert result.steps == 3
    assert [item.tool_name for item in result.tool_results] == [
        "add",
        "multiply",
    ]
    assert [item.value for item in result.tool_results] == [19, 95]
    assert [message.role for message in result.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    print_trace("dependent calls: add then multiply", result)


def experiment_multiple_calls_in_one_response() -> None:
    llm = ScriptedLLM([
        ModelResponse(tool_calls=[
            ToolCall(name="subtract", arguments={"a": 10, "b": 4}),
            ToolCall(name="divide", arguments={"a": 20, "b": 5}),
        ]),
        ModelResponse(content="The independent results are 6 and 4."),
    ])
    result = AgentLoop(llm, create_executor()).run([
        Message(
            role="user",
            content="Calculate 10 - 4 and, independently, 20 / 5.",
        ),
    ])

    assert result.completed
    assert result.steps == 2
    assert [item.tool_name for item in result.tool_results] == [
        "subtract",
        "divide",
    ]
    assert [item.value for item in result.tool_results] == [6, 4.0]
    print_trace("multiple calls in one model response", result)


def experiment_division_by_zero_recovery() -> None:
    llm = ScriptedLLM([
        ModelResponse(tool_calls=[
            ToolCall(name="divide", arguments={"a": 10, "b": 0}),
        ]),
        ModelResponse(content="10 cannot be divided by zero."),
    ])
    result = AgentLoop(llm, create_executor()).run([
        Message(role="user", content="Calculate 10 / 0."),
    ])

    assert result.completed
    assert len(result.tool_results) == 1
    assert not result.tool_results[0].success
    assert (
        result.tool_results[0].error_type
        is ToolErrorType.EXECUTION_ERROR
    )
    assert "Cannot divide by zero" in result.messages[-2].content
    print_trace("division by zero becomes an observation", result)


def experiment_empty_response_recovery() -> None:
    llm = ScriptedLLM([
        ModelResponse(),
        ModelResponse(content="The final answer is 95."),
    ])
    result = AgentLoop(llm, create_executor()).run([
        Message(role="user", content="Return a final answer."),
    ])

    assert result.completed
    assert result.steps == 2
    assert result.response.content == "The final answer is 95."
    print_trace("empty model response is not a final answer", result)


if __name__ == "__main__":
    experiment_dependent_calls()
    experiment_multiple_calls_in_one_response()
    experiment_division_by_zero_recovery()
    experiment_empty_response_recovery()
    print("\nAll multiple-tools experiments passed.")
