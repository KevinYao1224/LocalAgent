"""Deterministic experiments for understanding AgentLoop.

Run from the project root:

    python experiments/agent_loop_experiment.py

No Ollama server is needed. ScriptedLLM returns prepared model responses so
each experiment isolates one control-flow behavior of the runtime.
"""

from collections.abc import Iterable
from pathlib import Path
import sys

# Allow this file to be launched directly from the experiments directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.loop import AgentLoop
from llm.base import LLM, Message, ModelResponse, ToolCall
from tools.calculator import multiply_tool
from tools.registry import ToolRegistry


class ScriptedLLM(LLM):
    """A predictable LLM replacement used only by these experiments."""

    def __init__(self, responses: Iterable[ModelResponse]) -> None:
        self._responses = iter(responses)

    def chat(self, messages, tools=None) -> ModelResponse:
        return next(self._responses)


def create_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(multiply_tool)
    return registry


def print_trace(title: str, result) -> None:
    print(f"\n=== {title} ===")
    print(f"stop_reason={result.stop_reason.value}, steps={result.steps}")
    for message in result.messages:
        calls = [call.name for call in message.tool_calls]
        extra = f", tool_calls={calls}" if calls else ""
        print(
            f"{message.role:>9}: {message.content!r}"
            f"{extra}"
        )


def experiment_successful_tool_call() -> None:
    llm = ScriptedLLM([
        ModelResponse(tool_calls=[
            ToolCall(name="multiply", arguments={"a": 23, "b": 47}),
        ]),
        ModelResponse(content="23 multiplied by 47 is 1081."),
    ])
    result = AgentLoop(llm, create_registry()).run([
        Message(role="user", content="What is 23 multiplied by 47?"),
    ])

    assert result.completed
    assert result.messages[-2].content == "1081"
    print_trace("successful tool call", result)


def experiment_unknown_tool_recovery() -> None:
    llm = ScriptedLLM([
        ModelResponse(tool_calls=[
            ToolCall(name="missing_tool", arguments={}),
        ]),
        ModelResponse(content="I could not use the requested tool."),
    ])
    result = AgentLoop(llm, create_registry()).run([
        Message(role="user", content="Call a tool that is unavailable."),
    ])

    assert result.completed
    assert result.messages[-2].content.startswith("Error:")
    print_trace("unknown tool becomes an observation", result)


def experiment_invalid_arguments_recovery() -> None:
    llm = ScriptedLLM([
        ModelResponse(tool_calls=[
            ToolCall(name="multiply", arguments={"a": 23}),
        ]),
        ModelResponse(content="The tool call was missing an argument."),
    ])
    result = AgentLoop(llm, create_registry()).run([
        Message(role="user", content="Multiply 23 by an unspecified value."),
    ])

    assert result.completed
    assert "Invalid arguments" in result.messages[-2].content
    print_trace("tool error becomes an observation", result)


def experiment_step_limit() -> None:
    repeated_call = ModelResponse(tool_calls=[
        ToolCall(name="multiply", arguments={"a": 2, "b": 3}),
    ])
    llm = ScriptedLLM([repeated_call, repeated_call])
    result = AgentLoop(
        llm,
        create_registry(),
        max_steps=2,
    ).run([
        Message(role="user", content="Keep calling multiply forever."),
    ])

    assert not result.completed
    assert result.steps == 2
    print_trace("max_steps stops a repeated tool loop", result)


if __name__ == "__main__":
    experiment_successful_tool_call()
    experiment_unknown_tool_recovery()
    experiment_invalid_arguments_recovery()
    experiment_step_limit()
    print("\nAll AgentLoop experiments passed.")
