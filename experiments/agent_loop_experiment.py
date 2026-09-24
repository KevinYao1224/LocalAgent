"""用于理解 AgentLoop 的确定性实验。

从项目根目录运行：

    python experiments/agent_loop_experiment.py

无需 Ollama 服务。ScriptedLLM 返回预先准备好的模型响应，使每个实验都能单独观察
Runtime 的一种控制流行为。
"""

from collections.abc import Iterable
from pathlib import Path
import sys

# 允许从 experiments 目录直接启动此文件。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.loop import AgentLoop
from llm.base import LLM, Message, ModelResponse, ToolCall
from runtime.executor import ToolExecutor
from tools.calculator import multiply_tool
from tools.registry import ToolRegistry


class ScriptedLLM(LLM):
    """仅供这些实验使用、响应可预测的 LLM 替代实现。"""

    def __init__(self, responses: Iterable[ModelResponse]) -> None:
        self._responses = iter(responses)

    def chat(self, messages, tools=None) -> ModelResponse:
        return next(self._responses)


def create_executor() -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(multiply_tool)
    return ToolExecutor(registry)


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
    result = AgentLoop(llm, create_executor()).run([
        Message(role="user", content="What is 23 multiplied by 47?"),
    ])

    assert result.completed
    assert result.tool_results[0].success
    assert result.tool_results[0].value == 1081
    assert result.messages[-2].content == "1081"
    print_trace("successful tool call", result)


def experiment_unknown_tool_recovery() -> None:
    llm = ScriptedLLM([
        ModelResponse(tool_calls=[
            ToolCall(name="missing_tool", arguments={}),
        ]),
        ModelResponse(content="I could not use the requested tool."),
    ])
    result = AgentLoop(llm, create_executor()).run([
        Message(role="user", content="Call a tool that is unavailable."),
    ])

    assert result.completed
    assert not result.tool_results[0].success
    assert result.messages[-2].content.startswith(
        "Error [tool_not_found]:"
    )
    print_trace("unknown tool becomes an observation", result)


def experiment_invalid_arguments_recovery() -> None:
    llm = ScriptedLLM([
        ModelResponse(tool_calls=[
            ToolCall(name="multiply", arguments={"a": 23}),
        ]),
        ModelResponse(content="The tool call was missing an argument."),
    ])
    result = AgentLoop(llm, create_executor()).run([
        Message(role="user", content="Multiply 23 by an unspecified value."),
    ])

    assert result.completed
    assert not result.tool_results[0].success
    assert "Invalid arguments" in result.messages[-2].content
    print_trace("tool error becomes an observation", result)


def experiment_step_limit() -> None:
    repeated_call = ModelResponse(tool_calls=[
        ToolCall(name="multiply", arguments={"a": 2, "b": 3}),
    ])
    llm = ScriptedLLM([repeated_call, repeated_call])
    result = AgentLoop(
        llm,
        create_executor(),
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
