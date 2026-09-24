"""无需 Ollama 服务即可检查 AgentState 和 AgentStep。

从项目根目录运行：

    python experiments/trajectory_experiment.py

这些实验展示正式对话消息与完整运行轨迹为何需要使用不同的数据结构。
"""

from collections.abc import Iterable
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import AgentLoop, AgentRunResult, AgentStep, InputMessage
from llm.base import LLM, Message, ModelResponse, ToolCall
from runtime import ToolErrorType, ToolExecutor
from tools import ToolRegistry, calculator_tools


class ScriptedLLM(LLM):
    """返回预先准备的响应，使轨迹行为可重复验证。"""

    def __init__(self, responses: Iterable[ModelResponse]) -> None:
        self._responses = iter(responses)

    def chat(self, messages, tools=None) -> ModelResponse:
        return next(self._responses)


def create_executor() -> ToolExecutor:
    registry = ToolRegistry()

    for tool in calculator_tools:
        registry.register(tool)

    return ToolExecutor(registry)


def print_trajectory(title: str, result: AgentRunResult) -> None:
    print(f"\n=== {title} ===")
    print(
        f"task={result.state.current_task!r}, "
        f"stop_reason={result.stop_reason.value}"
    )

    for entry in result.trajectory:
        if isinstance(entry, InputMessage):
            message = entry.message
            print(f"input role={message.role}, content={message.content!r}")
            continue

        assert isinstance(entry, AgentStep)
        response = entry.model_response
        tool_names = [
            execution.result.tool_name
            for execution in entry.tool_executions
        ]
        print(
            f"step={entry.step}, content={response.content!r}, "
            f"thinking={response.thinking!r}, tools={tool_names}"
        )


def experiment_completed_trajectory() -> None:
    first_response = ModelResponse(tool_calls=[
        ToolCall(name="add", arguments={"a": 12, "b": 7}),
    ])
    final_response = ModelResponse(content="12 + 7 is 19.")
    result = AgentLoop(
        ScriptedLLM([first_response, final_response]),
        create_executor(),
    ).run([
        Message(role="system", content="Use calculator tools."),
        Message(role="user", content="Calculate 12 + 7."),
    ])

    assert result.completed
    assert result.state.current_task == "Calculate 12 + 7."
    assert result.state.step == 2
    assert len(result.trajectory) == 4
    assert len(result.agent_steps) == 2
    assert result.agent_steps[0].model_response is first_response
    assert result.agent_steps[0].tool_results == [result.tool_results[0]]
    assert result.agent_steps[0].tool_executions[0].call is (
        first_response.tool_calls[0]
    )
    assert result.agent_steps[1].model_response is final_response
    assert result.agent_steps[1].tool_results == []
    print_trajectory("completed run", result)


def experiment_thinking_only_response() -> None:
    empty_response = ModelResponse()
    thinking_only = ModelResponse(
        thinking="I have the observation but have not answered yet.",
        completion_tokens=10,
        done_reason="stop",
    )
    result = AgentLoop(
        ScriptedLLM([
            empty_response,
            thinking_only,
            ModelResponse(content="The final answer is 19."),
        ]),
        create_executor(),
    ).run([
        Message(role="user", content="Return the answer."),
    ])

    assert result.completed
    assert result.agent_steps[0].model_response is empty_response
    assert result.agent_steps[1].model_response is thinking_only
    assert result.agent_steps[1].model_response.thinking
    assert len(result.messages) == 2
    assert all(message.content for message in result.messages)
    print_trajectory("empty and thinking-only responses", result)


def experiment_tool_failure_trajectory() -> None:
    result = AgentLoop(
        ScriptedLLM([
            ModelResponse(tool_calls=[
                ToolCall(name="divide", arguments={"a": 10, "b": 0}),
            ]),
            ModelResponse(content="Division by zero is undefined."),
        ]),
        create_executor(),
    ).run([
        Message(role="user", content="Calculate 10 / 0."),
    ])

    failed_result = result.agent_steps[0].tool_results[0]
    assert not failed_result.success
    assert failed_result.error_type is ToolErrorType.EXECUTION_ERROR
    assert failed_result is result.tool_results[0]
    assert result.messages[-2].content.startswith(
        "Error [execution_error]:"
    )
    print_trajectory("tool failure", result)


def experiment_max_steps_trajectory() -> None:
    responses = [
        ModelResponse(tool_calls=[
            ToolCall(name="multiply", arguments={"a": 2, "b": 3}),
        ]),
        ModelResponse(thinking="I will repeat the same action."),
    ]
    result = AgentLoop(
        ScriptedLLM(responses),
        create_executor(),
        max_steps=2,
    ).run([
        Message(role="user", content="Keep working forever."),
    ])

    assert not result.completed
    assert result.steps == 2
    assert len(result.agent_steps) == 2
    assert result.agent_steps[1].model_response.thinking
    assert len(result.messages) == 3
    print_trajectory("max_steps", result)


def experiment_provider_failure_state() -> None:
    class FailingLLM(LLM):
        def chat(self, messages, tools=None) -> ModelResponse:
            raise RuntimeError("provider unavailable")

    agent = AgentLoop(FailingLLM(), create_executor())

    try:
        agent.run([
            Message(role="user", content="Try to answer."),
        ])
    except RuntimeError as exc:
        assert str(exc) == "provider unavailable"
    else:
        raise AssertionError("The provider error must be re-raised.")

    assert agent.last_state is not None
    assert agent.last_state.step == 1
    assert agent.last_state.steps == []
    assert agent.last_state.current_task == "Try to answer."
    print("\n=== provider failure ===")
    print("attempted_steps=1, recorded_responses=0")


if __name__ == "__main__":
    experiment_completed_trajectory()
    experiment_thinking_only_response()
    experiment_tool_failure_trajectory()
    experiment_max_steps_trajectory()
    experiment_provider_failure_state()
    print("\nAll trajectory experiments passed.")
