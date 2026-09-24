"""无需 Ollama，检查 run/step 关联关系和单调时钟耗时。

运行：.venv/bin/python experiments/structured_trace_timing_experiment.py
"""

from datetime import timezone
from io import StringIO
from pathlib import Path
import sys
from time import sleep

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import AgentLoop
from llm.base import LLM, Message, ModelResponse, ToolCall
from observability import HumanReadableLogger
from observability.events import (
    AgentFinished,
    AgentStarted,
    ModelCallFailed,
    ModelCallStarted,
    ModelResponseReceived,
    ToolExecutionFailed,
    ToolExecutionFinished,
    ToolExecutionStarted,
)
from runtime import ToolExecutor
from tools import Tool, ToolRegistry


class CollectingLogger:
    def __init__(self) -> None:
        self.events = []

    def log(self, event) -> None:
        self.events.append(event)


class FanOutLogger:
    def __init__(self, *loggers) -> None:
        self.loggers = loggers

    def log(self, event) -> None:
        for logger in self.loggers:
            logger.log(event)


class ScriptedLLM(LLM):
    def __init__(self, responses) -> None:
        self.responses = iter(responses)

    def chat(self, messages, tools=None) -> ModelResponse:
        sleep(0.01)  # 不依赖远程模型，也能观察到耗时。
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def executor(handler=lambda: 12) -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(Tool(
        name="measure",
        description="A small instrumented tool",
        parameters={"type": "object", "properties": {}},
        handler=handler,
    ))
    return ToolExecutor(registry)


def check_events(events, run_id: str) -> None:
    assert isinstance(events[0], AgentStarted)
    assert isinstance(events[-1], AgentFinished)
    for event in events:
        assert event.run_id == run_id
        assert event.timestamp is not None
        assert event.timestamp.tzinfo == timezone.utc
        if event.duration_ms is not None:
            assert event.duration_ms >= 0
    assert events[-1].duration_ms is not None


def successful_run() -> None:
    events = CollectingLogger()
    text = StringIO()

    def slow_tool():
        sleep(0.01)
        return 12

    loop = AgentLoop(
        ScriptedLLM([
            ModelResponse(tool_calls=[ToolCall("measure", {})], prompt_tokens=8),
            ModelResponse(content="12", prompt_tokens=13, completion_tokens=2),
            ModelResponse(content="again"),
        ]), executor(slow_tool),
        logger=FanOutLogger(events, HumanReadableLogger(stream=text)),
    )
    result = loop.run([Message(role="user", content="Use the tool")])
    check_events(events.events, result.run_id)
    assert result.completed and result.steps == 2
    assert [step.step_id for step in result.agent_steps] == [
        f"{result.run_id}:1", f"{result.run_id}:2"
    ]
    for event in events.events:
        if hasattr(event, "step"):
            assert event.step_id == result.state.step_id(event.step)
        else:
            assert event.step_id is None
    calls = [event for event in events.events if isinstance(event, ModelCallStarted)]
    responses = [event for event in events.events if isinstance(event, ModelResponseReceived)]
    tool_start = next(e for e in events.events if isinstance(e, ToolExecutionStarted))
    tool_end = next(e for e in events.events if isinstance(e, ToolExecutionFinished))
    assert len(calls) == len(responses) == 2
    assert all(e.duration_ms >= 5 for e in responses)
    assert tool_start.step_id == tool_end.step_id == calls[0].step_id
    assert tool_end.duration_ms >= 5
    assert responses[0].response.prompt_tokens == 8
    assert responses[1].response.completion_tokens == 2
    assert result.messages[-1].content == "12"
    assert f"Run ID: {result.run_id}" in text.getvalue()
    assert "Duration:" in text.getvalue()
    print("\n=== completed run ===")
    print(text.getvalue())

    second = loop.run([Message(role="user", content="A second run")])
    assert second.run_id != result.run_id
    assert second.agent_steps[0].step_id == f"{second.run_id}:1"


def failed_runs() -> None:
    model_events = CollectingLogger()
    agent = AgentLoop(
        ScriptedLLM([RuntimeError("model unavailable")]), executor(),
        logger=model_events,
    )
    try:
        agent.run([Message(role="user", content="Fail at model")])
    except RuntimeError:
        pass
    else:
        raise AssertionError("Model error should propagate")
    state = agent.last_state
    assert state is not None and state.steps == []
    check_events(model_events.events, state.run_id)
    failure = next(e for e in model_events.events if isinstance(e, ModelCallFailed))
    assert failure.step_id == state.step_id(1) and failure.duration_ms >= 5
    assert model_events.events[-1].stop_reason == "error"
    print("=== model failure ===")
    print(f"run={state.run_id}, attempt={failure.step_id}, elapsed={failure.duration_ms:.2f} ms")

        # ToolExecutor 会将普通 handler 异常转换为 ToolResult；下面用失败的 executor
        # 模拟 Runtime 层异常。
    class FailingExecutor:
        def available_tools(self):
            return []

        def execute(self, call):
            raise RuntimeError("executor failed")

    tool_events = CollectingLogger()
    tool_agent = AgentLoop(
        ScriptedLLM([ModelResponse(tool_calls=[ToolCall("measure", {})])]),
        FailingExecutor(), logger=tool_events,
    )
    try:
        tool_agent.run([Message(role="user", content="Fail at tool")])
    except RuntimeError:
        pass
    else:
        raise AssertionError("Executor error should propagate")
    assert tool_agent.last_state is not None
    check_events(tool_events.events, tool_agent.last_state.run_id)
    failure = next(e for e in tool_events.events if isinstance(e, ToolExecutionFailed))
    assert failure.step_id == tool_agent.last_state.step_id(1)
    assert failure.duration_ms >= 0
    print("=== executor failure ===")
    print(f"run={failure.run_id}, step={failure.step_id}, elapsed={failure.duration_ms:.2f} ms")


def step_limit() -> None:
    events = CollectingLogger()
    result = AgentLoop(
        ScriptedLLM([ModelResponse(thinking="not finished")]),
        executor(), max_steps=1, logger=events,
    ).run([Message(role="user", content="Keep going")])
    check_events(events.events, result.run_id)
    assert not result.completed
    assert events.events[-1].stop_reason == "max_steps"
    assert result.agent_steps[0].step_id == f"{result.run_id}:1"
    print("=== step limit ===")
    print(f"run={result.run_id}, stop={events.events[-1].stop_reason}")


if __name__ == "__main__":
    successful_run()
    failed_runs()
    step_limit()
    print("All trace timing experiments passed.")
