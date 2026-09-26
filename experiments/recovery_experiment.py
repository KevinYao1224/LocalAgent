"""Phase 10：离线观察有界纠正、终止条件和工具执行事实。

从仓库根目录运行：.venv/bin/python experiments/recovery_experiment.py
不需要 Ollama；以下脚本模型只验证控制流，不代表真实模型的纠错能力。
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.loop import AgentLoop, StopReason
from agent.session import AgentSession
from evaluation import EvaluationCase, ExpectedCall, evaluate
from llm.base import LLM, Message, ModelResponse, ToolCall
from observability.events import (
    AgentFinished, ModelCallFailed, RecoveryDecision, ToolExecutionFailed,
)
from observability.jsonl import event_record
from observability.logger import CompositeLogger, HumanReadableLogger
from runtime import RecoveryPolicy, ToolErrorType, ToolExecutor
from runtime.recovery import SELF_CRITIQUE_PROMPT
from tools.base import Tool
from tools.registry import ToolRegistry


class ScriptedLLM(LLM):
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0
        self.inputs = []

    def chat(self, messages, tools=None):
        self.calls += 1
        self.inputs.append((list(messages), [tool.name for tool in tools or ()]))
        return next(self.responses)


class CollectLogger:
    def __init__(self):
        self.events = []

    def log(self, event):
        self.events.append(event)


def create_executor(counter):
    registry = ToolRegistry()

    def multiply(a, b):
        counter.append((a, b))
        return a * b

    def fails_after_work():
        counter.append("side effect")
        raise RuntimeError("handler failed after work")

    schema = {
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
        "additionalProperties": False,
    }
    registry.register(Tool("multiply", "Multiply numbers", schema, multiply))
    registry.register(Tool("fails_after_work", "Demonstrate failure", {
        "type": "object", "properties": {}, "additionalProperties": False,
    }, fails_after_work))
    return ToolExecutor(registry)


def call(name, **arguments):
    return ModelResponse(tool_calls=[ToolCall(name=name, arguments=arguments)])


def run_case(title, responses, *, policy, max_steps=5, restrict=False):
    print(f"\n=== {title} ===")
    counter = []
    executor = create_executor(counter)
    if restrict:
        executor.disable_tool("multiply")
    model = ScriptedLLM(responses)
    events = CollectLogger()
    agent = AgentLoop(model, executor, max_steps=max_steps, recovery_policy=policy,
                      logger=CompositeLogger(HumanReadableLogger(), events))
    result = agent.run([Message(role="user", content="Compute 2 * 3")])
    assert result.metrics is not None
    assert result.metrics.tool_errors == sum(not item.success for item in result.tool_results)
    assert len([event for event in events.events if isinstance(event, AgentFinished)]) == 1
    assert all(event.run_id == result.run_id for event in events.events)
    for decision in (event for event in events.events if isinstance(event, RecoveryDecision)):
        record = event_record(decision)
        assert record["data"]["decision"] == decision.decision
        assert "arguments" not in str(record)
    return result, model.calls, counter, events.events


def main():
    for value in (-1, True, 1.5):
        try:
            RecoveryPolicy(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid recovery budget accepted: {value}")
    try:
        RecoveryPolicy(self_critique="yes")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid self_critique accepted")

    bad = call("multiply", a=2)
    good = call("multiply", a=2, b=3)
    result, calls, effects, events = run_case(
        "参数错误 → 修正一次 → 回答", [bad, good, ModelResponse(content="6")],
        policy=RecoveryPolicy(1),
    )
    assert result.completed and calls == 3 and effects == [(2, 3)]
    assert [event.decision for event in events if isinstance(event, RecoveryDecision)] == ["continue"]
    assert result.messages[2].content.startswith("Error [validation_error]:")

    scored = evaluate(
        EvaluationCase(
            "bounded-repair", "Compute 2 * 3", "completed",
            expected_calls=(
                ExpectedCall("multiply", {"a": 2}),
                ExpectedCall("multiply", {"a": 2, "b": 3}),
            ),
            expected_answer="6", expected_tool_errors=1,
        ),
        ScriptedLLM([bad, good, ModelResponse(content="6")]),
        recovery_policy=RecoveryPolicy(1),
    )
    assert scored.passed and scored.steps == 3

    print("\n=== 自检提示只加入纠正步骤，不污染轨迹与短期记忆 ===")
    critic_model = ScriptedLLM([bad, good, ModelResponse(content="6")])
    critic_events = CollectLogger()
    critic_agent = AgentLoop(
        critic_model, create_executor([]), recovery_policy=RecoveryPolicy(
            max_corrections=1, self_critique=True,
        ), logger=CompositeLogger(HumanReadableLogger(), critic_events),
    )
    session = AgentSession(critic_agent)
    critic_result = session.run("Compute 2 * 3")
    assert critic_result.completed and session.memory.turn_count == 1
    assert len(critic_model.inputs) == 3
    assert SELF_CRITIQUE_PROMPT not in [m.content for m in critic_model.inputs[0][0]]
    assert critic_model.inputs[1][0][-1].content == SELF_CRITIQUE_PROMPT
    assert critic_model.inputs[1][0][-1].role == "user"
    assert critic_model.inputs[1][1] == critic_model.inputs[0][1]
    assert SELF_CRITIQUE_PROMPT not in [m.content for m in critic_model.inputs[2][0]]
    assert SELF_CRITIQUE_PROMPT not in [m.content for m in critic_result.messages]
    assert SELF_CRITIQUE_PROMPT not in [m.content for m in session.memory.retrieve()]
    decision = next(e for e in critic_events.events if isinstance(e, RecoveryDecision))
    assert decision.self_critique_next_step
    assert event_record(decision)["data"]["self_critique_next_step"]

    result, calls, effects, _ = run_case(
        "未知工具 → 选择现有工具", [call("missing"), good, ModelResponse(content="6")],
        policy=RecoveryPolicy(1),
    )
    assert result.completed and calls == 3 and effects == [(2, 3)]

    result, calls, effects, decisions = run_case(
        "一批两次参数错误只占一个纠正轮次", [ModelResponse(tool_calls=[
            ToolCall("multiply", {"a": 2}), ToolCall("multiply", {"b": 3}),
        ]), good, ModelResponse(content="6")], policy=RecoveryPolicy(1),
    )
    assert result.completed and calls == 3 and effects == [(2, 3)]
    assert [(event.error_types, event.corrections_used) for event in decisions
            if isinstance(event, RecoveryDecision)] == [
                (("validation_error", "validation_error"), 1),
            ]

    result, calls, effects, _ = run_case(
        "预算用尽 → 不再询问模型", [bad, bad, good], policy=RecoveryPolicy(1),
    )
    assert result.stop_reason is StopReason.RECOVERY_EXHAUSTED
    assert calls == 2 and effects == [] and len(result.tool_results) == 2

    result, calls, effects, _ = run_case(
        "零次纠正", [bad, good], policy=RecoveryPolicy(0),
    )
    assert result.stop_reason is StopReason.RECOVERY_EXHAUSTED and calls == 1

    result, calls, effects, _ = run_case(
        "能力拒绝不能通过纠正绕过", [good, ModelResponse(content="6")],
        policy=RecoveryPolicy(3), restrict=True,
    )
    assert result.stop_reason is StopReason.UNRECOVERABLE_TOOL_ERROR
    assert result.tool_results[0].error_type is ToolErrorType.TOOL_NOT_AVAILABLE
    assert calls == 1 and effects == []

    result, calls, effects, _ = run_case(
        "handler 失败不自动重放", [call("fails_after_work"), call("fails_after_work")],
        policy=RecoveryPolicy(3),
    )
    assert result.stop_reason is StopReason.UNRECOVERABLE_TOOL_ERROR
    assert calls == 1 and effects == ["side effect"]

    result, calls, effects, _ = run_case(
        "同批调用全部执行，终止前保留各自结果", [ModelResponse(tool_calls=[
            ToolCall("multiply", {"a": 2, "b": 3}), ToolCall("fails_after_work", {}),
        ]), ModelResponse(content="6")], policy=RecoveryPolicy(1),
    )
    assert result.stop_reason is StopReason.UNRECOVERABLE_TOOL_ERROR
    assert calls == 1 and effects == [(2, 3), "side effect"]
    assert len(result.agent_steps[0].tool_executions) == 2

    result, calls, _, events = run_case(
        "步骤上限仍生效", [bad, good], policy=RecoveryPolicy(2), max_steps=1,
    )
    assert result.stop_reason is StopReason.MAX_STEPS and calls == 1
    assert [event.decision for event in events if isinstance(event, RecoveryDecision)] == ["max_steps"]

    result, calls, _, events = run_case(
        "默认策略保持旧行为", [bad, bad, ModelResponse(content="stop")], policy=None,
    )
    assert result.completed and calls == 3
    assert not any(isinstance(event, RecoveryDecision) for event in events)

    print("\n=== 模型/Runtime 异常不自动重试 ===")

    class FailingLLM(LLM):
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None):
            self.calls += 1
            raise RuntimeError("model request status is unknown")

    failing_model = FailingLLM()
    model_events = CollectLogger()
    try:
        AgentLoop(
            failing_model, create_executor([]), recovery_policy=RecoveryPolicy(3, True),
            logger=model_events,
        ).run([Message(role="user", content="test")])
    except RuntimeError:
        pass
    else:
        raise AssertionError("Model exception was hidden")
    assert failing_model.calls == 1
    assert len([e for e in model_events.events if isinstance(e, ModelCallFailed)]) == 1
    assert not any(isinstance(e, RecoveryDecision) for e in model_events.events)

    class CrashingExecutor(ToolExecutor):
        def __init__(self):
            super().__init__(ToolRegistry())
            self.calls = 0

        def execute(self, tool_call):
            self.calls += 1
            raise RuntimeError("unknown execution state")

    crashing_executor = CrashingExecutor()
    tool_model = ScriptedLLM([good, good])
    tool_events = CollectLogger()
    try:
        AgentLoop(
            tool_model, crashing_executor, recovery_policy=RecoveryPolicy(3, True),
            logger=tool_events,
        ).run([Message(role="user", content="test")])
    except RuntimeError:
        pass
    else:
        raise AssertionError("Runtime exception was hidden")
    assert crashing_executor.calls == tool_model.calls == 1
    assert len([e for e in tool_events.events if isinstance(e, ToolExecutionFailed)]) == 1
    assert not any(isinstance(e, RecoveryDecision) for e in tool_events.events)

    # Session 只能提交真正完成的轮次；终止的轮次不进入历史。
    agent = AgentLoop(ScriptedLLM([bad]), create_executor([]), recovery_policy=RecoveryPolicy(0))
    session = AgentSession(agent)
    assert not session.run("Try multiply").completed
    assert session.memory.turn_count == 0
    print("\n所有有界恢复实验通过。")


if __name__ == "__main__":
    main()
