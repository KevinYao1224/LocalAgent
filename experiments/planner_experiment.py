"""Phase 9 离线实验：.venv/bin/python experiments/planner_experiment.py"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import CalculatorPlanner, PlanStopReason
from evaluation import evaluate
from evaluation.cases import ScriptedLLM
from evaluation.runner import EvaluationCase, ExpectedCall, EvaluationReport
from llm.base import ModelResponse, ToolCall
from runtime import ToolExecutor
from tools import ToolRegistry, calculator_tools


def make_executor() -> ToolExecutor:
    registry = ToolRegistry()
    for tool in calculator_tools:
        registry.register(tool)
    return ToolExecutor(registry)


def scripted_plan(steps: list[dict]) -> ScriptedLLM:
    return ScriptedLLM((ModelResponse(content=json.dumps({"steps": steps})),))


def run_case(label: str, task: str, steps: list[dict],
             expected: PlanStopReason, answer: int | float | None,
             executed: int) -> None:
    planner = CalculatorPlanner(scripted_plan(steps), make_executor())
    result = planner.run(task)
    print(f"\n=== {label} ===")
    print(result.render())
    assert result.stop_reason is expected
    assert result.answer == answer
    assert len(result.state.executions) == executed


def main() -> None:
    # 用 Phase 13 的已有 runner 校验原 AgentLoop 的脚本化基线。
    baseline = evaluate(
        EvaluationCase(
            "baseline-two-tools", "Compute (12 + 7) * 5", "completed",
            (ExpectedCall("add", {"a": 12, "b": 7}),
             ExpectedCall("multiply", {"a": 19, "b": 5})), "95",
        ),
        ScriptedLLM((
            ModelResponse(tool_calls=[ToolCall("add", {"a": 12, "b": 7})]),
            ModelResponse(tool_calls=[ToolCall("multiply", {"a": 19, "b": 5})]),
            ModelResponse(content="95"),
        )),
    )
    print("=== Phase 13 scripted baseline (not model quality) ===")
    print(EvaluationReport((baseline,)).render())
    assert baseline.passed

    run_case("dependent arithmetic", "Compute (12 + 7) * 5", [
        {"tool": "add", "arguments": {"a": 12, "b": 7}},
        {"tool": "multiply", "arguments": {"a": {"from_step": 1}, "b": 5}},
    ], PlanStopReason.COMPLETED, 95, 2)
    run_case("preflight rejects future dependency; no tools executed", "Bad plan", [
        {"tool": "add", "arguments": {"a": 1, "b": 2}},
        {"tool": "multiply", "arguments": {"a": {"from_step": 3}, "b": 5}},
    ], PlanStopReason.INVALID_PLAN, None, 0)
    run_case("division by zero halts remaining steps", "Bad calculation", [
        {"tool": "add", "arguments": {"a": 1, "b": 2}},
        {"tool": "divide", "arguments": {"a": {"from_step": 1}, "b": 0}},
        {"tool": "multiply", "arguments": {"a": 3, "b": 5}},
    ], PlanStopReason.TOOL_FAILED, None, 2)

    for label, raw in (
        ("unknown tool", '{"steps":[{"tool":"shell","arguments":{"a":1,"b":2}}]}'),
        ("not JSON", "```json"),
        ("duplicate field", '{"steps":[],"steps":[]}'),
        ("NaN", '{"steps":[{"tool":"add","arguments":{"a":NaN,"b":2}}]}'),
        ("boolean", '{"steps":[{"tool":"add","arguments":{"a":true,"b":2}}]}'),
        ("step limit", json.dumps({"steps": [
            {"tool": "add", "arguments": {"a": 1, "b": 2}},
            {"tool": "add", "arguments": {"a": 1, "b": 2}},
        ]})),
    ):
        planner = CalculatorPlanner(ScriptedLLM((ModelResponse(content=raw),)),
                                    make_executor(), max_plan_steps=1)
        result = planner.run(label)
        assert result.stop_reason is PlanStopReason.INVALID_PLAN
        assert result.state.executions == []
        print(f"[PASS] {label}: {result.error}")

    later = CalculatorPlanner(ScriptedLLM((ModelResponse(content=json.dumps({"steps": [
        {"tool": "add", "arguments": {"a": 1, "b": 2}},
        {"tool": "divide", "arguments": {"a": "oops", "b": 2}},
    ]})),)), make_executor(), max_plan_steps=2).run("Invalid later step")
    assert later.stop_reason is PlanStopReason.INVALID_PLAN
    assert not later.state.executions and "Step 2" in later.error
    print(f"[PASS] whole-plan preflight: {later.error}; no earlier tool executed")

    executor = make_executor()
    executor.disable_tool("multiply")
    planner = CalculatorPlanner(scripted_plan([
        {"tool": "multiply", "arguments": {"a": 3, "b": 5}},
    ]), executor)
    result = planner.run("Unavailable calculator")
    assert result.stop_reason is PlanStopReason.INVALID_PLAN and not result.state.executions
    print(f"[PASS] disabled capability before planning: {result.error}")

    no_tools = make_executor()
    no_tools.set_enabled_tools([])
    planner = CalculatorPlanner(ScriptedLLM(()), no_tools)
    assert planner.run("No tools").stop_reason is PlanStopReason.INVALID_PLAN
    print("[PASS] no tools: model not called")

    planner = CalculatorPlanner(ScriptedLLM((ModelResponse(
        content='{"steps":[]}', tool_calls=[ToolCall("add", {"a": 1, "b": 2})],
    ),)), make_executor())
    assert planner.run("Unexpected tool call").stop_reason is PlanStopReason.INVALID_PLAN
    print("[PASS] model tool call refused")

    run_case("non-finite arithmetic result", "Overflow", [
        {"tool": "multiply", "arguments": {"a": 1e308, "b": 1e308}},
    ], PlanStopReason.TOOL_FAILED, None, 1)

    wrong = CalculatorPlanner(scripted_plan([
        {"tool": "add", "arguments": {"a": 12, "b": 8}},
    ]), make_executor()).run("Compute 12 + 7")
    assert wrong.completed and wrong.answer == 20
    print("[LIMITATION] well-formed but wrong plan is not semantic task verification; "
          "an external expected-answer evaluation must catch it.")

    class RevokingExecutor(ToolExecutor):
        def execute(self, call):
            if call.name == "multiply":
                self.disable_tool("multiply")
            return super().execute(call)

    registry = ToolRegistry()
    for tool in calculator_tools:
        registry.register(tool)
    planner = CalculatorPlanner(scripted_plan([
        {"tool": "add", "arguments": {"a": 1, "b": 2}},
        {"tool": "multiply", "arguments": {"a": {"from_step": 1}, "b": 5}},
    ]), RevokingExecutor(registry))
    result = planner.run("Capability revoked during execution")
    assert result.stop_reason is PlanStopReason.TOOL_FAILED
    assert len(result.state.executions) == 2
    assert not result.state.executions[1].result.success
    print("[PASS] capability checked again at execution: " + result.render())
    print("\nAll Phase 9 offline checks passed. Scripted cases do not measure model quality.")


if __name__ == "__main__":
    main()
