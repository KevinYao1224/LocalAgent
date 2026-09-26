"""Phase 9：仅针对计算器的显式 plan-then-execute 路径。"""

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter
from typing import Any
from uuid import uuid4

from llm.base import LLM, Message, ToolCall
from runtime.executor import ToolExecutor
from runtime.result import ToolResult
from runtime.validation import ToolArgumentsValidationError, ToolArgumentsValidator
from tools.base import Tool


CALCULATOR_NAMES = frozenset({"add", "subtract", "multiply", "divide"})


class PlanValidationError(ValueError):
    """模型产出的计划不符合本阶段的受限协议。"""


@dataclass(frozen=True, slots=True)
class StepReference:
    """引用之前步骤的数值结果；不是字符串插值或任意表达式。"""

    from_step: int


@dataclass(frozen=True, slots=True)
class PlanStep:
    number: int
    tool: str
    arguments: dict[str, int | float | StepReference]


@dataclass(frozen=True, slots=True)
class Plan:
    steps: tuple[PlanStep, ...]


@dataclass(frozen=True, slots=True)
class PlanExecution:
    step: PlanStep
    call: ToolCall
    result: ToolResult
    verified: bool
    error: str | None
    duration_ms: float


@dataclass(slots=True)
class PlanState:
    """本次规划和逐步执行的事实；不会写进 AgentLoop 的对话轨迹。"""

    task: str
    run_id: str = field(default_factory=lambda: uuid4().hex)
    plan: Plan | None = None
    executions: list[PlanExecution] = field(default_factory=list)


class PlanStopReason(str, Enum):
    COMPLETED = "completed"
    INVALID_PLAN = "invalid_plan"
    TOOL_FAILED = "tool_failed"


@dataclass(frozen=True, slots=True)
class PlanRunResult:
    state: PlanState
    stop_reason: PlanStopReason
    answer: int | float | None
    error: str | None
    duration_ms: float

    @property
    def completed(self) -> bool:
        return self.stop_reason is PlanStopReason.COMPLETED

    def render(self) -> str:
        """便于离线实验阅读的逐步状态；不是 JSONL checkpoint。"""

        lines = [f"Plan run: {self.state.run_id}", f"Task: {self.state.task}"]
        if self.state.plan is not None:
            for step in self.state.plan.steps:
                execution = next(
                    (item for item in self.state.executions if item.step.number == step.number),
                    None,
                )
                status = (
                    "pending" if execution is None
                    else "verified" if execution.verified else "failed"
                )
                line = f"  {step.number}. [{status}] {step.tool}({step.arguments})"
                if execution is not None:
                    line += (f" [actual: {execution.call.arguments}]"
                             f" -> {execution.result.to_message_content()}"
                             f" ({execution.duration_ms:.2f} ms)")
                lines.append(line)
        lines.append(f"Stop: {self.stop_reason.value}; answer: {self.answer!r}")
        if self.error is not None:
            lines.append(f"Reason: {self.error}")
        lines.append(f"Duration: {self.duration_ms:.2f} ms")
        return "\n".join(lines)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """拒绝重复 JSON 字段，避免校验和实际使用的值产生歧义。"""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PlanValidationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise PlanValidationError(f"Non-finite JSON number: {value}")


def _finite_number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # 超大 JSON 整数不能安全转换为浮点数
        return False


def parse_plan(text: str, tools: list[Tool], max_plan_steps: int) -> Plan:
    """在产生任何副作用前验证整个计划、依赖和当前可用工具。"""

    try:
        data = json.loads(
            text, object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (ValueError, TypeError) as exc:
        raise PlanValidationError(f"Invalid plan JSON: {exc}") from exc
    if not isinstance(data, dict) or set(data) != {"steps"} or not isinstance(data["steps"], list):
        raise PlanValidationError("Plan must be an object containing only a steps array.")
    raw_steps = data["steps"]
    if not 1 <= len(raw_steps) <= max_plan_steps:
        raise PlanValidationError(f"Plan must contain 1 to {max_plan_steps} steps.")

    available = {tool.name: tool for tool in tools if tool.name in CALCULATOR_NAMES}
    validator = ToolArgumentsValidator()
    steps: list[PlanStep] = []
    for number, raw in enumerate(raw_steps, start=1):
        if not isinstance(raw, dict) or set(raw) != {"tool", "arguments"}:
            raise PlanValidationError(f"Step {number} requires tool and arguments only.")
        name = raw["tool"]
        if not isinstance(name, str) or name not in available:
            raise PlanValidationError(f"Step {number}: tool is not an available calculator tool.")
        args = raw["arguments"]
        if not isinstance(args, dict) or set(args) != {"a", "b"}:
            raise PlanValidationError(f"Step {number}: arguments must be exactly a and b.")
        parsed: dict[str, int | float | StepReference] = {}
        for key, value in args.items():
            if isinstance(value, dict) and set(value) == {"from_step"}:
                previous = value["from_step"]
                if type(previous) is not int or not 1 <= previous < number:
                    raise PlanValidationError(f"Step {number}: {key} must refer to an earlier step.")
                parsed[key] = StepReference(previous)
            elif _finite_number(value):
                parsed[key] = value
            else:
                raise PlanValidationError(f"Step {number}: {key} must be a finite number or prior-step reference.")
        try:
            validator.validate(
                available[name],
                {key: 0 if isinstance(value, StepReference) else value
                 for key, value in parsed.items()},
            )
        except ToolArgumentsValidationError as exc:
            raise PlanValidationError(f"Step {number}: {exc}") from exc
        steps.append(PlanStep(number, name, parsed))
    return Plan(tuple(steps))


class CalculatorPlanner:
    """模型提议整个计算计划；Runtime 独立校验并顺序执行，不自动重试。"""

    def __init__(self, llm: LLM, executor: ToolExecutor, max_plan_steps: int = 5) -> None:
        if type(max_plan_steps) is not int or max_plan_steps < 1:
            raise ValueError("max_plan_steps must be a positive integer")
        self._llm = llm
        self._executor = executor
        self._max_plan_steps = max_plan_steps
        self.last_state: PlanState | None = None

    def run(self, task: str) -> PlanRunResult:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a nonempty string")
        state = PlanState(task=task)
        self.last_state = state
        started = perf_counter()

        def finish(reason: PlanStopReason, answer: int | float | None = None,
                   error: str | None = None) -> PlanRunResult:
            return PlanRunResult(state, reason, answer, error, (perf_counter() - started) * 1000)

        available = [tool for tool in self._executor.available_tools()
                     if tool.name in CALCULATOR_NAMES]
        if not available:
            return finish(PlanStopReason.INVALID_PLAN, error="No calculator tools available.")
        tool_descriptions = [
            {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
            for tool in available
        ]
        response = self._llm.chat([
            Message(role="system", content=(
                "Produce ONLY a JSON object with a nonempty 'steps' array for the user's "
                "arithmetic task. Each step has 'tool' and 'arguments' with numeric a and b. "
                "To use a prior result, use {\"from_step\": N} (1-based, earlier steps only). "
                f"At most {self._max_plan_steps} steps. The final step must compute the answer. "
                "Do not write prose, code or tool calls. Available calculator tools: "
                + json.dumps(tool_descriptions, ensure_ascii=False)
            )),
            Message(role="user", content=task),
        ], tools=[])
        if response.tool_calls:
            return finish(PlanStopReason.INVALID_PLAN, error="Planner returned tool calls instead of JSON.")
        try:
            state.plan = parse_plan(response.content, available, self._max_plan_steps)
        except PlanValidationError as exc:
            return finish(PlanStopReason.INVALID_PLAN, error=str(exc))

        values: dict[int, int | float] = {}
        for step in state.plan.steps:
            arguments = {
                key: values[value.from_step] if isinstance(value, StepReference) else value
                for key, value in step.arguments.items()
            }
            call = ToolCall(step.tool, arguments)
            tool_started = perf_counter()
            result = self._executor.execute(call)  # 再次检查执行时 capability 与 schema
            duration_ms = (perf_counter() - tool_started) * 1000
            valid_number = _finite_number(result.value)
            verified = result.success and valid_number
            error = (
                result.error if not result.success else
                None if valid_number else "Tool returned a non-finite or non-numeric result."
            )
            state.executions.append(PlanExecution(step, call, result, verified, error, duration_ms))
            if not verified:
                return finish(PlanStopReason.TOOL_FAILED, error=f"Step {step.number}: {error}")
            values[step.number] = result.value
        return finish(PlanStopReason.COMPLETED, answer=values[len(state.plan.steps)])
