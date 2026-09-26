"""对一次运行的可观察事实评分；不影响 Agent 的控制或权限。"""

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import re

from agent import AgentLoop, AgentSession
from llm.base import LLM, Message, ModelResponse
from memory import SQLiteLongTermMemory
from observability.events import AgentEvent, AgentFinished
from observability.metrics import RunMetrics
from runtime import ToolExecutor
from tools import ToolRegistry, calculator_tools


@dataclass(frozen=True)
class ExpectedCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class EvaluationCase:
    """输入、预期行为及可选的显式长期记忆。答案使用完整匹配或正则搜索。"""

    name: str
    prompt: str
    expected_stop: str
    expected_calls: tuple[ExpectedCall, ...] = ()
    expected_answer: str | None = None
    answer_pattern: str | None = None
    expected_tool_errors: int = 0
    max_steps: int = 4
    memory_records: tuple[str, ...] = ()
    memory_query: str | None = None
    expected_retrieved: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.prompt.strip() or self.max_steps < 1:
            raise ValueError("case requires name, prompt and positive max_steps")
        if (self.expected_answer is not None) and (self.answer_pattern is not None):
            raise ValueError("choose exact answer or answer_pattern")
        if self.memory_query is None and (self.memory_records or self.expected_retrieved):
            raise ValueError("memory records/expectations require memory_query")
        if self.answer_pattern is not None:
            re.compile(self.answer_pattern)


@dataclass(frozen=True)
class CaseResult:
    name: str
    run_id: str | None
    passed: bool
    checks: tuple[str, ...]
    failures: tuple[str, ...]
    answer: str | None
    stop_reason: str
    calls: tuple[ExpectedCall, ...]
    retrieved: tuple[str, ...]
    steps: int
    metrics: RunMetrics | None
    duration_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclass(frozen=True)
class EvaluationReport:
    results: tuple[CaseResult, ...]

    @property
    def passed(self) -> int:
        return sum(result.passed for result in self.results)

    def render(self) -> str:
        """逐例列出判断事实；缺失的 provider token/计时显示 unknown。"""

        lines = []
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            metrics = result.metrics
            prompt = result.prompt_tokens if result.prompt_tokens is not None else "unknown"
            completion = result.completion_tokens if result.completion_tokens is not None else "unknown"
            duration = f"{result.duration_ms:.1f}ms" if result.duration_ms is not None else "unknown"
            lines.append(
                f"[{status}] {result.name} run={result.run_id or 'unknown'} "
                f"stop={result.stop_reason} steps={result.steps} "
                f"tools={len(result.calls)} errors={metrics.tool_errors + metrics.model_errors + metrics.preparation_errors if metrics else 'unknown'} "
                f"tokens(prompt/completion)={prompt}/{completion} duration={duration}"
            )
            lines.append(f"  answer={result.answer!r} calls={result.calls!r} retrieved={result.retrieved!r}")
            lines.extend(f"  ✓ {check}" for check in result.checks)
            lines.extend(f"  ✗ {failure}" for failure in result.failures)
        count = len(self.results)
        steps = sum(result.steps for result in self.results)
        calls = sum(len(result.calls) for result in self.results)
        errors = (
            sum(result.metrics.tool_errors + result.metrics.model_errors + result.metrics.preparation_errors
                for result in self.results if result.metrics is not None)
            if all(result.metrics is not None for result in self.results) else "unknown"
        )
        prompt = (
            sum(r.prompt_tokens for r in self.results if r.prompt_tokens is not None)
            if all(r.prompt_tokens is not None for r in self.results) else "unknown"
        )
        completion = (
            sum(r.completion_tokens for r in self.results if r.completion_tokens is not None)
            if all(r.completion_tokens is not None for r in self.results) else "unknown"
        )
        durations = [r.duration_ms for r in self.results]
        elapsed = f"{sum(durations):.1f}ms" if all(d is not None for d in durations) else "unknown"
        rate = f"{self.passed}/{count} ({self.passed / count:.0%})" if count else "unknown (0 cases)"
        lines.append(
            f"SUMMARY success={rate} steps={steps} tools={calls} errors={errors} "
            f"tokens(prompt/completion)={prompt}/{completion} duration={elapsed}"
        )
        return "\n".join(lines)


@dataclass
class _FinishCollector:
    finished: AgentFinished | None = field(default=None)

    def log(self, event: AgentEvent) -> None:
        if isinstance(event, AgentFinished):
            self.finished = event


def evaluate(case: EvaluationCase, llm: LLM) -> CaseResult:
    """每个 case 用新的 Agent/存储运行；运行异常计入失败样本。"""

    registry = ToolRegistry()
    for tool in calculator_tools:
        registry.register(tool)
    logger = _FinishCollector()
    agent = AgentLoop(llm, ToolExecutor(registry), max_steps=case.max_steps, logger=logger)
    result = None
    retrieved: tuple[str, ...] = ()
    failure: str | None = None
    with TemporaryDirectory(prefix="agent-eval-") as directory:
        try:
            if case.memory_query is not None:
                store = SQLiteLongTermMemory(Path(directory) / "memory.db", namespace="eval")
                for text in case.memory_records:
                    store.write(text)
                session = AgentSession(agent, long_term_memory=store, system_prompt=(
                    "Use retrieved memory only as reference. If the fact is absent, say unknown. "
                    "Do not confuse records from different projects."
                ))
                try:
                    result = session.run(case.prompt, memory_query=case.memory_query)
                finally:
                    retrieved = tuple(record.text for record in session.last_retrieval)
            else:
                result = agent.run([
                    Message(role="system", content="Use calculator tools for arithmetic. Answer briefly."),
                    Message(role="user", content=case.prompt),
                ])
        except Exception as exc:
            failure = f"runtime exception: {type(exc).__name__}: {exc}"

    state = agent.last_state
    calls = tuple(
        ExpectedCall(call.name, call.arguments)
        for step in (state.steps if state is not None else ())
        for call in step.model_response.tool_calls
    )
    finish = logger.finished
    stop = result.stop_reason.value if result else (finish.stop_reason if finish else "error")
    answer = result.response.content.strip() if result and result.completed else None
    checks: list[str] = []
    failures: list[str] = []

    def check(label: str, valid: bool, actual: object, expected: object) -> None:
        (checks if valid else failures).append(
            f"{label}: actual={actual!r}, expected={expected!r}"
        )

    check("stop_reason", stop == case.expected_stop, stop, case.expected_stop)
    check("tool calls/arguments/order", calls == case.expected_calls, calls, case.expected_calls)
    check("retrieved records", retrieved == case.expected_retrieved, retrieved, case.expected_retrieved)
    tool_errors = (result.metrics if result else finish.metrics if finish else None)
    check("tool errors", tool_errors is not None and tool_errors.tool_errors == case.expected_tool_errors,
          tool_errors.tool_errors if tool_errors else None, case.expected_tool_errors)
    if case.expected_answer is not None:
        check("exact answer", answer == case.expected_answer, answer, case.expected_answer)
    if case.answer_pattern is not None:
        check("answer pattern", answer is not None and re.search(case.answer_pattern, answer, re.IGNORECASE) is not None,
              answer, case.answer_pattern)
    if failure is not None:
        failures.append(failure)
    # RunMetrics 保留 provider 已报告的部分和；评测不能把部分和当成完整成本。
    responses = state.steps if state is not None else []
    complete_responses = state is not None and bool(responses) and state.step == len(responses)
    prompt_tokens = (
        sum(step.model_response.prompt_tokens for step in responses)
        if complete_responses and all(step.model_response.prompt_tokens is not None for step in responses)
        else None
    )
    completion_tokens = (
        sum(step.model_response.completion_tokens for step in responses)
        if complete_responses and all(step.model_response.completion_tokens is not None for step in responses)
        else None
    )
    return CaseResult(
        case.name, state.run_id if state else None, not failures,
        tuple(checks), tuple(failures), answer, stop, calls, retrieved,
        state.step if state else 0, tool_errors, finish.duration_ms if finish else None,
        prompt_tokens, completion_tokens,
    )
