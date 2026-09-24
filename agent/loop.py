from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from time import perf_counter

from agent.context import ContextBuilder
from agent.conversation import ConversationProjector
from agent.state import AgentState, AgentStep, TrajectoryEntry
from llm.base import LLM, Message, ModelResponse
from observability.events import (
    AgentEvent,
    AgentFinished,
    AgentStarted,
    EmptyModelResponse,
    ModelCallFailed,
    ModelCallStarted,
    ModelResponseReceived,
    StepPreparationFailed,
    ToolExecutionFailed,
    ToolExecutionFinished,
    ToolExecutionStarted,
)
from observability.logger import AgentLogger, NullLogger
from observability.metrics import RunMetrics
from runtime.executor import ToolExecutor
from runtime.result import ToolResult


class StopReason(str, Enum):
    """Why an agent run stopped."""

    COMPLETED = "completed"
    MAX_STEPS = "max_steps"


@dataclass(slots=True)
class AgentRunResult:
    """The final state of one AgentLoop run."""

    response: ModelResponse
    state: AgentState
    stop_reason: StopReason
    metrics: RunMetrics | None = None

    @property
    def completed(self) -> bool:
        return self.stop_reason is StopReason.COMPLETED

    @property
    def run_id(self) -> str:
        """Correlation ID shared by this run's state and events."""

        return self.state.run_id

    @property
    def messages(self) -> list[Message]:
        """Formal conversation history kept for backward compatibility."""

        return ConversationProjector().project(self.state.trajectory)

    @property
    def steps(self) -> int:
        """Number of model-call attempts made during the run."""

        return self.state.step

    @property
    def tool_results(self) -> list[ToolResult]:
        """All tool results in execution order."""

        return self.state.tool_results

    @property
    def trajectory(self) -> list[TrajectoryEntry]:
        """External inputs and every model step in chronological order."""

        return self.state.trajectory

    @property
    def agent_steps(self) -> list[AgentStep]:
        """Every recorded model response without input-message entries."""

        return self.state.steps


class AgentLoop:
    """Repeatedly ask the model what to do and execute requested tools."""

    def __init__(
        self,
        llm: LLM,
        executor: ToolExecutor,
        max_steps: int = 5,
        logger: AgentLogger | None = None,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")

        self._llm = llm
        self._executor = executor
        self._max_steps = max_steps
        self._logger = logger if logger is not None else NullLogger()
        self._context_builder = (
            context_builder
            if context_builder is not None
            else ContextBuilder()
        )
        self._last_state: AgentState | None = None

    @property
    def last_state(self) -> AgentState | None:
        """Most recent run state, including state left by a failed run."""

        return self._last_state

    def run(self, messages: list[Message]) -> AgentRunResult:
        """Run until the model answers normally or the step limit is reached.

        The input list is copied. This lets callers reuse their original prompt,
        while the returned result contains the complete conversation trace.
        """

        state = AgentState.from_messages(
            messages=messages,
            max_steps=self._max_steps,
        )
        self._last_state = state
        last_response: ModelResponse | None = None
        metrics = RunMetrics()
        run_start = perf_counter()

        self._emit(state, AgentStarted(
            max_steps=state.max_steps,
            initial_messages=tuple(messages),
        ), timestamp=state.started_at)

        while state.step < state.max_steps:
            step = state.begin_step()
            try:
                available_tools = self._executor.available_tools()
                context = self._context_builder.build(state, self._llm)
            except Exception as exc:
                metrics = replace(
                    metrics,
                    preparation_errors=metrics.preparation_errors + 1,
                )
                self._emit(state, StepPreparationFailed(
                    step=step,
                    error_type=type(exc).__name__,
                    error=str(exc),
                ), step=step)
                self._emit(state, AgentFinished(
                    steps=step,
                    stop_reason="error",
                    final_content="",
                    metrics=metrics,
                ), duration_ms=self._elapsed_ms(run_start))
                raise
            self._emit(state, ModelCallStarted(
                step=step,
                message_count=len(context.messages),
                tool_names=tuple(tool.name for tool in available_tools),
            ), step=step)
            metrics = replace(metrics, model_calls=metrics.model_calls + 1)
            model_start = perf_counter()
            try:
                response = self._llm.chat(
                    messages=context.messages,
                    tools=available_tools,
                )
            except Exception as exc:
                model_duration_ms = self._elapsed_ms(model_start)
                metrics = replace(
                    metrics,
                    model_errors=metrics.model_errors + 1,
                    model_duration_ms=metrics.model_duration_ms + model_duration_ms,
                )
                self._emit(state, ModelCallFailed(
                    step=step,
                    error_type=type(exc).__name__,
                    error=str(exc),
                ), step=step, duration_ms=model_duration_ms)
                self._emit(state, AgentFinished(
                    steps=step,
                    stop_reason="error",
                    final_content="",
                    metrics=metrics,
                ), duration_ms=self._elapsed_ms(run_start))
                raise
            model_duration_ms = self._elapsed_ms(model_start)
            metrics = metrics.model_returned(response, model_duration_ms)
            last_response = response
            state.record_model_response(
                response=response,
                reasoning=self._llm.create_reasoning_block(response),
            )
            self._emit(state, ModelResponseReceived(
                step=step,
                response=response,
            ), step=step, duration_ms=model_duration_ms)

            if (
                not response.tool_calls
                and response.content.strip()
            ):
                result = AgentRunResult(
                    response=response,
                    state=state,
                    stop_reason=StopReason.COMPLETED,
                    metrics=metrics,
                )
                self._log_finished(result, run_start)
                return result

            if not response.tool_calls:
                self._emit(state, EmptyModelResponse(step=step), step=step)
                continue

            for call in response.tool_calls:
                self._emit(state, ToolExecutionStarted(
                    step=step,
                    call=call,
                ), step=step)
                metrics = replace(metrics, tool_calls=metrics.tool_calls + 1)
                tool_start = perf_counter()
                try:
                    tool_result = self._executor.execute(call)
                except Exception as exc:
                    tool_duration_ms = self._elapsed_ms(tool_start)
                    metrics = replace(
                        metrics,
                        tool_errors=metrics.tool_errors + 1,
                        tool_duration_ms=metrics.tool_duration_ms + tool_duration_ms,
                    )
                    self._emit(state, ToolExecutionFailed(
                        step=step,
                        call=call,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    ), step=step, duration_ms=tool_duration_ms)
                    self._emit(state, AgentFinished(
                        steps=step,
                        stop_reason="error",
                        final_content="",
                        metrics=metrics,
                    ), duration_ms=self._elapsed_ms(run_start))
                    raise
                tool_duration_ms = self._elapsed_ms(tool_start)
                metrics = metrics.tool_finished(tool_result, tool_duration_ms)
                self._emit(state, ToolExecutionFinished(
                    step=step,
                    call=call,
                    result=tool_result,
                ), step=step, duration_ms=tool_duration_ms)
                state.record_tool_execution(call, tool_result)

        # max_steps is always at least one, so the loop always sets this value.
        assert last_response is not None
        result = AgentRunResult(
            response=last_response,
            state=state,
            stop_reason=StopReason.MAX_STEPS,
            metrics=metrics,
        )
        self._log_finished(result, run_start)
        return result

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        """Durations use a monotonic clock, never wall-clock subtraction."""

        return (perf_counter() - start) * 1000

    def _emit(
        self,
        state: AgentState,
        event: AgentEvent,
        *,
        step: int | None = None,
        duration_ms: float | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        self._logger.log(replace(
            event,
            run_id=state.run_id,
            step_id=state.step_id(step) if step is not None else None,
            timestamp=timestamp or datetime.now(timezone.utc),
            duration_ms=duration_ms,
        ))

    def _log_finished(self, result: AgentRunResult, run_start: float) -> None:
        self._emit(result.state, AgentFinished(
            steps=result.steps,
            stop_reason=result.stop_reason.value,
            final_content=result.response.content,
            metrics=result.metrics,
        ), duration_ms=self._elapsed_ms(run_start))
