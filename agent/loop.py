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
    """说明一次 Agent 运行停止的原因。"""

    COMPLETED = "completed"
    MAX_STEPS = "max_steps"


@dataclass(slots=True)
class AgentRunResult:
    """保存一次 AgentLoop 运行结束时的状态和结果。"""

    response: ModelResponse
    state: AgentState
    stop_reason: StopReason
    metrics: RunMetrics | None = None

    @property
    def completed(self) -> bool:
        return self.stop_reason is StopReason.COMPLETED

    @property
    def run_id(self) -> str:
        """本次运行的关联 ID；状态和事件共用此 ID。"""

        return self.state.run_id

    @property
    def messages(self) -> list[Message]:
        """返回正式对话历史；保留此属性以兼容旧调用方。"""

        return ConversationProjector().project(self.state.trajectory)

    @property
    def steps(self) -> int:
        """返回本次运行尝试调用模型的次数。"""

        return self.state.step

    @property
    def tool_results(self) -> list[ToolResult]:
        """按执行顺序返回本次运行产生的所有工具结果。"""

        return self.state.tool_results

    @property
    def trajectory(self) -> list[TrajectoryEntry]:
        """按时间顺序返回外部输入和所有模型步骤。"""

        return self.state.trajectory

    @property
    def agent_steps(self) -> list[AgentStep]:
        """返回记录过的所有模型响应步骤，不包含输入消息条目。"""

        return self.state.steps


class AgentLoop:
    """循环询问模型下一步行动，并执行模型请求的工具。"""

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
        """返回最近一次运行的状态；运行失败时也保留当时的状态。"""

        return self._last_state

    def run(self, messages: list[Message]) -> AgentRunResult:
        """运行至模型给出正常回答或达到步骤上限。

        输入列表会被复制，因此调用方可以继续复用原始提示；返回结果包含完整运行轨迹。
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

        # max_steps 至少为 1，因此循环必定会为该变量赋值。
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
        """使用单调时钟计算耗时，不通过墙上时钟相减。"""

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
