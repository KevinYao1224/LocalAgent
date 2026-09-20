from dataclasses import dataclass
from enum import Enum

from agent.conversation import ConversationProjector
from agent.state import AgentState, AgentStep, TrajectoryEntry
from llm.base import LLM, Message, ModelResponse
from observability.events import (
    AgentFinished,
    AgentStarted,
    EmptyModelResponse,
    ModelCallFailed,
    ModelCallStarted,
    ModelResponseReceived,
    ToolExecutionFailed,
    ToolExecutionFinished,
    ToolExecutionStarted,
)
from observability.logger import AgentLogger, NullLogger
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

    @property
    def completed(self) -> bool:
        return self.stop_reason is StopReason.COMPLETED

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
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")

        self._llm = llm
        self._executor = executor
        self._max_steps = max_steps
        self._logger = logger if logger is not None else NullLogger()
        self._conversation_projector = ConversationProjector()
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

        self._logger.log(AgentStarted(
            max_steps=state.max_steps,
            initial_messages=tuple(messages),
        ))

        while state.step < state.max_steps:
            step = state.begin_step()
            available_tools = self._executor.available_tools()
            conversation = self._conversation_projector.project(
                state.trajectory
            )
            self._logger.log(ModelCallStarted(
                step=step,
                message_count=len(conversation),
                tool_names=tuple(tool.name for tool in available_tools),
            ))
            try:
                response = self._llm.chat(
                    messages=conversation,
                    tools=available_tools,
                )
            except Exception as exc:
                self._logger.log(ModelCallFailed(
                    step=step,
                    error_type=type(exc).__name__,
                    error=str(exc),
                ))
                self._logger.log(AgentFinished(
                    steps=step,
                    stop_reason="error",
                    final_content="",
                ))
                raise
            last_response = response
            state.record_model_response(response)
            self._logger.log(ModelResponseReceived(
                step=step,
                response=response,
            ))

            if (
                not response.tool_calls
                and response.content.strip()
            ):
                result = AgentRunResult(
                    response=response,
                    state=state,
                    stop_reason=StopReason.COMPLETED,
                )
                self._log_finished(result)
                return result

            if not response.tool_calls:
                self._logger.log(EmptyModelResponse(step=step))
                continue

            for call in response.tool_calls:
                self._logger.log(ToolExecutionStarted(
                    step=step,
                    call=call,
                ))
                try:
                    tool_result = self._executor.execute(call)
                except Exception as exc:
                    self._logger.log(ToolExecutionFailed(
                        step=step,
                        call=call,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    ))
                    self._logger.log(AgentFinished(
                        steps=step,
                        stop_reason="error",
                        final_content="",
                    ))
                    raise
                self._logger.log(ToolExecutionFinished(
                    step=step,
                    call=call,
                    result=tool_result,
                ))
                state.record_tool_execution(call, tool_result)

        # max_steps is always at least one, so the loop always sets this value.
        assert last_response is not None
        result = AgentRunResult(
            response=last_response,
            state=state,
            stop_reason=StopReason.MAX_STEPS,
        )
        self._log_finished(result)
        return result

    def _log_finished(self, result: AgentRunResult) -> None:
        self._logger.log(AgentFinished(
            steps=result.steps,
            stop_reason=result.stop_reason.value,
            final_content=result.response.content,
        ))
