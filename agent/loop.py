from dataclasses import dataclass
from enum import Enum

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
    messages: list[Message]
    steps: int
    stop_reason: StopReason
    tool_results: list[ToolResult]

    @property
    def completed(self) -> bool:
        return self.stop_reason is StopReason.COMPLETED


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

    def run(self, messages: list[Message]) -> AgentRunResult:
        """Run until the model answers normally or the step limit is reached.

        The input list is copied. This lets callers reuse their original prompt,
        while the returned result contains the complete conversation trace.
        """

        history = list(messages)
        tool_results: list[ToolResult] = []
        last_response: ModelResponse | None = None

        self._logger.log(AgentStarted(
            max_steps=self._max_steps,
            initial_messages=tuple(history),
        ))

        for step in range(1, self._max_steps + 1):
            available_tools = self._executor.available_tools()
            self._logger.log(ModelCallStarted(
                step=step,
                message_count=len(history),
                tool_names=tuple(tool.name for tool in available_tools),
            ))
            try:
                response = self._llm.chat(
                    messages=history,
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
            self._logger.log(ModelResponseReceived(
                step=step,
                response=response,
            ))

            if response.tool_calls or response.content.strip():
                history.append(Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                ))

            if (
                not response.tool_calls
                and response.content.strip()
            ):
                result = AgentRunResult(
                    response=response,
                    messages=history,
                    steps=step,
                    stop_reason=StopReason.COMPLETED,
                    tool_results=tool_results,
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
                tool_results.append(tool_result)
                history.append(Message(
                    role="tool",
                    tool_name=call.name,
                    content=tool_result.to_message_content(),
                ))

        # max_steps is always at least one, so the loop always sets this value.
        assert last_response is not None
        result = AgentRunResult(
            response=last_response,
            messages=history,
            steps=self._max_steps,
            stop_reason=StopReason.MAX_STEPS,
            tool_results=tool_results,
        )
        self._log_finished(result)
        return result

    def _log_finished(self, result: AgentRunResult) -> None:
        self._logger.log(AgentFinished(
            steps=result.steps,
            stop_reason=result.stop_reason.value,
            final_content=result.response.content,
        ))
