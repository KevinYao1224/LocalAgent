from dataclasses import dataclass
from enum import Enum

from llm.base import LLM, Message, ModelResponse
from tools.base import ToolExecutionError
from tools.registry import ToolNotFoundError, ToolRegistry


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

    @property
    def completed(self) -> bool:
        return self.stop_reason is StopReason.COMPLETED


class AgentLoop:
    """Repeatedly ask the model what to do and execute requested tools."""

    def __init__(
        self,
        llm: LLM,
        registry: ToolRegistry,
        max_steps: int = 5,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")

        self._llm = llm
        self._registry = registry
        self._max_steps = max_steps

    def run(self, messages: list[Message]) -> AgentRunResult:
        """Run until the model answers normally or the step limit is reached.

        The input list is copied. This lets callers reuse their original prompt,
        while the returned result contains the complete conversation trace.
        """

        history = list(messages)
        last_response: ModelResponse | None = None

        for step in range(1, self._max_steps + 1):
            response = self._llm.chat(
                messages=history,
                tools=self._registry.all(),
            )
            last_response = response

            history.append(Message(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls,
            ))

            if not response.tool_calls:
                return AgentRunResult(
                    response=response,
                    messages=history,
                    steps=step,
                    stop_reason=StopReason.COMPLETED,
                )

            for call in response.tool_calls:
                tool_output = self._execute_tool(
                    name=call.name,
                    arguments=call.arguments,
                )
                history.append(Message(
                    role="tool",
                    tool_name=call.name,
                    content=tool_output,
                ))

        # max_steps is always at least one, so the loop always sets this value.
        assert last_response is not None
        return AgentRunResult(
            response=last_response,
            messages=history,
            steps=self._max_steps,
            stop_reason=StopReason.MAX_STEPS,
        )

    def _execute_tool(
        self,
        name: str,
        arguments: dict,
    ) -> str:
        """Turn expected tool failures into observations for the model."""

        try:
            result = self._registry.execute(name, arguments)
        except (ToolNotFoundError, ToolExecutionError) as exc:
            return f"Error: {exc}"

        return str(result)
