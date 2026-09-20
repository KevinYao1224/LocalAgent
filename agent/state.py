from dataclasses import dataclass, field
from typing import Any

from llm.base import Message, ModelResponse, ToolCall
from runtime.result import ToolResult


@dataclass(frozen=True, slots=True)
class InputMessage:
    """A conversation message supplied from outside the current agent run."""

    message: Message


@dataclass(frozen=True, slots=True)
class ToolExecution:
    """A tool call and the structured result produced for that exact call."""

    call: ToolCall
    result: ToolResult


@dataclass(slots=True)
class AgentStep:
    """One model response and the tool executions caused by it."""

    step: int
    model_response: ModelResponse
    tool_executions: list[ToolExecution] = field(default_factory=list)

    @property
    def tool_results(self) -> list[ToolResult]:
        """Return this step's results in execution order."""

        return [execution.result for execution in self.tool_executions]


TrajectoryEntry = InputMessage | AgentStep


@dataclass(slots=True)
class AgentState:
    """Runtime control state and a chronological trajectory of run facts.

    AgentState deliberately does not build provider conversation messages.
    That derived representation belongs to ConversationProjector.
    """

    max_steps: int
    step: int = 0
    trajectory: list[TrajectoryEntry] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_messages(
        cls,
        messages: list[Message],
        max_steps: int,
        metadata: dict[str, Any] | None = None,
    ) -> "AgentState":
        """Create run state and record each supplied conversation message."""

        state = cls(
            max_steps=max_steps,
            metadata=dict(metadata) if metadata is not None else {},
        )
        for message in messages:
            state.record_input(message)
        return state

    @property
    def current_task(self) -> str:
        """Return the latest externally supplied user message."""

        return next(
            (
                entry.message.content
                for entry in reversed(self.trajectory)
                if (
                    isinstance(entry, InputMessage)
                    and entry.message.role == "user"
                )
            ),
            "",
        )

    @property
    def steps(self) -> list[AgentStep]:
        """Return model-response steps in trajectory order."""

        return [
            entry
            for entry in self.trajectory
            if isinstance(entry, AgentStep)
        ]

    @property
    def tool_results(self) -> list[ToolResult]:
        """Return all tool results in execution order."""

        return [
            execution.result
            for agent_step in self.steps
            for execution in agent_step.tool_executions
        ]

    def record_input(self, message: Message) -> InputMessage:
        """Record a system, user, or pre-existing conversation message."""

        entry = InputMessage(message=message)
        self.trajectory.append(entry)
        return entry

    def begin_step(self) -> int:
        """Advance to the next model-call attempt and return its number."""

        if self.step >= self.max_steps:
            raise RuntimeError("Cannot begin a step beyond max_steps.")

        self.step += 1
        return self.step

    def record_model_response(self, response: ModelResponse) -> AgentStep:
        """Record the model response for the current step."""

        if self.step < 1:
            raise RuntimeError("begin_step() must be called first.")
        if self.steps and self.steps[-1].step == self.step:
            raise RuntimeError(
                f"Step {self.step} already has a model response."
            )

        agent_step = AgentStep(
            step=self.step,
            model_response=response,
        )
        self.trajectory.append(agent_step)
        return agent_step

    def record_tool_execution(
        self,
        call: ToolCall,
        result: ToolResult,
    ) -> ToolExecution:
        """Attach a tool call and its result to the current model step."""

        if not self.steps or self.steps[-1].step != self.step:
            raise RuntimeError(
                "A model response must be recorded before its tool results."
            )

        current_step = self.steps[-1]
        if call not in current_step.model_response.tool_calls:
            raise ValueError(
                "The tool call does not belong to the current model response."
            )
        if result.tool_name != call.name:
            raise ValueError(
                "The ToolResult name must match its originating ToolCall."
            )

        execution = ToolExecution(call=call, result=result)
        current_step.tool_executions.append(execution)
        return execution
