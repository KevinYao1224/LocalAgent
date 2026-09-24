from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from llm.base import Message, ModelResponse, ReasoningBlock, ToolCall
from runtime.result import ToolResult


@dataclass(frozen=True, slots=True)
class InputMessage:
    """由当前 Agent 运行之外提供的对话消息。"""

    message: Message


@dataclass(frozen=True, slots=True)
class ToolExecution:
    """一次工具调用，以及专属于该调用的结构化结果。"""

    call: ToolCall
    result: ToolResult


@dataclass(slots=True)
class AgentStep:
    """一次模型响应，以及由该响应触发的工具执行。"""

    step: int
    model_response: ModelResponse
    reasoning: ReasoningBlock | None = None
    tool_executions: list[ToolExecution] = field(default_factory=list)
    step_id: str | None = None

    @property
    def tool_results(self) -> list[ToolResult]:
        """按执行顺序返回本步骤的工具结果。"""

        return [execution.result for execution in self.tool_executions]


TrajectoryEntry = InputMessage | AgentStep


@dataclass(slots=True)
class AgentState:
    """保存运行时控制状态和按时间排列的运行事实轨迹。

    AgentState 有意不构造 provider 对话消息；这种派生表示由
    ConversationProjector 负责。
    """

    max_steps: int
    run_id: str = field(default_factory=lambda: uuid4().hex)
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
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
        """创建本轮运行状态，并记录传入的每一条对话消息。"""

        state = cls(
            max_steps=max_steps,
            metadata=dict(metadata) if metadata is not None else {},
        )
        for message in messages:
            state.record_input(message)
        return state

    @property
    def current_task(self) -> str:
        """返回最近一条由外部提供的 user 消息。"""

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
        """按轨迹顺序返回模型响应步骤。"""

        return [
            entry
            for entry in self.trajectory
            if isinstance(entry, AgentStep)
        ]

    @property
    def tool_results(self) -> list[ToolResult]:
        """按执行顺序返回所有工具结果。"""

        return [
            execution.result
            for agent_step in self.steps
            for execution in agent_step.tool_executions
        ]

    def record_input(self, message: Message) -> InputMessage:
        """记录 system、user 或先前对话中的消息。"""

        entry = InputMessage(message=message)
        self.trajectory.append(entry)
        return entry

    def begin_step(self) -> int:
        """开始下一次模型调用尝试，并返回步骤编号。"""

        if self.step >= self.max_steps:
            raise RuntimeError("Cannot begin a step beyond max_steps.")

        self.step += 1
        return self.step

    def step_id(self, step: int) -> str:
        """返回模型调用尝试的稳定标识，失败的尝试也有标识。"""

        if step < 1 or step > self.step:
            raise ValueError("Step must refer to a started model attempt.")
        return f"{self.run_id}:{step}"

    def record_model_response(
        self,
        response: ModelResponse,
        reasoning: ReasoningBlock | None = None,
    ) -> AgentStep:
        """记录当前步骤的模型响应。"""

        if self.step < 1:
            raise RuntimeError("begin_step() must be called first.")
        if self.steps and self.steps[-1].step == self.step:
            raise RuntimeError(
                f"Step {self.step} already has a model response."
            )

        agent_step = AgentStep(
            step=self.step,
            model_response=response,
            reasoning=reasoning,
            step_id=self.step_id(self.step),
        )
        self.trajectory.append(agent_step)
        return agent_step

    def record_tool_execution(
        self,
        call: ToolCall,
        result: ToolResult,
    ) -> ToolExecution:
        """将工具调用及其结果关联到当前模型步骤。"""

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
