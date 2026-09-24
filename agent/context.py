from dataclasses import dataclass
from enum import Enum

from agent.conversation import ConversationProjector
from agent.state import AgentState, AgentStep
from llm.base import LLM, Message, ReasoningReplaySupport


class ReasoningReplayPolicy(str, Enum):
    """规定 ContextBuilder 是否以及如何把已捕获的 reasoning 放入模型输入。"""

    OFF = "off"
    LATEST_PENDING = "latest_pending"


@dataclass(slots=True)
class ContextBuildResult:
    """保存本次模型输入，以及实际回放 reasoning 的 step（如果有）。"""

    messages: list[Message]
    replayed_reasoning_step: int | None = None

    @property
    def reasoning_replayed(self) -> bool:
        return self.replayed_reasoning_step is not None


class ContextBuilder:
    """根据 AgentState 轨迹和策略构造模型输入。"""

    def __init__(
        self,
        reasoning_replay: ReasoningReplayPolicy | str = (
            ReasoningReplayPolicy.OFF
        ),
        projector: ConversationProjector | None = None,
    ) -> None:
        self.reasoning_replay = ReasoningReplayPolicy(reasoning_replay)
        self._projector = (
            projector if projector is not None else ConversationProjector()
        )

    def build(self, state: AgentState, llm: LLM) -> ContextBuildResult:
        selected_step = self._select_reasoning_step(state, llm)
        messages: list[Message] = []

        for entry in state.trajectory:
            entry_messages = self._projector.project_entry(entry)
            if entry is selected_step:
                entry_messages = self._with_replayed_reasoning(
                    selected_step,
                    entry_messages,
                )
            messages.extend(entry_messages)

        return ContextBuildResult(
            messages=messages,
            replayed_reasoning_step=(
                selected_step.step if selected_step is not None else None
            ),
        )

    def _select_reasoning_step(
        self,
        state: AgentState,
        llm: LLM,
    ) -> AgentStep | None:
        if self.reasoning_replay is ReasoningReplayPolicy.OFF:
            return None
        if (
            llm.capabilities.reasoning_replay
            is not ReasoningReplaySupport.SUPPORTED
        ):
            return None

        for agent_step in reversed(state.steps):
            if self._is_inert_empty_step(agent_step):
                continue
            if not self._awaits_continuation(agent_step):
                return None
            if not self._matches_current_model(agent_step, llm):
                return None
            return agent_step

        return None

    @staticmethod
    def _is_inert_empty_step(agent_step: AgentStep) -> bool:
        response = agent_step.model_response
        return (
            not response.content.strip()
            and not response.tool_calls
            and agent_step.reasoning is None
        )

    @staticmethod
    def _awaits_continuation(agent_step: AgentStep) -> bool:
        response = agent_step.model_response

        if response.tool_calls:
            return len(agent_step.tool_executions) == len(
                response.tool_calls
            )

        return not response.content.strip()

    @staticmethod
    def _matches_current_model(agent_step: AgentStep, llm: LLM) -> bool:
        reasoning = agent_step.reasoning
        if reasoning is None or not reasoning.replayable:
            return False
        if reasoning.provider != llm.provider_name:
            return False
        if reasoning.model is None or llm.model_name is None:
            return False
        return reasoning.model == llm.model_name

    @staticmethod
    def _with_replayed_reasoning(
        agent_step: AgentStep,
        messages: list[Message],
    ) -> list[Message]:
        reasoning = agent_step.reasoning
        assert reasoning is not None

        replay_message = Message(
            role="assistant",
            content=agent_step.model_response.content,
            tool_calls=agent_step.model_response.tool_calls,
            thinking=reasoning.raw_thinking,
        )

        if messages and messages[0].role == "assistant":
            return [replay_message, *messages[1:]]

        return [replay_message, *messages]
