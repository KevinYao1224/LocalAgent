"""Deterministic experiments for Phase 6C ContextBuilder policies.

Run from the project root:

    python experiments/context_builder_experiment.py

No Ollama server is needed.
"""

from collections.abc import Iterable
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import (
    AgentLoop,
    AgentState,
    ContextBuilder,
    ReasoningReplayPolicy,
)
from llm.base import (
    LLM,
    LLMCapabilities,
    Message,
    ModelResponse,
    ReasoningBlock,
    ReasoningReplaySupport,
    ToolCall,
)
from runtime import ToolExecutor, ToolResult
from tools import ToolRegistry, add_tool


class DeclaredLLM(LLM):
    """An LLM identity used to test capability and provenance matching."""

    def __init__(
        self,
        provider: str = "test-provider",
        model: str | None = "test-model",
        replay_supported: bool = True,
    ) -> None:
        self._provider = provider
        self._model = model
        self._replay_supported = replay_supported

    @property
    def provider_name(self) -> str:
        return self._provider

    @property
    def model_name(self) -> str | None:
        return self._model

    @property
    def capabilities(self) -> LLMCapabilities:
        support = (
            ReasoningReplaySupport.SUPPORTED
            if self._replay_supported
            else ReasoningReplaySupport.UNSUPPORTED
        )
        return LLMCapabilities(reasoning_replay=support)

    def chat(self, messages, tools=None) -> ModelResponse:
        raise AssertionError("This identity-only LLM must not be called.")


class RecordingReplayLLM(DeclaredLLM):
    """A replay-capable scripted LLM that records each received context."""

    def __init__(self, responses: Iterable[ModelResponse]) -> None:
        super().__init__()
        self._responses = iter(responses)
        self.calls: list[list[Message]] = []

    def chat(self, messages, tools=None) -> ModelResponse:
        self.calls.append(list(messages))
        return next(self._responses)


def create_state_with_tool_reasoning(
    *,
    provider: str = "test-provider",
    model: str | None = "test-model",
    replayable: bool = True,
    execute_tool: bool = True,
) -> tuple[AgentState, ToolCall]:
    state = AgentState.from_messages(
        messages=[Message(role="user", content="Calculate 1 + 2.")],
        max_steps=4,
    )
    call = ToolCall(name="add", arguments={"a": 1, "b": 2})
    state.begin_step()
    state.record_model_response(
        response=ModelResponse(
            thinking="I should call add and inspect its result.",
            tool_calls=[call],
        ),
        reasoning=ReasoningBlock(
            provider=provider,
            model=model,
            raw_thinking="I should call add and inspect its result.",
            replayable=replayable,
        ),
    )
    if execute_tool:
        state.record_tool_execution(
            call=call,
            result=ToolResult.succeeded(tool_name="add", value=3),
        )
    return state, call


def latest_pending_builder() -> ContextBuilder:
    return ContextBuilder(
        reasoning_replay=ReasoningReplayPolicy.LATEST_PENDING,
    )


def experiment_invalid_policy_is_rejected() -> None:
    try:
        ContextBuilder(reasoning_replay="invalid")
    except ValueError:
        print("invalid policy: unsupported policy values are rejected.")
        return

    raise AssertionError("ContextBuilder accepted an unsupported policy.")


def experiment_default_off() -> None:
    state, _ = create_state_with_tool_reasoning()
    result = ContextBuilder().build(state, DeclaredLLM())

    assert not result.reasoning_replayed
    assert result.replayed_reasoning_step is None
    assert [message.role for message in result.messages] == [
        "user",
        "assistant",
        "tool",
    ]
    assert result.messages[1].thinking == ""
    print("off: canonical conversation is unchanged.")


def experiment_latest_pending_tool_step() -> None:
    state, call = create_state_with_tool_reasoning()
    result = latest_pending_builder().build(state, DeclaredLLM())

    assert result.reasoning_replayed
    assert result.replayed_reasoning_step == 1
    assert result.messages[1].role == "assistant"
    assert result.messages[1].thinking.startswith("I should call add")
    assert result.messages[1].tool_calls == [call]
    assert result.messages[2].role == "tool"
    assert result.messages[2].content == "3"
    print("latest_pending: reasoning is merged into its tool-call message.")


def experiment_safe_fallbacks() -> None:
    cases = [
        (
            create_state_with_tool_reasoning()[0],
            DeclaredLLM(replay_supported=False),
            "unsupported adapter",
        ),
        (
            create_state_with_tool_reasoning(provider="other-provider")[0],
            DeclaredLLM(),
            "provider mismatch",
        ),
        (
            create_state_with_tool_reasoning(model="other-model")[0],
            DeclaredLLM(),
            "model mismatch",
        ),
        (
            create_state_with_tool_reasoning(model=None)[0],
            DeclaredLLM(model=None),
            "unknown model identity",
        ),
        (
            create_state_with_tool_reasoning(replayable=False)[0],
            DeclaredLLM(),
            "non-replayable block",
        ),
        (
            create_state_with_tool_reasoning(execute_tool=False)[0],
            DeclaredLLM(),
            "missing tool observation",
        ),
    ]

    for state, llm, label in cases:
        result = latest_pending_builder().build(state, llm)
        assert not result.reasoning_replayed, label
        assert all(not message.thinking for message in result.messages), label

    print("fallbacks: capability, provenance, and readiness checks passed.")


def experiment_inert_empty_step_keeps_previous_reasoning_pending() -> None:
    state, _ = create_state_with_tool_reasoning()
    state.begin_step()
    state.record_model_response(ModelResponse())

    result = latest_pending_builder().build(state, DeclaredLLM())

    assert result.replayed_reasoning_step == 1
    assert sum(bool(message.thinking) for message in result.messages) == 1
    print("empty retry: the previous pending reasoning remains selectable.")


def experiment_new_decision_blocks_stale_reasoning() -> None:
    state, _ = create_state_with_tool_reasoning()
    second_call = ToolCall(name="add", arguments={"a": 3, "b": 4})
    state.begin_step()
    state.record_model_response(ModelResponse(tool_calls=[second_call]))
    state.record_tool_execution(
        call=second_call,
        result=ToolResult.succeeded(tool_name="add", value=7),
    )

    result = latest_pending_builder().build(state, DeclaredLLM())

    assert not result.reasoning_replayed
    assert all(not message.thinking for message in result.messages)
    print("stale reasoning: a newer decision step forms a replay barrier.")


def experiment_thinking_only_step() -> None:
    state = AgentState.from_messages(
        messages=[Message(role="user", content="Think, then answer.")],
        max_steps=2,
    )
    state.begin_step()
    state.record_model_response(
        response=ModelResponse(thinking="I am not ready to answer yet."),
        reasoning=ReasoningBlock(
            provider="test-provider",
            model="test-model",
            raw_thinking="I am not ready to answer yet.",
            replayable=True,
        ),
    )

    result = latest_pending_builder().build(state, DeclaredLLM())

    assert result.replayed_reasoning_step == 1
    assert [message.role for message in result.messages] == [
        "user",
        "assistant",
    ]
    assert result.messages[-1].content == ""
    assert result.messages[-1].thinking
    print("thinking-only: replay creates a reasoning-only assistant message.")


def experiment_agent_loop_integration() -> None:
    call = ToolCall(name="add", arguments={"a": 1, "b": 2})
    llm = RecordingReplayLLM([
        ModelResponse(
            thinking="I will add the two numbers.",
            tool_calls=[call],
        ),
        ModelResponse(content="1 + 2 is 3."),
    ])
    registry = ToolRegistry()
    registry.register(add_tool)
    result = AgentLoop(
        llm=llm,
        executor=ToolExecutor(registry),
        context_builder=latest_pending_builder(),
    ).run([
        Message(role="user", content="Calculate 1 + 2."),
    ])

    assert result.completed
    assert len(llm.calls) == 2
    assert llm.calls[1][1].role == "assistant"
    assert llm.calls[1][1].thinking == "I will add the two numbers."
    assert llm.calls[1][2].role == "tool"
    print("AgentLoop: injected ContextBuilder controls the second model call.")


if __name__ == "__main__":
    experiment_invalid_policy_is_rejected()
    experiment_default_off()
    experiment_latest_pending_tool_step()
    experiment_safe_fallbacks()
    experiment_inert_empty_step_keeps_previous_reasoning_pending()
    experiment_new_decision_blocks_stale_reasoning()
    experiment_thinking_only_step()
    experiment_agent_loop_integration()
    print("\nAll ContextBuilder experiments passed.")
