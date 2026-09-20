"""Inspect Phase 6B reasoning types and safe replay capability handling.

Run from the project root:

    python experiments/reasoning_continuity_experiment.py

No Ollama server is needed. The adapter experiment only inspects local request
conversion and capability metadata; it does not make an HTTP request.
"""

from collections.abc import Iterable
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import AgentLoop
from llm.base import (
    LLM,
    LLMCapabilities,
    Message,
    ModelResponse,
    ReasoningReplaySupport,
    ToolCall,
)
from llm.ollama import OllamaClient
from runtime import ToolExecutor
from tools import ToolRegistry


class RecordingScriptedLLM(LLM):
    """Return prepared responses and retain each projected conversation."""

    def __init__(self, responses: Iterable[ModelResponse]) -> None:
        self._responses = iter(responses)
        self.calls: list[list[Message]] = []

    def chat(self, messages, tools=None) -> ModelResponse:
        self.calls.append(list(messages))
        return next(self._responses)


class ReplayCapableScriptedLLM(RecordingScriptedLLM):
    @property
    def provider_name(self) -> str:
        return "test-provider"

    @property
    def model_name(self) -> str:
        return "test-thinking-model"

    @property
    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(
            reasoning_replay=ReasoningReplaySupport.SUPPORTED,
        )


def create_empty_executor() -> ToolExecutor:
    return ToolExecutor(ToolRegistry())


def experiment_unsupported_provider_falls_back_safely() -> None:
    llm = RecordingScriptedLLM([
        ModelResponse(thinking="Internal reasoning from an unknown provider."),
        ModelResponse(content="Visible final answer."),
    ])
    result = AgentLoop(
        llm=llm,
        executor=create_empty_executor(),
        max_steps=2,
    ).run([
        Message(role="user", content="Answer the request."),
    ])

    reasoning = result.agent_steps[0].reasoning
    assert reasoning is not None
    assert reasoning.provider == "RecordingScriptedLLM"
    assert reasoning.model is None
    assert not reasoning.replayable
    assert reasoning.raw_thinking.startswith("Internal reasoning")
    assert result.agent_steps[1].reasoning is None

    # Default projection remains replay-off: the second model call receives
    # the user message but no assistant thinking message.
    assert len(llm.calls[1]) == 1
    assert llm.calls[1][0].role == "user"
    assert llm.calls[1][0].thinking == ""
    print("Unsupported provider: reasoning captured, replay disabled.")


def experiment_supported_capability_does_not_enable_replay() -> None:
    llm = ReplayCapableScriptedLLM([
        ModelResponse(thinking="Replayable provider reasoning."),
        ModelResponse(content="Visible final answer."),
    ])
    result = AgentLoop(
        llm=llm,
        executor=create_empty_executor(),
        max_steps=2,
    ).run([
        Message(role="user", content="Answer the request."),
    ])

    reasoning = result.agent_steps[0].reasoning
    assert reasoning is not None
    assert reasoning.provider == "test-provider"
    assert reasoning.model == "test-thinking-model"
    assert reasoning.replayable

    # Capability means replay is possible, not that the default projector
    # should activate it. ContextBuilder policy will make that choice later.
    assert len(llm.calls[1]) == 1
    assert llm.calls[1][0].thinking == ""
    print("Supported provider: block is replayable, default remains off.")


def experiment_ollama_transport_capability() -> None:
    with OllamaClient(model="qwen3.5:9b") as client:
        assert (
            client.capabilities.reasoning_replay
            is ReasoningReplaySupport.SUPPORTED
        )
        message = Message(
            role="assistant",
            content="",
            thinking="Reasoning retained for a follow-up tool turn.",
            tool_calls=[
                ToolCall(name="add", arguments={"a": 1, "b": 2}),
            ],
        )
        wire_message = client._convert_message(message)
        reasoning = client.create_reasoning_block(ModelResponse(
            thinking=message.thinking,
        ))

    assert wire_message["thinking"] == message.thinking
    assert wire_message["tool_calls"][0]["function"]["name"] == "add"
    assert reasoning is not None
    assert reasoning.provider == "ollama"
    assert reasoning.model == "qwen3.5:9b"
    assert reasoning.replayable
    print("Ollama: capability and thinking wire conversion verified.")


if __name__ == "__main__":
    experiment_unsupported_provider_falls_back_safely()
    experiment_supported_capability_does_not_enable_replay()
    experiment_ollama_transport_capability()
    print("\nAll reasoning continuity experiments passed.")
