"""检查 Phase 6B reasoning 类型及安全回放能力处理。

从项目根目录运行：

    python experiments/reasoning_continuity_experiment.py

无需 Ollama 服务。适配器实验只检查本地请求转换和 capability 元数据，不会发起 HTTP 请求。
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
    """返回预先准备的响应，并保留每次投影生成的对话。"""

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

    # 默认投影仍关闭 replay：第二次模型调用收到 user 消息，但不包含 assistant thinking 消息。
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

    # 声明 capability 只表示可以回放，不代表默认投影应启用回放；稍后由 ContextBuilder
    # 策略作出选择。
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
