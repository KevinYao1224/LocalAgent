from typing import Any

import httpx

from llm.base import (
    LLM,
    LLMCapabilities,
    Message,
    ModelResponse,
    ReasoningReplaySupport,
    ToolCall,
)
from tools.base import Tool


class OllamaClient(LLM):
    _CAPABILITIES = LLMCapabilities(
        reasoning_replay=ReasoningReplaySupport.SUPPORTED,
    )

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
    ):
        self.model = model

        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout
        )

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def capabilities(self) -> LLMCapabilities:
        return self._CAPABILITIES
    
    def _convert_tool(self, tool: Tool) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
    
    def _convert_message(self, message: Message) -> dict[str, Any]:
        data: dict[str, Any] = {
            "role": message.role,
            "content": message.content,
        }

        if message.tool_name is not None:
            data["tool_name"] = message.tool_name

        if message.thinking:
            data["thinking"] = message.thinking
        
        if message.tool_calls:
            data["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                }
                for call in message.tool_calls
            ]
        
        return data

    def chat(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
    ) -> ModelResponse:

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                self._convert_message(message)
                for message in messages
            ],
            "stream": False,
        }

        if tools:
            payload["tools"] = [
                self._convert_tool(tool)
                for tool in tools
            ]

        response = self._client.post(
            "/api/chat",
            json=payload,
        )

        response.raise_for_status()

        data = response.json()

        message = data["message"]

        tool_calls: list[ToolCall] = []

        for item in message.get("tool_calls", []):
            function = item["function"]

            tool_calls.append(ToolCall(
                name=function["name"],
                arguments=function.get("arguments", {})
            ))

        return ModelResponse(
            content=message.get("content", ""),
            thinking=message.get("thinking") or "",
            tool_calls=tool_calls,
            done_reason=data.get("done_reason"),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
