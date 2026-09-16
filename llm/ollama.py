from typing import Any

import httpx

from llm.base import LLM, Message, ModelResponse, ToolCall


class OllamaClient(LLM):

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

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                message.to_dict()
                for message in messages
            ],
            "stream": False,
        }

        if tools:
            payload["tools"] = tools

        response = self._client.post(
            "/api/chat",
            json=payload,
        )

        response.raise_for_status()

        data = response.json()

        message = data["message"]

        tool_calls: list[ToolCall] = []

        # 只处理 function 类型的 tool 调用
        for item in message.get("tool_calls", []):
            function = item["function"]

            tool_calls.append(
                ToolCall(
                    name=function["name"],
                    arguments=function.get("arguments", {}),
                )
            )

        return ModelResponse(
            content=message.get("content", ""),
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