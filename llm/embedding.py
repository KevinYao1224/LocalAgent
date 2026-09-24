"""Ollama /api/embed 适配器；与聊天模型接口相互独立。"""

import httpx


class OllamaEmbedder:
    def __init__(
        self, model: str, base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string")
        self.model_id = f"ollama:{model}"
        self.model = model
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def embed(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a nonempty string")
        response = self._client.post(
            "/api/embed", json={"model": self.model, "input": text},
        )
        response.raise_for_status()
        embeddings = response.json()["embeddings"]
        if not isinstance(embeddings, list) or len(embeddings) != 1:
            raise ValueError("Ollama must return one embedding for one input")
        return embeddings[0]  # 向量内容由 SQLiteSemanticMemory 校验。

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
