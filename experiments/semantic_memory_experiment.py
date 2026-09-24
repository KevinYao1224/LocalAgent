"""Offline Phase 8C: .venv/bin/python experiments/semantic_memory_experiment.py"""

import json
import httpx
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import AgentLoop, AgentSession
from llm.base import LLM, ModelResponse
from llm.embedding import OllamaEmbedder
from memory import SQLiteLongTermMemory, SQLiteSemanticMemory
from runtime import ToolExecutor
from tools import ToolRegistry


class FixedEmbedder:
    """Known geometry, not a fake answer generator: searches still use cosine."""

    def __init__(self, model_id="demo:v1", dimensions=2):
        self.model_id = model_id
        self.dimensions = dimensions

    def embed(self, text):
        if "deploy" in text or "region" in text or "东京" in text:
            vector = [1.0, 0.0]
        elif "color" in text or "blue" in text:
            vector = [0.0, 1.0]
        else:
            vector = [-1.0, 0.0]
        return vector + [0.0] * (self.dimensions - 2)


class CaptureLLM(LLM):
    def __init__(self):
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append(messages)
        return ModelResponse(content="Acknowledged.")


def reject(action):
    try:
        action()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def main():
    def mock_embed(request):
        assert request.url.path == "/api/embed"
        assert json.loads(request.content) == {"model": "demo", "input": "hello"}
        return httpx.Response(200, json={"embeddings": [[0.6, 0.8]]})

    with OllamaEmbedder("demo") as adapter:
        adapter._client.close()
        adapter._client = httpx.Client(transport=httpx.MockTransport(mock_embed),
                                       base_url="http://127.0.0.1:11434")
        assert adapter.embed("hello") == [0.6, 0.8]
        assert adapter.model_id == "ollama:demo"
    print("Ollama /api/embed request/response adapter: PASS")

    with TemporaryDirectory() as folder:
        path = Path(folder) / "memory.sqlite3"
        alice = SQLiteSemanticMemory(path, "alice", FixedEmbedder())
        bob = SQLiteSemanticMemory(path, "bob", FixedEmbedder())
        project = alice.write("Cedar deployment is in 东京", {"kind": "project"})
        alice.write("Cedar color is blue", {"kind": "note"})
        alice.write("Another color note", {"kind": "other"})
        bob.write("Bob deployment secret", {"kind": "project"})
        reopened = SQLiteSemanticMemory(path, "alice", FixedEmbedder())
        # Query shares no literal substring with the top record.
        hits = reopened.search_with_scores("Where is the region?", 1,
                                          {"kind": "project"})
        assert hits[0].record.id == project.id
        assert hits[0].similarity > 0.99
        assert reopened.search("region", 1, {"kind": "project"})[0].id == project.id
        assert SQLiteLongTermMemory(path, "alice").search("region") == []
        assert all("Bob" not in hit.text for hit in reopened.search("deployment"))
        print(f"semantic hit={hits[0].record.id} cosine={hits[0].similarity:.3f} "
              "(literal search misses it)")

        other_model = SQLiteSemanticMemory(path, "alice", FixedEmbedder("demo:v2"))
        assert other_model.search("region") == []
        reject(lambda: SQLiteSemanticMemory(
            path, "alice", FixedEmbedder("demo:v1", 3)).write("deploy something"))
        assert SQLiteLongTermMemory(path, "alice").search("deploy something") == []
        reject(lambda: reopened.search("region", limit=0))
        reject(lambda: reopened.write("bad", {"x": object()}))

        class ZeroEmbedder(FixedEmbedder):
            def embed(self, text):
                return [0.0, 0.0]

        reject(lambda: SQLiteSemanticMemory(path, "alice", ZeroEmbedder()).write("bad vector"))
        assert SQLiteLongTermMemory(path, "alice").search("bad vector") == []
        print("namespace/model isolation, dimension validation, atomic write: PASS")

        llm = CaptureLLM()
        session = AgentSession(
            AgentLoop(llm, ToolExecutor(ToolRegistry())), long_term_memory=reopened,
            retrieval_limit=1, retrieval_character_budget=1000,
        )
        result = session.run("Where is the project deployed?", memory_query="region")
        assert result.completed
        assert [record.id for record in session.last_retrieval] == [project.id]
        assert project.id in llm.calls[0][-2].content
        assert session.last_retrieval_selection.selected_count == 1
        assert all(project.id not in message.content for message in session.memory.retrieve())
        payload = json.loads(llm.calls[0][-2].content.split("\n", 1)[1])
        assert payload[0]["text"] == project.text
        print("AgentSession explicit semantic retrieval + Phase 8B budget: PASS")
        print("All Phase 8C experiments passed.")


if __name__ == "__main__":
    main()
