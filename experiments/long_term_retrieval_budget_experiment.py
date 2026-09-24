"""Phase 8B 离线实验；运行 .venv/bin/python experiments/long_term_retrieval_budget_experiment.py。"""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import AgentLoop, AgentSession
from llm.base import LLM, ModelResponse
from memory import SQLiteLongTermMemory
from runtime import ToolExecutor
from tools import ToolRegistry


class CapturingLLM(LLM):
    def __init__(self):
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append(messages)
        return ModelResponse(content="Acknowledged.")


def payload_size(record):
    return len(json.dumps([{
        "id": record.id, "created_at": record.created_at,
        "text": record.text, "metadata": record.metadata,
    }], ensure_ascii=False))


def main():
    with TemporaryDirectory() as folder:
        store = SQLiteLongTermMemory(Path(folder) / "memory.sqlite3", "demo")
        small = store.write("Cedar: small fact", {"kind": "fact"})
        large = store.write("Cedar: " + "x" * 10000, {"kind": "fact"})
        model = CapturingLLM()
        session = AgentSession(
            AgentLoop(model, ToolExecutor(ToolRegistry())),
            long_term_memory=store,
            retrieval_limit=2,
            retrieval_character_budget=payload_size(small),
        )
        result = session.run("What is known about Cedar?", memory_query="Cedar")
        selection = session.last_retrieval_selection
        assert result.completed
        assert [record.id for record in session.last_retrieval] == [small.id]
        assert (selection.candidate_count, selection.selected_count,
                selection.dropped_count, selection.payload_characters) == (
                    2, 1, 1, payload_size(small))
        assert large.id not in model.calls[0][-2].content
        assert small.id in model.calls[0][-2].content
        assert len(model.calls[0][-2].content.split("\n", 1)[1]) <= selection.character_budget
        assert all(small.id not in m.content for m in session.memory.retrieve())
        print(f"retrieval: candidates=2 selected=1 dropped=1 JSON chars={selection.payload_characters}")

        session.run("No retrieval this turn")
        assert session.last_retrieval == ()
        assert session.last_retrieval_selection is None
        assert all(small.id not in m.content for m in model.calls[-1])
        print("no implicit retrieval or short-term persistence: PASS")

        zero_model = CapturingLLM()
        zero = AgentSession(
            AgentLoop(zero_model, ToolExecutor(ToolRegistry())),
            long_term_memory=store, retrieval_character_budget=0,
        )
        zero.run("Look up Cedar", memory_query="Cedar")
        assert zero.last_retrieval_selection.dropped_count == 2
        assert zero.last_retrieval_selection.payload_characters == 0
        assert len(zero_model.calls[0]) == 1  # 不添加空的检索消息。
        assert len(store.search("Cedar")) == 2  # 选择过程没有删除记录。
        print("zero budget excludes whole records without changing SQLite: PASS")

        default_model = CapturingLLM()
        default = AgentSession(
            AgentLoop(default_model, ToolExecutor(ToolRegistry())),
            long_term_memory=store,
        )
        default.run("Look up Cedar", memory_query="Cedar")
        assert [record.id for record in default.last_retrieval] == [large.id, small.id]
        assert default.last_retrieval_selection.dropped_count == 0
        assert default.last_retrieval_selection.character_budget is None
        assert large.id in default_model.calls[0][-2].content
        print("default unlimited retrieval preserves Phase 8A behavior: PASS")

        for invalid in (-1, True, 1.5):
            try:
                AgentSession(
                    AgentLoop(CapturingLLM(), ToolExecutor(ToolRegistry())),
                    retrieval_character_budget=invalid,
                )
            except ValueError:
                pass
            else:
                raise AssertionError(f"invalid budget accepted: {invalid!r}")
        print("budget validation: PASS")
        print("All Phase 8B experiments passed.")


if __name__ == "__main__":
    main()
