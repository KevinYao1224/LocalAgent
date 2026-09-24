"""Small live retrieval evaluation; optionally test the full retrieval-to-answer path.

    .venv/bin/python experiments/semantic_retrieval_eval_experiment.py
    .venv/bin/python experiments/semantic_retrieval_eval_experiment.py --chat
"""

import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import AgentLoop, AgentSession
from llm.embedding import OllamaEmbedder
from llm.ollama import OllamaClient
from memory import SQLiteLongTermMemory, SQLiteSemanticMemory
from observability import HumanReadableLogger
from runtime import ToolExecutor
from tools import ToolRegistry


FACTS = (
    ("current", "Cedar's current deployment region is Tokyo."),
    ("color", "Cedar's product color is blue."),
    ("owner", "Maya owns the Cedar project."),
    ("archive", "Cedar's previous deployment region was Paris."),
)
CASES = (
    ("Where is Cedar hosted now?", "current", "Tokyo"),
    ("What shade is the Cedar product?", "color", "blue"),
    ("Who is responsible for Cedar?", "owner", "Maya"),
    ("Where did Cedar run before?", "archive", "Paris"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-model", default="qwen3-embedding:0.6b")
    parser.add_argument("--chat-model", default="qwen3.5:9b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--chat", action="store_true", help="Run one full AgentSession answer")
    args = parser.parse_args()

    with TemporaryDirectory() as folder, OllamaEmbedder(
        args.embedding_model, args.base_url, timeout=180.0,
    ) as embedder:
        store = SQLiteSemanticMemory(Path(folder) / "demo.sqlite3", "demo", embedder)
        records = {kind: store.write(text, {"kind": kind}) for kind, text in FACTS}
        literal = SQLiteLongTermMemory(Path(folder) / "demo.sqlite3", "demo")
        correct = 0
        print("query | expected rank | literal hits | semantic top-2 (cosine)")
        for query, expected, _ in CASES:
            hits = store.search_with_scores(query, limit=4)
            rank = [hit.record.id for hit in hits].index(records[expected].id) + 1
            correct += rank == 1
            preview = ", ".join(
                f"{hit.record.metadata['kind']}={hit.similarity:.3f}" for hit in hits[:2]
            )
            print(f"{query!r} | {rank}/4 | {len(literal.search(query))} | {preview}")
        print(f"semantic_top1={correct}/{len(CASES)}")
        print("Literal baseline matches the entire question as a substring; it is not keyword search.")

        if args.chat:
            query, expected, answer = CASES[0]
            with OllamaClient(args.chat_model, args.base_url, timeout=300.0) as llm:
                session = AgentSession(
                    AgentLoop(
                        llm, ToolExecutor(ToolRegistry()), max_steps=2,
                        logger=HumanReadableLogger(max_text_chars=240),
                    ),
                    system_prompt=(
                        "Answer the question using only retrieved reference data. "
                        "If it does not contain the answer, say UNKNOWN. "
                        "Reply with only the requested location."
                    ),
                    long_term_memory=store, retrieval_limit=1,
                    retrieval_character_budget=500,
                )
                result = session.run(query, memory_query=query)
            source_ids = [record.id for record in session.last_retrieval]
            tokens = sum(step.model_response.prompt_tokens or 0 for step in result.agent_steps)
            print(f"source_ids={source_ids} expected_source={records[expected].id}")
            print(f"payload_characters={session.last_retrieval_selection.payload_characters} "
                  f"prompt_tokens={tokens}")
            print(f"answer={result.response.content.strip()!r} expected={answer!r}")
            print(f"source_correct={source_ids == [records[expected].id]} "
                  f"answer_correct={result.response.content.strip() == answer}")


if __name__ == "__main__":
    main()
