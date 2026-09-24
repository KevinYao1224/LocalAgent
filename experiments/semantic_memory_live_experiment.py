"""Real Ollama embedding smoke test (requires an embedding-enabled server).

    .venv/bin/python experiments/semantic_memory_live_experiment.py --model <embedding-model>
"""

import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.embedding import OllamaEmbedder
from memory import SQLiteSemanticMemory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Installed Ollama embedding model")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()

    with TemporaryDirectory() as folder, OllamaEmbedder(
        args.model, base_url=args.base_url, timeout=180.0,
    ) as embedder:
        store = SQLiteSemanticMemory(Path(folder) / "memory.sqlite3", "demo", embedder)
        target = store.write("Cedar project deployment region: Tokyo", {"kind": "project"})
        store.write("Cedar project product color: blue", {"kind": "project"})
        store.write("Cedar deployment notes: previous region was Paris", {"kind": "archive"})
        reopened = SQLiteSemanticMemory(Path(folder) / "memory.sqlite3", "demo", embedder)
        hits = reopened.search_with_scores(
            "Where is the project hosted?", limit=2, metadata_filter={"kind": "project"},
        )

    for hit in hits:
        print(f"id={hit.record.id} similarity={hit.similarity:.4f} text={hit.record.text!r}")
    assert len(hits) == 2 and target.id in [hit.record.id for hit in hits]
    rank = [hit.record.id for hit in hits].index(target.id) + 1
    print(f"expected_source_rank={rank}/2")
    print("Persistence, metadata filtering, cosine ranking and source: PASS")
    print("This one query is a smoke test, not a retrieval-quality evaluation.")


if __name__ == "__main__":
    main()
