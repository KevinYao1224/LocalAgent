"""Phase 8B Ollama 实验：比较有预算和无预算的长期记忆检索。

    .venv/bin/python experiments/long_term_retrieval_budget_live_experiment.py
"""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import AgentLoop, AgentSession
from llm.ollama import OllamaClient
from memory import SQLiteLongTermMemory
from observability import HumanReadableLogger
from runtime import ToolExecutor
from tools import ToolRegistry


SHORT_CODE = "CEDAR-731"
LONG_CODE = "VAULT-48291"
SYSTEM = (
    "Answer using only the retrieved reference data. If the requested code is "
    "absent, answer exactly UNKNOWN. Otherwise answer with only the code."
)


def payload_size(record):
    return len(json.dumps([{
        "id": record.id, "created_at": record.created_at,
        "text": record.text, "metadata": record.metadata,
    }], ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--scenario",
        choices=("all", "unbounded_archive", "bounded_project", "bounded_archive", "zero_budget"),
        default="all",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with TemporaryDirectory() as folder:
        store = SQLiteLongTermMemory(Path(folder) / "demo.sqlite3", "live-demo")
        short = store.write(f"Cedar project code: {SHORT_CODE}", {"kind": "project"})
        long = store.write(
            "Cedar archived telemetry " * 350
            + f"\nCedar archive access code: {LONG_CODE}\n"
            + "Cedar historical telemetry " * 350,
            {"kind": "archive"},
        )
        budget = payload_size(short)
        print(f"record sizes: short_json={budget}, long_json={payload_size(long)}")

        scenarios = (
            ("unbounded_archive", None, "What is the Cedar archive access code?", LONG_CODE,
             (long.id, short.id)),
            ("bounded_project", budget, "What is the Cedar project code?", SHORT_CODE,
             (short.id,)),
            ("bounded_archive", budget, "What is the Cedar archive access code?", "UNKNOWN",
             (short.id,)),
            ("zero_budget", 0, "What is the Cedar project code?", "UNKNOWN", ()),
        )

        outcomes = []
        with OllamaClient(args.model, base_url=args.base_url, timeout=args.timeout) as llm:
            for name, character_budget, question, expected, ids in scenarios:
                if args.scenario not in ("all", name):
                    continue
                print(f"\n=== {name} ===", flush=True)
                session = AgentSession(
                    AgentLoop(
                        llm, ToolExecutor(ToolRegistry()), max_steps=2,
                        logger=HumanReadableLogger(max_text_chars=180),
                    ),
                    system_prompt=SYSTEM,
                    long_term_memory=store,
                    retrieval_limit=2,
                    retrieval_character_budget=character_budget,
                )
                started = perf_counter()
                result = session.run(question, memory_query="Cedar")
                elapsed = perf_counter() - started
                selection = session.last_retrieval_selection
                actual_ids = tuple(record.id for record in session.last_retrieval)
                assert actual_ids == ids, (name, actual_ids, ids)
                assert selection.selected_count == len(ids)
                if character_budget is not None:
                    assert selection.payload_characters <= character_budget
                # 检查实际的规范输入，而不只检查选择元数据。
                input_text = "\n".join(m.content for m in result.messages if m.role == "user")
                for record in (short, long):
                    assert (record.id in input_text) == (record.id in ids)

                answer = result.response.content.strip()
                tokens = sum(step.model_response.prompt_tokens or 0 for step in result.agent_steps)
                correct = result.completed and answer == expected
                print(
                    f"budget={character_budget} candidates={selection.candidate_count} "
                    f"selected={selection.selected_count} dropped={selection.dropped_count} "
                    f"json_characters={selection.payload_characters} "
                    f"prompt_tokens={tokens} latency={elapsed:.2f}s"
                )
                print(f"expected={expected!r} actual={answer!r} model_correct={correct}")
                outcomes.append((name, correct))

    print("\n=== Phase 8B live summary ===")
    print(f"context_selection={len(outcomes)}/{len(outcomes)} verified")
    print(f"model_answers={sum(correct for _, correct in outcomes)}/{len(outcomes)} correct")
    for name, correct in outcomes:
        print(f"  {name}: {'correct' if correct else 'incorrect'}")


if __name__ == "__main__":
    main()
