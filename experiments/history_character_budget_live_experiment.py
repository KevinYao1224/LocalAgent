"""Phase 7C 在线检查：将一轮超预算的工具交互整体排除。

Ollama 服务可用后，从项目根目录运行：

    .venv/bin/python experiments/history_character_budget_live_experiment.py
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import AgentLoop, AgentRunResult, AgentSession
from llm.ollama import OllamaClient
from memory import ConversationMemory
from observability import HumanReadableLogger
from runtime import ToolExecutor
from tools import Tool, ToolRegistry


DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_BUDGET = 4000
MARKER = "VAULT-48291"
SYSTEM_PROMPT = (
    "You are evaluating conversation memory. Follow the user's requested "
    "answer format exactly. Use only facts visible in the conversation or "
    "tool results. If a requested fact is absent, answer exactly UNKNOWN."
)


class OneShotArchive:
    """仅提供一次长 observation，避免模型重新调用工具获取该内容。"""

    def __init__(self) -> None:
        self._used = False

    def load(self) -> str:
        if self._used:
            raise RuntimeError("The archive is no longer available.")
        self._used = True

        prefix = "archived telemetry without actionable facts " * 180
        suffix = "historical diagnostics without actionable facts " * 180
        return f"{prefix}\nACCESS MARKER: {MARKER}\n{suffix}"


def prompt_tokens(result: AgentRunResult) -> int:
    return sum(
        step.model_response.prompt_tokens or 0
        for step in result.agent_steps
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live Phase 7C oversized-turn exclusion experiment."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = ToolRegistry()
    archive = OneShotArchive()
    registry.register(Tool(
        name="load_archive",
        description="Load the long archive record containing its access marker.",
        parameters={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        handler=archive.load,
    ))
    memory = ConversationMemory(max_turns=5)

    with OllamaClient(
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout,
    ) as llm:
        session = AgentSession(
            agent=AgentLoop(
                llm=llm,
                executor=ToolExecutor(registry),
                max_steps=4,
                logger=HumanReadableLogger(max_text_chars=300),
            ),
            system_prompt=SYSTEM_PROMPT,
            memory=memory,
            history_character_budget=args.budget,
        )

        first = session.run(
            "Call load_archive. Remember the ACCESS MARKER from its result, "
            "then reply exactly ACK."
        )
        stored_before_query = memory.retrieve()
        stored_characters = sum(
            len(message.content) for message in stored_before_query
        )

        second = session.run(
            "What was the archive ACCESS MARKER? Reply with only the marker."
        )

    selection = session.last_history_selection
    assert selection is not None
    final_content = second.response.content.strip()
    reacquisition_attempts = len(second.tool_results)
    successful_reacquisitions = sum(
        result.success for result in second.tool_results
    )
    marker_still_stored = any(
        MARKER in message.content for message in memory.retrieve()
    )
    passed = (
        first.completed
        and second.completed
        and selection.selected_turns == 0
        and selection.dropped_turns == 1
        and selection.selected_characters == 0
        and final_content == "UNKNOWN"
        and successful_reacquisitions == 0
        and marker_still_stored
    )

    print("\n=== Phase 7C live budget result ===")
    print(f"budget={args.budget}")
    print(f"stored_before_query_characters={stored_characters}")
    print(f"selected_turns={selection.selected_turns}")
    print(f"dropped_turns={selection.dropped_turns}")
    print(f"selected_characters={selection.selected_characters}")
    print(f"query_prompt_tokens={prompt_tokens(second)}")
    print(f"reacquisition_attempts={reacquisition_attempts}")
    print(f"successful_reacquisitions={successful_reacquisitions}")
    print(f"final_content={final_content!r}")
    print(f"marker_still_stored={marker_still_stored}")
    print(f"verdict={'PASS' if passed else 'FAIL'}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
