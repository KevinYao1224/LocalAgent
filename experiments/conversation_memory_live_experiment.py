"""Phase 7B: evaluate short-term memory with a real Ollama model.

Run from the project root:

    python experiments/conversation_memory_live_experiment.py --runs 3

The experiment uses fresh sessions for four different questions:

1. Can the model recall an exact fact from an earlier turn?
2. Does a newer fact replace an older one?
3. Does whole-turn eviction really make an old fact unavailable?
4. What does a long retained tool result do to prompt size and recall?

Results are exploratory rather than unit-test assertions because model output
can vary. The production memory implementation is not modified by this file.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import statistics
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import AgentLoop, AgentRunResult, AgentSession
from llm.ollama import OllamaClient
from memory import ConversationMemory
from runtime import ToolExecutor
from tools import Tool, ToolRegistry


DEFAULT_MODEL = "qwen3.5:9b"
SYSTEM_PROMPT = (
    "You are evaluating conversation memory. Follow the user's requested "
    "answer format exactly. Use only facts visible in the conversation or "
    "tool results. If a requested fact is absent, answer exactly UNKNOWN."
)
LONG_RESULT_MARKER = "VAULT-48291"


@dataclass(frozen=True, slots=True)
class ScenarioMetrics:
    """Observable result of one scenario, including its final query."""

    name: str
    correct: bool
    completed: bool
    turns: int
    query_prompt_tokens: int
    total_prompt_tokens: int
    retained_messages: int
    retained_characters: int
    latency_seconds: float
    final_content: str


def load_archive() -> str:
    """Return a deliberately long record with one fact near its middle."""

    prefix = "archived telemetry without actionable facts " * 180
    suffix = "historical diagnostics without actionable facts " * 180
    return f"{prefix}\nACCESS MARKER: {LONG_RESULT_MARKER}\n{suffix}"


archive_tool = Tool(
    name="load_archive",
    description="Load the long archive record containing its access marker.",
    parameters={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    handler=load_archive,
)


def create_executor(*, include_archive: bool = False) -> ToolExecutor:
    registry = ToolRegistry()
    if include_archive:
        registry.register(archive_tool)
    return ToolExecutor(registry)


def create_session(
    llm: OllamaClient,
    *,
    max_turns: int,
    include_archive: bool = False,
) -> AgentSession:
    agent = AgentLoop(
        llm=llm,
        executor=create_executor(include_archive=include_archive),
        max_steps=4,
    )
    return AgentSession(
        agent=agent,
        system_prompt=SYSTEM_PROMPT,
        memory=ConversationMemory(max_turns=max_turns),
    )


def prompt_tokens(result: AgentRunResult) -> int:
    return sum(
        step.model_response.prompt_tokens or 0
        for step in result.agent_steps
    )


def run_scenario(
    name: str,
    session: AgentSession,
    prompts: list[str],
    expected: str,
    *,
    reject: str | None = None,
) -> ScenarioMetrics:
    started = perf_counter()
    results = [session.run(prompt) for prompt in prompts]
    latency = perf_counter() - started

    final = results[-1]
    final_content = final.response.content.strip()
    retained = session.memory.retrieve()
    correct = expected.casefold() in final_content.casefold()
    if reject is not None:
        correct = correct and reject.casefold() not in final_content.casefold()

    return ScenarioMetrics(
        name=name,
        correct=correct and final.completed,
        completed=all(result.completed for result in results),
        turns=len(results),
        query_prompt_tokens=prompt_tokens(final),
        total_prompt_tokens=sum(prompt_tokens(result) for result in results),
        retained_messages=len(retained),
        retained_characters=sum(len(message.content) for message in retained),
        latency_seconds=latency,
        final_content=final_content,
    )


def run_suite(llm: OllamaClient) -> list[ScenarioMetrics]:
    recall = run_scenario(
        "recall",
        create_session(llm, max_turns=5),
        [
            "Remember that my project code is CEDAR-731. Reply exactly ACK.",
            "What is my project code? Reply with only the code.",
        ],
        expected="CEDAR-731",
    )

    update = run_scenario(
        "fact_update",
        create_session(llm, max_turns=5),
        [
            "My deployment region is BLUE-17. Reply exactly ACK.",
            "Update: my deployment region is now AMBER-42. Reply exactly ACK.",
            "What is my current deployment region? Reply with only the region.",
        ],
        expected="AMBER-42",
        reject="BLUE-17",
    )

    eviction = run_scenario(
        "whole_turn_eviction",
        create_session(llm, max_turns=2),
        [
            "Remember that my private label is OBSIDIAN-905. Reply exactly ACK.",
            "This is filler turn one. Reply exactly FILLER-ONE.",
            "This is filler turn two. Reply exactly FILLER-TWO.",
            "What is my private label? Reply only with the label, or UNKNOWN "
            "if it is absent from the conversation.",
        ],
        expected="UNKNOWN",
        reject="OBSIDIAN-905",
    )

    long_tool_result = run_scenario(
        "long_tool_result",
        create_session(llm, max_turns=5, include_archive=True),
        [
            "Call load_archive. Remember the ACCESS MARKER from its result, "
            "then reply exactly ACK.",
            "What was the archive ACCESS MARKER? Reply with only the marker.",
        ],
        expected=LONG_RESULT_MARKER,
    )

    return [recall, update, eviction, long_tool_result]


def print_metrics(run: int, metrics: ScenarioMetrics) -> None:
    print(
        f"run={run} scenario={metrics.name} correct={metrics.correct} "
        f"completed={metrics.completed} turns={metrics.turns} "
        f"query_prompt_tokens={metrics.query_prompt_tokens} "
        f"total_prompt_tokens={metrics.total_prompt_tokens} "
        f"retained_messages={metrics.retained_messages} "
        f"retained_characters={metrics.retained_characters} "
        f"latency={metrics.latency_seconds:.2f}s"
    )
    print(f"  final={metrics.final_content!r}")


def print_summary(all_metrics: list[ScenarioMetrics]) -> None:
    print("\n=== Phase 7B summary ===")
    names = dict.fromkeys(item.name for item in all_metrics)
    for name in names:
        items = [item for item in all_metrics if item.name == name]
        print(
            f"{name}: success_rate="
            f"{sum(item.correct for item in items) / len(items):.2%}, "
            f"avg_query_prompt_tokens="
            f"{statistics.mean(item.query_prompt_tokens for item in items):.2f}, "
            f"avg_retained_characters="
            f"{statistics.mean(item.retained_characters for item in items):.2f}, "
            f"avg_latency={statistics.mean(item.latency_seconds for item in items):.2f}s"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of full four-scenario suites.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434",
        help="Ollama API base URL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1.")

    all_metrics: list[ScenarioMetrics] = []
    with OllamaClient(
        model=args.model,
        base_url=args.base_url,
        timeout=180.0,
    ) as llm:
        for run in range(1, args.runs + 1):
            for metrics in run_suite(llm):
                all_metrics.append(metrics)
                print_metrics(run, metrics)

    print_summary(all_metrics)


if __name__ == "__main__":
    main()
