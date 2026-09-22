"""Visually inspect Phase 7B memory scenarios with the standard agent stack.

Run every scenario:

    python main.py

Run one scenario when you want a shorter trace:

    python main.py --scenario recall
    python main.py --scenario fact-update
    python main.py --scenario eviction
    python main.py --scenario long-tool
"""

import argparse
from dataclasses import dataclass

from agent import AgentLoop, AgentRunResult, AgentSession
from llm.ollama import OllamaClient
from memory import ConversationMemory
from observability import HumanReadableLogger
from runtime import ToolExecutor
from tools import Tool, ToolRegistry


DEFAULT_MODEL = "qwen3.5:9b"
SYSTEM_PROMPT = (
    "You are evaluating conversation memory. Follow the user's requested "
    "answer format exactly. Use only facts visible in the conversation or "
    "tool results. If a requested fact is absent, answer exactly UNKNOWN."
)
LONG_RESULT_MARKER = "VAULT-48291"
LOGGER_PREVIEW_CHARS = 500
MEMORY_PREVIEW_CHARS = 100


@dataclass(frozen=True, slots=True)
class Scenario:
    """One human-inspectable sequence of user turns."""

    key: str
    title: str
    max_turns: int
    prompts: tuple[str, ...]
    expected: str
    rejected: str | None = None
    include_archive_tool: bool = False


def load_archive() -> str:
    """Return a long tool observation with one marker near its middle."""

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


SCENARIOS = (
    Scenario(
        key="recall",
        title="Cross-turn fact recall",
        max_turns=5,
        prompts=(
            "Remember that my project code is CEDAR-731. Reply exactly ACK.",
            "What is my project code? Reply with only the code.",
        ),
        expected="CEDAR-731",
    ),
    Scenario(
        key="fact-update",
        title="New fact replaces old fact",
        max_turns=5,
        prompts=(
            "My deployment region is BLUE-17. Reply exactly ACK.",
            "Update: my deployment region is now AMBER-42. Reply exactly ACK.",
            "What is my current deployment region? Reply with only the region.",
        ),
        expected="AMBER-42",
        rejected="BLUE-17",
    ),
    Scenario(
        key="eviction",
        title="Whole-turn eviction",
        max_turns=2,
        prompts=(
            "Remember that my private label is OBSIDIAN-905. Reply exactly ACK.",
            "This is filler turn one. Reply exactly FILLER-ONE.",
            "This is filler turn two. Reply exactly FILLER-TWO.",
            "What is my private label? Reply only with the label, or UNKNOWN "
            "if it is absent from the conversation.",
        ),
        expected="UNKNOWN",
        rejected="OBSIDIAN-905",
    ),
    Scenario(
        key="long-tool",
        title="Recall from a long retained tool result",
        max_turns=5,
        prompts=(
            "Call load_archive. Remember the ACCESS MARKER from its result, "
            "then reply exactly ACK.",
            "What was the archive ACCESS MARKER? Reply with only the marker.",
        ),
        expected=LONG_RESULT_MARKER,
        include_archive_tool=True,
    ),
)


def create_session(llm: OllamaClient, scenario: Scenario) -> AgentSession:
    """Build the same public AgentSession stack an application would use."""

    registry = ToolRegistry()
    if scenario.include_archive_tool:
        registry.register(archive_tool)

    agent = AgentLoop(
        llm=llm,
        executor=ToolExecutor(registry),
        max_steps=4,
        logger=HumanReadableLogger(max_text_chars=LOGGER_PREVIEW_CHARS),
    )
    return AgentSession(
        agent=agent,
        system_prompt=SYSTEM_PROMPT,
        memory=ConversationMemory(max_turns=scenario.max_turns),
    )


def prompt_tokens(result: AgentRunResult) -> int:
    """Sum provider-reported prompt tokens for one user turn."""

    return sum(
        step.model_response.prompt_tokens or 0
        for step in result.agent_steps
    )


def preview(text: str, limit: int = MEMORY_PREVIEW_CHARS) -> str:
    """Keep the explicit memory snapshot readable in a terminal."""

    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}... <{len(compact) - limit} chars omitted>"


def print_memory(session: AgentSession) -> None:
    """Show exactly which canonical messages survive after the turn."""

    messages = session.memory.retrieve()
    characters = sum(len(message.content) for message in messages)
    print(
        f"\n[Memory snapshot] turns={session.memory.turn_count}, "
        f"messages={len(messages)}, content_characters={characters}"
    )
    for index, message in enumerate(messages, start=1):
        tool = f", tool={message.tool_name}" if message.tool_name else ""
        calls = (
            f", calls={[call.name for call in message.tool_calls]}"
            if message.tool_calls else ""
        )
        print(
            f"  {index}. role={message.role}{tool}{calls}: "
            f"{preview(message.content)!r}"
        )


def run_scenario(llm: OllamaClient, scenario: Scenario) -> bool:
    """Run one scenario and print traces, memory snapshots, and verdict."""

    print("\n" + "=" * 78)
    print(f"SCENARIO: {scenario.title} ({scenario.key})")
    print(f"Memory capacity: {scenario.max_turns} complete turns")
    print(f"Expected final answer contains: {scenario.expected!r}")
    if scenario.rejected:
        print(f"Expected final answer excludes: {scenario.rejected!r}")
    print("=" * 78)

    session = create_session(llm, scenario)
    results: list[AgentRunResult] = []

    for turn, user_content in enumerate(scenario.prompts, start=1):
        print(f"\n>>> USER TURN {turn}: {user_content}")
        result = session.run(user_content)
        results.append(result)
        print(
            f"\n[Turn result] completed={result.completed}, "
            f"steps={result.steps}, prompt_tokens={prompt_tokens(result)}, "
            f"answer={result.response.content.strip()!r}"
        )
        print_memory(session)

    final_content = results[-1].response.content.casefold()
    passed = (
        all(result.completed for result in results)
        and scenario.expected.casefold() in final_content
        and (
            scenario.rejected is None
            or scenario.rejected.casefold() not in final_content
        )
    )
    print(f"\nSCENARIO VERDICT: {'PASS' if passed else 'FAIL'}")
    return passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visual Phase 7B AgentSession and Logger verification."
    )
    parser.add_argument(
        "--scenario",
        choices=("all", *(scenario.key for scenario in SCENARIOS)),
        default="all",
        help="Run all scenarios or select one shorter trace.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = (
        SCENARIOS
        if args.scenario == "all"
        else tuple(
            scenario for scenario in SCENARIOS
            if scenario.key == args.scenario
        )
    )

    print(f"Model: {args.model}")
    print(f"Scenarios: {', '.join(item.key for item in selected)}")
    print(
        f"Logger previews text after {LOGGER_PREVIEW_CHARS} characters; "
        "model context and memory remain complete."
    )

    with OllamaClient(
        model=args.model,
        base_url=args.base_url,
        timeout=180.0,
    ) as llm:
        verdicts = [run_scenario(llm, scenario) for scenario in selected]

    passed = sum(verdicts)
    print("\n" + "=" * 78)
    print(f"FINAL VERDICT: {passed}/{len(verdicts)} scenarios passed")
    print("=" * 78)
    if passed != len(verdicts):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
