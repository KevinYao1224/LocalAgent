"""Phase 7C 完整轮次字符预算的离线实验。

从项目根目录运行：

    .venv/bin/python experiments/history_character_budget_experiment.py
"""

from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import AgentLoop, AgentSession
from llm.base import LLM, Message, ModelResponse, ToolCall
from memory import ConversationMemory
from runtime import ToolExecutor
from tools import ToolRegistry, add_tool


class RecordingLLM(LLM):
    """返回脚本化回答，同时保存每次模型调用实际收到的上下文。"""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = iter(responses)
        self.calls: list[list[Message]] = []

    def chat(self, messages, tools=None):
        self.calls.append(deepcopy(messages))
        return next(self._responses)


def make_agent(llm: LLM) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(add_tool)
    return AgentLoop(llm=llm, executor=ToolExecutor(registry), max_steps=3)


def simple_turn(label: str) -> list[Message]:
    return [
        Message(role="user", content=f"user-{label}"),
        Message(role="assistant", content=f"answer-{label}"),
    ]


def experiment_suffix_selection() -> None:
    memory = ConversationMemory(max_turns=5)
    for label in ("old", "middle", "new"):
        memory.append(simple_turn(label))

    newest_size = sum(len(message.content) for message in simple_turn("new"))
    middle_size = sum(len(message.content) for message in simple_turn("middle"))
    selection = memory.select(newest_size + middle_size)

    assert [message.content for message in selection.messages] == [
        "user-middle", "answer-middle", "user-new", "answer-new",
    ]
    assert selection.selected_turns == 2
    assert selection.dropped_turns == 1
    assert selection.selected_characters == newest_size + middle_size
    assert memory.turn_count == 3
    assert len(memory.retrieve()) == 6

    print(
        "Suffix selection: "
        f"selected={selection.selected_turns}, "
        f"dropped={selection.dropped_turns}, "
        f"characters={selection.selected_characters} (passed)"
    )


def experiment_oversized_newest_turn() -> None:
    memory = ConversationMemory(max_turns=3)
    memory.append(simple_turn("small"))
    long_turn = [
        Message(role="user", content="load"),
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(name="add", arguments={"a": 1, "b": 2})],
        ),
        Message(role="tool", content="x" * 100, tool_name="add"),
        Message(role="assistant", content="done"),
    ]
    memory.append(long_turn)

    selection = memory.select(50)
    assert selection.messages == []
    assert selection.selected_turns == 0
    assert selection.dropped_turns == 2
    assert selection.selected_characters == 0
    assert len(memory.retrieve()) == 6

    print(
        "Oversized newest turn: selected=0, dropped=2; "
        "the tool exchange remains stored and uncut (passed)"
    )


def experiment_session_integration() -> None:
    memory = ConversationMemory(max_turns=5)
    memory.append(simple_turn("hidden"))
    memory.append(simple_turn("visible"))
    visible_size = sum(
        len(message.content) for message in simple_turn("visible")
    )
    llm = RecordingLLM([ModelResponse(content="current-answer")])
    session = AgentSession(
        make_agent(llm),
        system_prompt="system",
        memory=memory,
        history_character_budget=visible_size,
    )

    result = session.run("current-user")
    selection = session.last_history_selection
    assert selection is not None
    assert selection.selected_turns == 1
    assert selection.dropped_turns == 1
    assert [message.content for message in llm.calls[0]] == [
        "system", "user-visible", "answer-visible", "current-user",
    ]
    assert result.completed
    assert memory.turn_count == 3

    print(
        "AgentSession context: system + 1 selected history turn + current user; "
        "completed current turn was committed (passed)"
    )


def experiment_boundaries_and_snapshots() -> None:
    memory = ConversationMemory(max_turns=2)
    memory.append(simple_turn("one"))
    exact_size = sum(len(message.content) for message in simple_turn("one"))

    exact = memory.select(exact_size)
    assert exact.selected_turns == 1
    assert memory.select().selected_turns == memory.turn_count
    assert memory.select(0).selected_turns == 0
    exact.messages[0].content = "mutated"
    assert memory.retrieve()[0].content == "user-one"

    for invalid in (-1, 1.5, True, "10"):
        try:
            memory.select(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid budget accepted: {invalid!r}")

        try:
            AgentSession(
                make_agent(RecordingLLM([])),
                history_character_budget=invalid,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"AgentSession accepted invalid budget: {invalid!r}"
            )

    print("Exact/zero budgets, validation, and deep-copy isolation: passed")


if __name__ == "__main__":
    experiment_suffix_selection()
    experiment_oversized_newest_turn()
    experiment_session_integration()
    experiment_boundaries_and_snapshots()
    print("\nAll Phase 7C history character budget experiments passed.")
