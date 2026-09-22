"""Offline Phase 7A experiments: run python experiments/conversation_memory_experiment.py."""

from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import AgentLoop, AgentSession, ContextBuilder, ReasoningReplayPolicy
from llm.base import (
    LLM, LLMCapabilities, Message, ModelResponse, ReasoningReplaySupport, ToolCall,
)
from memory import ConversationMemory
from runtime import ToolExecutor
from tools import ToolRegistry, add_tool


class ScriptedLLM(LLM):
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls: list[list[Message]] = []

    @property
    def model_name(self):
        return "memory-test"

    @property
    def capabilities(self):
        return LLMCapabilities(reasoning_replay=ReasoningReplaySupport.SUPPORTED)

    def chat(self, messages, tools=None):
        self.calls.append(deepcopy(messages))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def make_agent(llm, max_steps=5, replay=False):
    registry = ToolRegistry()
    registry.register(add_tool)
    return AgentLoop(
        llm=llm,
        executor=ToolExecutor(registry),
        max_steps=max_steps,
        context_builder=ContextBuilder(
            reasoning_replay=(
                ReasoningReplayPolicy.LATEST_PENDING if replay
                else ReasoningReplayPolicy.OFF
            ),
        ),
    )


def experiment_cross_turn_history():
    class RecallLLM(LLM):
        def chat(self, messages, tools=None):
            if messages[-1].content == "What is my project name?":
                remembered = any(
                    message.role == "user" and message.content == "My project is Cedar."
                    for message in messages[:-1]
                )
                return ModelResponse(content="Cedar" if remembered else "Unknown")
            return ModelResponse(content="Noted.")

    agent = make_agent(RecallLLM())
    session = AgentSession(agent, system_prompt="Remember the conversation.")
    first = session.run("My project is Cedar.")
    second = session.run("What is my project name?")
    assert second.response.content == "Cedar"
    assert AgentSession(agent).run("What is my project name?").response.content == "Unknown"
    assert first.state is not second.state
    assert first.steps == second.steps == 1
    assert second.state.current_task == "What is my project name?"
    assert [m.role for m in second.messages] == [
        "system", "user", "assistant", "user", "assistant",
    ]
    assert len(session.memory.retrieve()) == 4
    assert session.memory.turn_count == 2
    session.memory.clear()
    assert session.run("What is my project name?").response.content == "Unknown"
    assert session.memory.turn_count == 1
    print("Cross-turn recall, session isolation, fresh state, delta commit, clear: passed.")


def experiment_tool_turn_eviction_and_replay():
    llm = ScriptedLLM([
        ModelResponse(thinking="Add first.", tool_calls=[
            ToolCall(name="add", arguments={"a": 1, "b": 2}),
            ToolCall(name="add", arguments={"a": 3, "b": 4}),
        ]),
        ModelResponse(content="Results: 3 and 7.", thinking="Ready to answer."),
        ModelResponse(content="The results were 3 and 7."),
        ModelResponse(content="Third answer."),
    ])
    memory = ConversationMemory(max_turns=1)
    session = AgentSession(make_agent(llm, replay=True), "Use tools.", memory)
    first = session.run("Add two independent pairs.")
    assert llm.calls[1][2].thinking == "Add first."
    saved = memory.retrieve()
    assert [m.role for m in saved] == ["user", "assistant", "tool", "tool", "assistant"]
    assert [m.content for m in saved if m.role == "tool"] == ["3", "7"]
    assert all(not m.thinking for m in saved)
    assert first.agent_steps[0].reasoning.raw_thinking == "Add first."

    # Neither modifying the result nor a retrieved nested argument changes memory.
    first.agent_steps[0].model_response.tool_calls[0].arguments["a"] = 999
    saved[1].tool_calls[0].arguments["b"] = 999
    assert memory.retrieve()[1].tool_calls[0].arguments == {"a": 1, "b": 2}

    session.run("Repeat the results.")
    assert [m.role for m in llm.calls[2]] == [
        "system", "user", "assistant", "tool", "tool", "assistant", "user",
    ]
    assert all(not m.thinking for m in llm.calls[2])
    assert memory.turn_count == 1
    session.run("Third request.")
    assert [m.role for m in llm.calls[3]] == ["system", "user", "assistant", "user"]
    assert llm.calls[3][1].content == "Repeat the results."
    assert len(memory.retrieve()) == 2
    print("Whole-turn eviction, pinned system prompt, nested snapshots, replay isolation: passed.")


def experiment_failed_runs_do_not_commit():
    llm = ScriptedLLM([
        ModelResponse(content="Initial answer."),
        ModelResponse(tool_calls=[ToolCall(name="add", arguments={"a": 1, "b": 2})]),
        RuntimeError("provider unavailable"),
        ModelResponse(content="Recovered."),
    ])
    agent = make_agent(llm, max_steps=1)
    session = AgentSession(agent)
    session.run("Initial request.")
    baseline = session.memory.retrieve()
    incomplete = session.run("Exhaust the step budget.")
    assert not incomplete.completed and incomplete.tool_results[0].value == 3
    assert session.memory.retrieve() == baseline
    try:
        session.run("Fail the provider.")
    except RuntimeError as exc:
        assert str(exc) == "provider unavailable"
    else:
        raise AssertionError("Expected provider failure")
    assert agent.last_state.current_task == "Fail the provider."
    assert session.memory.retrieve() == baseline
    session.run("Recover.")
    assert [m.content for m in llm.calls[-1]] == [
        "Initial request.", "Initial answer.", "Recover.",
    ]
    print("max_steps and provider errors preserve memory; executed tools are not rolled back: passed.")


def experiment_invalid_turns_are_atomic():
    memory = ConversationMemory(max_turns=1)
    memory.append([Message("user", "Valid"), Message("assistant", "Answer")])
    baseline = memory.retrieve()
    call = ToolCall(name="add", arguments={"a": 1, "b": 2})
    invalid_turns = [
        [],
        [Message("user", "No answer")],
        [Message("user", "Orphan"), Message("tool", "3", tool_name="add"), Message("assistant", "3")],
        [Message("user", "Missing tool"), Message("assistant", "", tool_calls=[call]), Message("assistant", "3")],
        [Message("user", "Wrong tool"), Message("assistant", "", tool_calls=[call]), Message("tool", "3", tool_name="other"), Message("assistant", "3")],
        [Message("user", "Empty"), Message("assistant", " ")],
        [Message("user", "Two turns"), Message("assistant", "Answer"), Message("user", "Again"), Message("assistant", "Answer")],
    ]
    for turn in invalid_turns:
        try:
            memory.append(turn)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed turn accepted")
        assert memory.retrieve() == baseline
    print("Incomplete or malformed turns rejected before memory mutation: passed.")


if __name__ == "__main__":
    experiment_cross_turn_history()
    experiment_tool_turn_eviction_and_replay()
    experiment_failed_runs_do_not_commit()
    experiment_invalid_turns_are_atomic()
    print("\nAll conversation memory experiments passed.")
