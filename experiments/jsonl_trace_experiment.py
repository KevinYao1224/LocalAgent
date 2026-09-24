"""Read back a versioned Phase 12 JSONL trace without a model server.

Run: .venv/bin/python experiments/jsonl_trace_experiment.py
"""

from datetime import datetime
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import AgentLoop
from llm.base import LLM, Message, ModelResponse, ToolCall
from observability import CompositeLogger, HumanReadableLogger, JsonlTraceLogger
from runtime import ToolExecutor
from tools import Tool, ToolRegistry


SECRET = "PRIVATE-CEDAR-731"


class ScriptedLLM(LLM):
    def __init__(self, responses):
        self.responses = iter(responses)

    def chat(self, messages, tools=None):
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def executor() -> ToolExecutor:
    def echo(value: str) -> str:
        if value == "bad":
            raise ValueError("bad input")
        return value

    registry = ToolRegistry()
    registry.register(Tool(
        name="echo",
        description="Return an input, or fail on 'bad'.",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        handler=echo,
    ))
    return ToolExecutor(registry)


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def completed_and_appended(path: Path) -> None:
    terminal = StringIO()
    logger = CompositeLogger(
        HumanReadableLogger(stream=terminal, max_text_chars=80),
        JsonlTraceLogger(path),
    )
    loop = AgentLoop(
        ScriptedLLM([
            ModelResponse(
                tool_calls=[ToolCall("echo", {"value": SECRET}),
                            ToolCall("echo", {"value": "bad"})],
                prompt_tokens=10, completion_tokens=3,
            ),
            ModelResponse(thinking="Working with " + SECRET),
            ModelResponse(content="done " + SECRET, prompt_tokens=7),
            ModelResponse(content="another turn"),
        ]), executor(), logger=logger,
    )
    result = loop.run([Message(role="user", content="Please remember " + SECRET)])
    records = read_records(path)
    assert result.completed and result.metrics is not None
    assert len(records) == 13  # start + 3 model calls + 2 tools + empty + finish
    assert [r["event"] for r in records].count("ToolExecutionFinished") == 2
    assert all(r["schema_version"] == 1 for r in records)
    assert all(r["run_id"] == result.run_id for r in records)
    assert all(datetime.fromisoformat(r["timestamp"]).utcoffset().total_seconds() == 0
               for r in records)
    assert SECRET not in path.read_text(encoding="utf-8")
    assert records[0]["step_id"] is None and records[-1]["step_id"] is None
    assert all(r["step_id"] == f"{result.run_id}:{r['data']['step']}"
               for r in records[1:-1])
    summary = records[-1]
    assert summary["data"]["stop_reason"] == "completed"
    assert summary["duration_ms"] >= 0
    assert summary["data"]["metrics"] == {
        "model_calls": 3, "preparation_errors": 0, "model_errors": 0,
        "tool_calls": 2, "tool_errors": 1,
        "prompt_tokens": 17, "completion_tokens": 3,
        "model_duration_ms": result.metrics.model_duration_ms,
        "tool_duration_ms": result.metrics.tool_duration_ms,
    }
    print("=== JSONL trace (content excluded) ===")
    for record in records:
        print(f"{record['event']:<24} step={record['step_id']} data={record['data']}")
    print("=== human-readable summary ===")
    print(terminal.getvalue().split("=== Agent run finished ===")[-1].strip())

    other = loop.run([Message(role="user", content="New invocation")])
    appended = read_records(path)
    assert other.run_id != result.run_id
    assert len(appended) == len(records) + 4  # started, model call, response, finished
    assert all(r["run_id"] == other.run_id for r in appended[len(records):])
    assert other.metrics.prompt_tokens is None
    print(f"Appended a second run with ID {other.run_id}")


def errors_and_limits(directory: Path) -> None:
    path = directory / "failures.jsonl"
    logger = JsonlTraceLogger(path)
    agent = AgentLoop(
        ScriptedLLM([RuntimeError("provider secret " + SECRET)]),
        executor(), logger=logger,
    )
    try:
        agent.run([Message(role="user", content=SECRET)])
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected provider failure")
    records = read_records(path)
    assert [r["event"] for r in records] == [
        "AgentStarted", "ModelCallStarted", "ModelCallFailed", "AgentFinished"
    ]
    assert records[-1]["data"]["metrics"]["model_errors"] == 1
    assert records[-1]["data"]["stop_reason"] == "error"
    assert records[2]["duration_ms"] >= 0
    assert SECRET not in path.read_text(encoding="utf-8")

    limit = AgentLoop(
        ScriptedLLM([ModelResponse(thinking="still going")]),
        executor(), logger=logger, max_steps=1,
    ).run([Message(role="user", content="keep going")])
    records = read_records(path)
    assert records[-1]["run_id"] == limit.run_id
    assert records[-1]["data"]["stop_reason"] == "max_steps"
    assert records[-1]["data"]["metrics"]["prompt_tokens"] is None
    print("Failure and max_steps summaries: error / max_steps")

    class FailingExecutor:
        def available_tools(self):
            return []

        def execute(self, call):
            raise RuntimeError("executor secret " + SECRET)

    tool_path = directory / "tool_failure.jsonl"
    failing_tool = AgentLoop(
        ScriptedLLM([ModelResponse(tool_calls=[ToolCall("echo", {})])]),
        FailingExecutor(), logger=JsonlTraceLogger(tool_path),
    )
    try:
        failing_tool.run([Message(role="user", content="tool failure")])
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected executor failure")
    tool_records = read_records(tool_path)
    assert tool_records[-2]["event"] == "ToolExecutionFailed"
    assert tool_records[-1]["data"]["metrics"]["tool_errors"] == 1
    assert SECRET not in tool_path.read_text()

    class PreparationExecutor:
        def available_tools(self):
            raise RuntimeError("discovery failed " + SECRET)

    preparation_path = directory / "preparation.jsonl"
    preparing = AgentLoop(
        ScriptedLLM([]), PreparationExecutor(),
        logger=JsonlTraceLogger(preparation_path),
    )
    try:
        preparing.run([Message(role="user", content="tool discovery")])
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected preparation failure")
    prep_records = read_records(preparation_path)
    assert [r["event"] for r in prep_records] == [
        "AgentStarted", "StepPreparationFailed", "AgentFinished"
    ]
    assert prep_records[-1]["data"]["metrics"]["preparation_errors"] == 1
    assert prep_records[-1]["data"]["metrics"]["model_calls"] == 0
    assert prep_records[1]["step_id"] == f"{prep_records[0]['run_id']}:1"
    assert SECRET not in preparation_path.read_text()
    print("Executor and pre-model preparation failures have complete run endings")


def opt_in_and_failure_semantics(directory: Path) -> None:
    path = directory / "full.jsonl"
    result = AgentLoop(
        ScriptedLLM([
            ModelResponse(tool_calls=[ToolCall("echo", {"value": SECRET})],
                          thinking="thinking " + SECRET),
            ModelResponse(content="final " + SECRET),
        ]), executor(), logger=JsonlTraceLogger(path, include_content=True),
    ).run([Message(role="user", content=SECRET)])
    records = read_records(path)
    assert records[0]["data"]["initial_messages"][0]["content"] == SECRET
    assert records[2]["data"]["thinking"] == "thinking " + SECRET
    assert records[4]["data"]["value"] == SECRET
    assert records[-1]["data"]["final_content"] == "final " + SECRET
    assert result.run_id == records[-1]["run_id"]
    print("Explicit full-content trace includes prompt, thinking, arguments and result")

    # Serialization finishes before opening the file: unsupported payloads
    # never become partial JSONL records.
    class ObjectLLM(LLM):
        def chat(self, messages, tools=None):
            return ModelResponse(tool_calls=[ToolCall("echo", {"value": object()})])

    invalid = directory / "unsupported.jsonl"
    try:
        AgentLoop(ObjectLLM(), executor(),
                  logger=JsonlTraceLogger(invalid, include_content=True)).run(
            [Message(role="user", content="invalid")]
        )
    except TypeError as exc:
        assert "not JSON-compatible" in str(exc)
    else:
        raise AssertionError("Expected unsupported payload to fail")
    assert all(json.loads(line) for line in invalid.read_text().splitlines())

    missing_parent = directory / "missing" / "trace.jsonl"
    try:
        AgentLoop(ScriptedLLM([ModelResponse(content="ok")]), executor(),
                  logger=JsonlTraceLogger(missing_parent)).run(
            [Message(role="user", content="try logging")]
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Trace I/O errors must propagate")
    print("Serialization and file I/O failures propagated (no silent loss)")


if __name__ == "__main__":
    with TemporaryDirectory(dir="/tmp/opencode") as tmp:
        directory = Path(tmp)
        completed_and_appended(directory / "runs.jsonl")
        errors_and_limits(directory)
        opt_in_and_failure_semantics(directory)
    print("All JSONL trace experiments passed.")
