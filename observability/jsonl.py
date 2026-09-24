"""Versioned, append-only structured trace for local experiments."""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any

from observability.events import (
    AgentEvent,
    AgentFinished,
    AgentStarted,
    EmptyModelResponse,
    ModelCallFailed,
    ModelCallStarted,
    ModelResponseReceived,
    StepPreparationFailed,
    ToolExecutionFailed,
    ToolExecutionFinished,
    ToolExecutionStarted,
)


SCHEMA_VERSION = 1


def _json_value(value: Any) -> Any:
    """Convert opted-in payloads without silently stringifying unknown objects."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json_value(item) for key, item in value.items()}
    raise TypeError(f"Trace payload is not JSON-compatible: {type(value).__name__}")


def event_record(event: AgentEvent, *, include_content: bool = False) -> dict[str, Any]:
    """Project one event to a stable JSON record; raw content is opt-in."""

    if not event.run_id or event.timestamp is None:
        raise ValueError("Trace events need a run_id and timestamp.")
    if not isinstance(event.timestamp, datetime) or event.timestamp.utcoffset() is None:
        raise ValueError("Trace timestamps must be timezone-aware datetimes.")

    data: dict[str, Any]
    if isinstance(event, AgentStarted):
        data = {
            "max_steps": event.max_steps,
            "initial_message_roles": [m.role for m in event.initial_messages],
        }
        if include_content:
            data["initial_messages"] = [
                {
                    "role": m.role,
                    "content": m.content,
                    "tool_name": m.tool_name,
                    "tool_calls": [
                        {"name": c.name, "arguments": _json_value(c.arguments)}
                        for c in m.tool_calls
                    ],
                    "thinking": m.thinking,
                }
                for m in event.initial_messages
            ]
    elif isinstance(event, ModelCallStarted):
        data = {
            "step": event.step,
            "message_count": event.message_count,
            "tool_names": list(event.tool_names),
        }
    elif isinstance(event, StepPreparationFailed):
        data = {"step": event.step, "error_type": event.error_type}
        if include_content:
            data["error"] = event.error
    elif isinstance(event, ModelResponseReceived):
        response = event.response
        data = {
            "step": event.step,
            "tool_names": [c.name for c in response.tool_calls],
            "has_content": bool(response.content.strip()),
            "has_thinking": bool(response.thinking),
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "done_reason": response.done_reason,
        }
        if include_content:
            data.update({
                "content": response.content,
                "thinking": response.thinking,
                "tool_calls": [
                    {"name": c.name, "arguments": _json_value(c.arguments)}
                    for c in response.tool_calls
                ],
            })
    elif isinstance(event, ModelCallFailed):
        data = {"step": event.step, "error_type": event.error_type}
        if include_content:
            data["error"] = event.error
    elif isinstance(event, ToolExecutionStarted):
        data = {"step": event.step, "tool_name": event.call.name}
        if include_content:
            data["arguments"] = _json_value(event.call.arguments)
    elif isinstance(event, ToolExecutionFinished):
        result = event.result
        data = {
            "step": event.step,
            "tool_name": event.call.name,
            "success": result.success,
            "error_type": result.error_type.value if result.error_type else None,
        }
        if include_content:
            data["arguments"] = _json_value(event.call.arguments)
            data["value"] = _json_value(result.value)
            data["error"] = result.error
    elif isinstance(event, ToolExecutionFailed):
        data = {
            "step": event.step,
            "tool_name": event.call.name,
            "error_type": event.error_type,
        }
        if include_content:
            data["arguments"] = _json_value(event.call.arguments)
            data["error"] = event.error
    elif isinstance(event, EmptyModelResponse):
        data = {"step": event.step}
    elif isinstance(event, AgentFinished):
        data = {
            "steps": event.steps,
            "stop_reason": event.stop_reason,
            "metrics": asdict(event.metrics) if event.metrics is not None else None,
        }
        if include_content:
            data["final_content"] = event.final_content
    else:
        raise TypeError(f"Unsupported trace event: {type(event).__name__}")

    return {
        "schema_version": SCHEMA_VERSION,
        "event": type(event).__name__,
        "run_id": event.run_id,
        "step_id": event.step_id,
        "timestamp": event.timestamp.astimezone(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "duration_ms": event.duration_ms,
        "data": data,
    }


class JsonlTraceLogger:
    """Append one UTF-8 JSON line per event; propagate serialization/I/O errors.

    Parent directories must exist. One lock protects writes from this logger
    instance; sharing a path between processes is outside this contract.
    """

    def __init__(self, path: str | Path, *, include_content: bool = False) -> None:
        self.path = Path(path)
        self.include_content = include_content
        self._lock = Lock()

    def log(self, event: AgentEvent) -> None:
        line = json.dumps(
            event_record(event, include_content=self.include_content),
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line)
