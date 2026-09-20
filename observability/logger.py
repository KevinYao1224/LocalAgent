import json
import sys

from typing import Protocol, TextIO

from observability.events import (
    AgentEvent,
    AgentFinished,
    AgentStarted,
    EmptyModelResponse,
    ModelCallFailed,
    ModelCallStarted,
    ModelResponseReceived,
    ToolExecutionFailed,
    ToolExecutionFinished,
    ToolExecutionStarted,
)


class AgentLogger(Protocol):
    """Receive structured events emitted by AgentLoop."""

    def log(self, event: AgentEvent) -> None:
        pass


class NullLogger:
    """Discard events when logging is not requested."""

    def log(self, event: AgentEvent) -> None:
        pass


class HumanReadableLogger:
    """Write an AgentLoop trace intended for local terminal experiments."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdout

    def log(self, event: AgentEvent) -> None:
        if isinstance(event, AgentStarted):
            self._log_agent_started(event)
        elif isinstance(event, ModelCallStarted):
            self._log_model_call_started(event)
        elif isinstance(event, ModelResponseReceived):
            self._log_model_response(event)
        elif isinstance(event, ModelCallFailed):
            self._write(
                f"Model call failed: {event.error_type}: {event.error}"
            )
        elif isinstance(event, ToolExecutionStarted):
            self._log_tool_started(event)
        elif isinstance(event, ToolExecutionFinished):
            self._log_tool_finished(event)
        elif isinstance(event, ToolExecutionFailed):
            self._write(
                f"Tool execution failed: {event.call.name}: "
                f"{event.error_type}: {event.error}"
            )
        elif isinstance(event, EmptyModelResponse):
            self._write(
                "Decision: empty model response; continue to the next step."
            )
        elif isinstance(event, AgentFinished):
            self._log_agent_finished(event)
        else:
            raise TypeError(f"Unsupported agent event: {type(event).__name__}")

        self._stream.flush()

    def _log_agent_started(self, event: AgentStarted) -> None:
        self._write("=== Agent run started ===")
        self._write(f"Max steps: {event.max_steps}")
        self._write(f"Initial messages: {len(event.initial_messages)}")

        for message in event.initial_messages:
            self._write(f"  [{message.role}] {message.content}")

    def _log_model_call_started(self, event: ModelCallStarted) -> None:
        self._write(f"\n--- Step {event.step}: model call ---")
        self._write(f"History messages: {event.message_count}")
        tools = ", ".join(event.tool_names) if event.tool_names else "<none>"
        self._write(f"Available tools: {tools}")

    def _log_model_response(self, event: ModelResponseReceived) -> None:
        response = event.response
        self._write_text("Model thinking", response.thinking)
        self._write_text("Model content", response.content)

        if response.tool_calls:
            self._write(f"Tool calls: {len(response.tool_calls)}")
            for index, call in enumerate(response.tool_calls, start=1):
                arguments = json.dumps(
                    call.arguments,
                    ensure_ascii=False,
                    default=str,
                )
                self._write(
                    f"  {index}. {call.name}({arguments})"
                )
        else:
            self._write("Tool calls: 0")

        metrics = []
        if response.prompt_tokens is not None:
            metrics.append(f"prompt={response.prompt_tokens}")
        if response.completion_tokens is not None:
            metrics.append(f"completion={response.completion_tokens}")
        if response.done_reason is not None:
            metrics.append(f"done_reason={response.done_reason}")
        if metrics:
            self._write(f"Model metadata: {', '.join(metrics)}")

    def _log_tool_started(self, event: ToolExecutionStarted) -> None:
        arguments = json.dumps(
            event.call.arguments,
            ensure_ascii=False,
            default=str,
        )
        self._write(
            f"Tool start: {event.call.name}({arguments})"
        )

    def _log_tool_finished(self, event: ToolExecutionFinished) -> None:
        result = event.result

        if result.success:
            self._write(
                f"Tool result: {result.tool_name} -> "
                f"{result.to_message_content()}"
            )
            return

        assert result.error_type is not None
        self._write(
            f"Tool error: {result.tool_name} "
            f"[{result.error_type.value}] -> {result.error}"
        )

    def _log_agent_finished(self, event: AgentFinished) -> None:
        self._write("\n=== Agent run finished ===")
        self._write(f"Stop reason: {event.stop_reason}")
        self._write(f"Steps: {event.steps}")
        final_content = event.final_content.strip()
        self._write(
            f"Final content: {final_content if final_content else '<empty>'}"
        )

    def _write(self, text: str) -> None:
        self._stream.write(f"{text}\n")

    def _write_text(self, label: str, text: str) -> None:
        value = text.strip()

        if not value:
            self._write(f"{label}: <empty>")
            return

        if "\n" not in value:
            self._write(f"{label}: {value}")
            return

        self._write(f"{label}:")
        for line in value.splitlines():
            self._write(f"  {line}")
