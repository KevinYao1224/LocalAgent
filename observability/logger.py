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
    StepPreparationFailed,
    ToolExecutionFailed,
    ToolExecutionFinished,
    ToolExecutionStarted,
)


class AgentLogger(Protocol):
    """接收 AgentLoop 发出的结构化事件。"""

    def log(self, event: AgentEvent) -> None:
        pass


class NullLogger:
    """未请求日志记录时丢弃事件。"""

    def log(self, event: AgentEvent) -> None:
        pass


class CompositeLogger:
    """按顺序将事件转发给多个 logger，不隐藏其抛出的错误。"""

    def __init__(self, *loggers: AgentLogger) -> None:
        self._loggers = loggers

    def log(self, event: AgentEvent) -> None:
        for logger in self._loggers:
            logger.log(event)


class HumanReadableLogger:
    """将 AgentLoop 轨迹输出为适合本地终端实验阅读的日志。"""

    def __init__(
        self,
        stream: TextIO | None = None,
        max_text_chars: int | None = None,
    ) -> None:
        if max_text_chars is not None and (
            isinstance(max_text_chars, bool)
            or not isinstance(max_text_chars, int)
            or max_text_chars < 1
        ):
            raise ValueError(
                "max_text_chars must be a positive integer or None."
            )
        self._stream = stream if stream is not None else sys.stdout
        self._max_text_chars = max_text_chars

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
            self._write_duration(event.duration_ms)
        elif isinstance(event, StepPreparationFailed):
            self._write(
                f"Step preparation failed: {event.error_type}: {event.error}"
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
            self._write_duration(event.duration_ms)
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
        if event.run_id is not None:
            self._write(f"Run ID: {event.run_id}")
        if event.timestamp is not None:
            self._write(f"Started at: {event.timestamp.isoformat()}")
        self._write(f"Max steps: {event.max_steps}")
        self._write(f"Initial messages: {len(event.initial_messages)}")

        for message in event.initial_messages:
            self._write(
                f"  [{message.role}] {self._preview(message.content)}"
            )

    def _log_model_call_started(self, event: ModelCallStarted) -> None:
        self._write(f"\n--- Step {event.step}: model call ---")
        if event.step_id is not None:
            self._write(f"Step ID: {event.step_id}")
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
        self._write_duration(event.duration_ms)

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
                f"{self._preview(result.to_message_content())}"
            )
            self._write_duration(event.duration_ms)
            return

        assert result.error_type is not None
        self._write(
            f"Tool error: {result.tool_name} "
            f"[{result.error_type.value}] -> {result.error}"
        )
        self._write_duration(event.duration_ms)

    def _log_agent_finished(self, event: AgentFinished) -> None:
        self._write("\n=== Agent run finished ===")
        self._write(f"Stop reason: {event.stop_reason}")
        self._write(f"Steps: {event.steps}")
        self._write_duration(event.duration_ms)
        if event.metrics is not None:
            metrics = event.metrics
            self._write(
                "Run metrics: "
                f"model_calls={metrics.model_calls}, "
                f"preparation_errors={metrics.preparation_errors}, "
                f"model_errors={metrics.model_errors}, "
                f"tool_calls={metrics.tool_calls}, "
                f"tool_errors={metrics.tool_errors}, "
                f"prompt_tokens={metrics.prompt_tokens}, "
                f"completion_tokens={metrics.completion_tokens}, "
                f"model_ms={metrics.model_duration_ms:.2f}, "
                f"tool_ms={metrics.tool_duration_ms:.2f}"
            )
        final_content = event.final_content.strip()
        self._write(
            "Final content: "
            f"{self._preview(final_content) if final_content else '<empty>'}"
        )

    def _write_duration(self, duration_ms: float | None) -> None:
        if duration_ms is not None:
            self._write(f"Duration: {duration_ms:.2f} ms")

    def _write(self, text: str) -> None:
        self._stream.write(f"{text}\n")

    def _write_text(self, label: str, text: str) -> None:
        value = self._preview(text.strip())

        if not value:
            self._write(f"{label}: <empty>")
            return

        if "\n" not in value:
            self._write(f"{label}: {value}")
            return

        self._write(f"{label}:")
        for line in value.splitlines():
            self._write(f"  {line}")

    def _preview(self, text: str) -> str:
        """限制终端预览长度，但不修改底层事件内容。"""

        if self._max_text_chars is None or len(text) <= self._max_text_chars:
            return text

        omitted = len(text) - self._max_text_chars
        return f"{text[:self._max_text_chars]}... <{omitted} chars omitted>"
