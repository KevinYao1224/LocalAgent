from dataclasses import dataclass, replace

from llm.base import ModelResponse
from runtime.result import ToolResult


def _add_reported_tokens(total: int | None, reported: int | None) -> int | None:
    """Keep unknown usage unknown until a provider reports a value."""

    if reported is None:
        return total
    return (total or 0) + reported


@dataclass(frozen=True, slots=True)
class RunMetrics:
    """A snapshot of one run's measured calls, outcomes, and known token usage."""

    model_calls: int = 0
    preparation_errors: int = 0
    model_errors: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    model_duration_ms: float = 0.0
    tool_duration_ms: float = 0.0

    def model_returned(self, response: ModelResponse, duration_ms: float) -> "RunMetrics":
        return replace(
            self,
            prompt_tokens=_add_reported_tokens(
                self.prompt_tokens, response.prompt_tokens
            ),
            completion_tokens=_add_reported_tokens(
                self.completion_tokens, response.completion_tokens
            ),
            model_duration_ms=self.model_duration_ms + duration_ms,
        )

    def tool_finished(self, result: ToolResult, duration_ms: float) -> "RunMetrics":
        return replace(
            self,
            tool_errors=self.tool_errors + (not result.success),
            tool_duration_ms=self.tool_duration_ms + duration_ms,
        )
