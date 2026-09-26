"""小规模公开题集；脚本响应只用于验证评分机制，不能代表模型能力。"""

from dataclasses import dataclass

from evaluation.runner import EvaluationCase, ExpectedCall
from llm.base import LLM, Message, ModelResponse, ToolCall
from tools.base import Tool


class ScriptedLLM(LLM):
    def __init__(self, responses: tuple[ModelResponse, ...]):
        self._responses = iter(responses)

    def chat(self, messages: list[Message], tools: list[Tool] | None = None) -> ModelResponse:
        return next(self._responses)


@dataclass(frozen=True)
class ScriptedCase:
    case: EvaluationCase
    responses: tuple[ModelResponse, ...]


def offline_cases() -> tuple[ScriptedCase, ...]:
    return (
        ScriptedCase(
            EvaluationCase("addition", "Use add to compute 12 + 7.", "completed",
                           (ExpectedCall("add", {"a": 12, "b": 7}),), "19"),
            (ModelResponse(tool_calls=[ToolCall("add", {"a": 12, "b": 7})]),
             ModelResponse(content="19")),
        ),
        ScriptedCase(
            EvaluationCase("tool-choice-order", "Use subtract to compute 9 - 4.", "completed",
                           (ExpectedCall("subtract", {"a": 9, "b": 4}),), "5"),
            (ModelResponse(tool_calls=[ToolCall("subtract", {"a": 9, "b": 4})]),
             ModelResponse(content="5")),
        ),
        ScriptedCase(
            EvaluationCase("unknown-tool", "Try the unavailable sqrt tool, then explain.",
                           "completed", (ExpectedCall("sqrt", {"a": 4}),),
                           "not available", expected_tool_errors=1),
            (ModelResponse(tool_calls=[ToolCall("sqrt", {"a": 4})]),
             ModelResponse(content="not available")),
        ),
        ScriptedCase(
            EvaluationCase("no-tool", "Reply exactly: hello", "completed", expected_answer="hello"),
            (ModelResponse(content="hello"),),
        ),
        ScriptedCase(
            EvaluationCase("step-limit", "Keep calculating.", "max_steps",
                           (ExpectedCall("add", {"a": 1, "b": 2}),), max_steps=1),
            (ModelResponse(tool_calls=[ToolCall("add", {"a": 1, "b": 2})]),),
        ),
        ScriptedCase(
            EvaluationCase("missing-memory", "What is Project Cedar's deadline? If absent say unknown.",
                           "completed", expected_answer="unknown", memory_query="deadline",
                           memory_records=("Project Maple deadline: 2024",),
                           expected_retrieved=("Project Maple deadline: 2024",)),
            (ModelResponse(content="unknown"),),
        ),
        ScriptedCase(
            EvaluationCase("memory-attribution", "What is Project Cedar's deadline?",
                           "completed", expected_answer="2025", memory_query="deadline",
                           memory_records=("Project Maple deadline: 2024", "Project Cedar deadline: 2025"),
                           expected_retrieved=("Project Cedar deadline: 2025", "Project Maple deadline: 2024")),
            (ModelResponse(content="2025"),),
        ),
    )


def online_cases() -> tuple[EvaluationCase, ...]:
    """固定输入与检查标准；每次重复均使用新 run，不复用会话状态。"""

    return (
        EvaluationCase("addition", "Use the add tool to compute 12 + 7, then answer briefly.",
                       "completed", (ExpectedCall("add", {"a": 12, "b": 7}),),
                       answer_pattern=r"(?<!\d)19(?!\d)"),
        EvaluationCase("tool-choice-order", "Use the subtract tool to compute 9 - 4, then answer briefly.",
                       "completed", (ExpectedCall("subtract", {"a": 9, "b": 4}),),
                       answer_pattern=r"(?<!\d)5(?!\d)"),
        EvaluationCase("no-tool", "Reply with only hello. Do not use tools.",
                       "completed", answer_pattern=r"^hello[.!]?$"),
        EvaluationCase("missing-memory", "What is Project Cedar's deadline? If absent say unknown.",
                       "completed", answer_pattern=r"\bunknown\b|\bnot (?:known|provided|available)\b",
                       memory_query="deadline", memory_records=("Project Maple deadline: 2024",),
                       expected_retrieved=("Project Maple deadline: 2024",)),
        EvaluationCase("memory-attribution", "What is Project Cedar's deadline? Answer with its year only.",
                       "completed", answer_pattern=r"^2025$", memory_query="deadline",
                       memory_records=("Project Maple deadline: 2024", "Project Cedar deadline: 2025"),
                       expected_retrieved=("Project Cedar deadline: 2025", "Project Maple deadline: 2024")),
    )
