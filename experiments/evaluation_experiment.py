"""Phase 13 离线评测：.venv/bin/python experiments/evaluation_experiment.py"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import EvaluationReport, evaluate
from evaluation.cases import ScriptedLLM, offline_cases
from evaluation.runner import EvaluationCase, ExpectedCall
from llm.base import ModelResponse, ToolCall


def main() -> None:
    cases = offline_cases()
    report = EvaluationReport(tuple(evaluate(item.case, ScriptedLLM(item.responses)) for item in cases))
    print("=== Scripted baseline (not real-model quality) ===")
    print(report.render())
    assert report.passed == len(cases)

    # 答案正确但工具参数错误、答案错误、运行异常均不可误报为成功。
    expected = EvaluationCase("intentional-failures", "Compute 12 + 7", "completed",
                              (ExpectedCall("add", {"a": 12, "b": 7}),), "19")
    wrong = evaluate(expected, ScriptedLLM((
        ModelResponse(tool_calls=[ToolCall("add", {"a": 12, "b": 8})]),
        ModelResponse(content="20"),
    )))
    crashed = evaluate(expected, ScriptedLLM(()))
    confused = evaluate(cases[-1].case, ScriptedLLM((ModelResponse(content="2024"),)))
    assert not wrong.passed and any("tool calls" in x for x in wrong.failures)
    assert not crashed.passed and crashed.stop_reason == "error"
    assert not confused.passed and confused.retrieved == cases[-1].case.expected_retrieved
    partial = evaluate(expected, ScriptedLLM((
        ModelResponse(tool_calls=[ToolCall("add", {"a": 12, "b": 7})], prompt_tokens=10),
        ModelResponse(content="19", prompt_tokens=None, completion_tokens=3),
    )))
    assert partial.passed and partial.prompt_tokens is None
    assert partial.completion_tokens is None  # 第一轮未报告 completion
    assert "unknown/unknown" in EvaluationReport((partial,)).render()
    print("=== Deliberate failures (runner self-test) ===")
    print(EvaluationReport((wrong, crashed, confused)).render())


if __name__ == "__main__":
    main()
