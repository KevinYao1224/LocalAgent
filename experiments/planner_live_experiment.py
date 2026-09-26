"""Phase 9 可选在线 A/B：需要 Ollama；小样本不等于质量结论。"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import CalculatorPlanner
from evaluation import evaluate
from evaluation.runner import EvaluationCase, ExpectedCall
from llm.ollama import OllamaClient
from runtime import ToolExecutor
from tools import ToolRegistry, calculator_tools


TASK = "Compute (12 + 7) * 5. Use add then multiply, and answer with the number only."
EXPECTED = (
    ExpectedCall("add", {"a": 12, "b": 7}),
    ExpectedCall("multiply", {"a": 19, "b": 5}),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 9 calculator planning vs Phase 13 baseline")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.repeats < 1:
        parser.error("--timeout and --repeats must be positive")

    case = EvaluationCase("two-tools", TASK, "completed", EXPECTED, "95")
    baseline_passed = planned_passed = 0
    with OllamaClient(args.model, args.base_url, args.timeout) as llm:
        for number in range(1, args.repeats + 1):
            baseline = evaluate(case, llm)
            baseline_passed += baseline.passed
            print(f"[{number}] AgentLoop run={baseline.run_id} pass={baseline.passed} "
                  f"answer={baseline.answer!r} calls={baseline.calls} duration={baseline.duration_ms}ms")
            registry = ToolRegistry()
            for tool in calculator_tools:
                registry.register(tool)
            planner = CalculatorPlanner(llm, ToolExecutor(registry))
            try:
                result = planner.run(TASK)
            except Exception as exc:
                print(f"[{number}] Planner run={planner.last_state.run_id if planner.last_state else 'unknown'} "
                      f"exception={type(exc).__name__}: {exc}")
                continue
            calls = tuple(ExpectedCall(item.call.name, item.call.arguments)
                          for item in result.state.executions)
            passed = result.completed and result.answer == 95 and calls == EXPECTED
            planned_passed += passed
            print(f"[{number}] Planner run={result.state.run_id} pass={passed} "
                  f"answer={result.answer!r} calls={calls} duration={result.duration_ms:.1f}ms")
            if not passed:
                print(result.render())
    print(f"Observed exact-task success: AgentLoop={baseline_passed}/{args.repeats}; "
          f"Planner={planned_passed}/{args.repeats}")
    print("Different protocols and model-call counts; small sample, not a general quality claim. "
          "Plan traces are in memory only; no JSONL is persisted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
