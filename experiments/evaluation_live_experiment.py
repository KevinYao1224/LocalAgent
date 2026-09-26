"""Phase 13 在线重复评测；必须有 Ollama 服务与已拉取的模型。"""

import argparse
from pathlib import Path
from statistics import mean, pstdev
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import EvaluationReport, evaluate
from evaluation.cases import online_cases
from llm.ollama import OllamaClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 13 Ollama repeated evaluation")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.timeout <= 0:
        parser.error("--repeats and --timeout must be positive")

    cases = online_cases()
    rates: list[float] = []
    results = []
    print(f"Ollama model={args.model} url={args.base_url} repeats={args.repeats} cases={len(cases)}")
    with OllamaClient(args.model, args.base_url, args.timeout) as llm:
        for number in range(1, args.repeats + 1):
            report = EvaluationReport(tuple(evaluate(case, llm) for case in cases))
            rates.append(report.passed / len(cases))
            results.extend(report.results)
            print(f"=== Repeat {number} ===")
            print(report.render())
    summary = EvaluationReport(tuple(results))
    print(f"=== Aggregate (real model, {args.repeats} repeats) ===")
    print(summary.render().splitlines()[-1])
    durations = [result.duration_ms for result in results if result.duration_ms is not None]
    latency = (
        f"mean={mean(durations):.1f}ms stdev={pstdev(durations):.1f}ms"
        if len(durations) == len(results) else "unknown"
    )
    print(f"per-repeat success mean={mean(rates):.1%} stdev={pstdev(rates):.1%}; "
          f"per-run latency {latency}")
    print("Small-sample observation only; run IDs above link to the agent runs. No trace is persisted.")
    return 0 if summary.passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
