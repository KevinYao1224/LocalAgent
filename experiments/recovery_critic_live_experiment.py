"""Phase 10B：在相同的预置参数错误后，比较普通纠正和自检提示。

运行：.venv/bin/python experiments/recovery_critic_live_experiment.py --repeats 3
需要 Ollama。首个失败工具调用由脚本注入，不代表模型自己的初始出错率；
随后均由真实模型纠正。每个 arm 使用新的 Agent run，并保持相同工具与题目。
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import EvaluationCase, EvaluationReport, ExpectedCall, evaluate
from llm.base import LLM, Message, ModelResponse, ToolCall
from llm.ollama import OllamaClient
from runtime import RecoveryPolicy
from tools.base import Tool


class InjectFirstFailure(LLM):
    """每个 run 首次返回固定错误，其余请求交给 Ollama；单独计真实调用成本。"""

    def __init__(self, delegate: LLM, first: ToolCall) -> None:
        self._delegate = delegate
        self._first = first
        self.calls = 0
        self.real_responses: list[ModelResponse] = []

    @property
    def provider_name(self) -> str:
        return self._delegate.provider_name

    @property
    def model_name(self) -> str | None:
        return self._delegate.model_name

    @property
    def capabilities(self):
        return self._delegate.capabilities

    def chat(self, messages: list[Message], tools: list[Tool] | None = None) -> ModelResponse:
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(tool_calls=[self._first])
        response = self._delegate.chat(messages, tools)
        self.real_responses.append(response)
        return response


CASES = (
    (
        EvaluationCase(
            name="fix-add-arguments", prompt="Use the add tool to compute 12 + 7, then answer briefly.",
            expected_stop="completed", max_steps=4, expected_tool_errors=1,
            expected_calls=(ExpectedCall("add", {"a": 12}), ExpectedCall("add", {"a": 12, "b": 7})),
            answer_pattern=r"(?<!\d)19(?!\d)",
        ),
        ToolCall("add", {"a": 12}),
    ),
    (
        EvaluationCase(
            name="fix-unknown-tool", prompt="Use the multiply tool to compute 3 * 4, then answer briefly.",
            expected_stop="completed", max_steps=4, expected_tool_errors=1,
            expected_calls=(ExpectedCall("missing", {}), ExpectedCall("multiply", {"a": 3, "b": 4})),
            answer_pattern=r"(?<!\d)12(?!\d)",
        ),
        ToolCall("missing", {}),
    ),
)


def real_token_total(responses: list[ModelResponse], field: str) -> str:
    """有任何真实响应没报告 token，就不把部分和冒充完整成本。"""

    values = [getattr(response, field) for response in responses]
    return str(sum(values)) if values and all(value is not None for value in values) else "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 10B recovery self-critique A/B (Ollama)")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.timeout <= 0:
        parser.error("--repeats and --timeout must be positive")

    all_results = {False: [], True: []}
    real_responses = {False: [], True: []}
    print(f"model={args.model} repeats={args.repeats}; first invalid call is scripted")
    with OllamaClient(args.model, args.base_url, args.timeout) as llm:
        for repeat in range(1, args.repeats + 1):
            print(f"=== Repeat {repeat} ===")
            for case, first in CASES:
                # 交替顺序，减小仅由一方总在前面造成的冷启动偏差。
                for use_critic in ((False, True) if repeat % 2 else (True, False)):
                    probe = InjectFirstFailure(llm, first)
                    outcome = evaluate(
                        case, probe,
                        recovery_policy=RecoveryPolicy(1, self_critique=use_critic),
                    )
                    all_results[use_critic].append(outcome)
                    real_responses[use_critic].extend(probe.real_responses)
                    print(f"arm={'self-check' if use_critic else 'plain'} "
                          f"real_model_calls={len(probe.real_responses)}")
                    print(EvaluationReport((outcome,)).render())

    for use_critic in (False, True):
        report = EvaluationReport(tuple(all_results[use_critic]))
        responses = real_responses[use_critic]
        print(f"=== {'self-check' if use_critic else 'plain'} aggregate ===")
        print(report.render().splitlines()[-1])
        print(f"real_model_calls={len(responses)} "
              f"real_tokens(prompt/completion)="
              f"{real_token_total(responses, 'prompt_tokens')}/"
              f"{real_token_total(responses, 'completion_tokens')}")
    print("Small paired sample, not evidence of a general benefit. "
          "Synthetic first response has no model token cost; per-run IDs are shown above.")
    return 0  # 失败用例是实验结果，不等于脚本自身运行故障。


if __name__ == "__main__":
    raise SystemExit(main())
