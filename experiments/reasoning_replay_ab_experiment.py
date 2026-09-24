"""使用本地 Ollama 模型比较 ContextBuilder 的 reasoning replay 策略。

从项目根目录运行：

    python experiments/reasoning_replay_ab_experiment.py --runs 3

与确定性实验不同，本脚本需要运行中的 Ollama 服务和已配置的模型。由于本地模型输出
可能随运行变化，结果用于探索观察，不作为单元测试断言。
"""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import AgentLoop, ContextBuilder, ReasoningReplayPolicy
from llm.base import Message
from llm.ollama import OllamaClient
from runtime import ToolExecutor
from tools import ToolRegistry, calculator_tools


MODEL = "qwen3.5:9b"
MAX_STEPS = 6


class CountingContextBuilder(ContextBuilder):
    """统计实际 replay 次数，同时保留 ContextBuilder 原有行为。"""

    def __init__(self, policy: ReasoningReplayPolicy) -> None:
        super().__init__(reasoning_replay=policy)
        self.replay_count = 0

    def build(self, state, llm):
        result = super().build(state, llm)
        if result.reasoning_replayed:
            self.replay_count += 1
        return result


@dataclass(frozen=True, slots=True)
class RunMetrics:
    policy: ReasoningReplayPolicy
    completed: bool
    correct: bool
    steps: int
    empty_responses: int
    thinking_responses: int
    repeated_tool_calls: int
    prompt_tokens: int
    completion_tokens: int
    replay_count: int
    latency_seconds: float
    final_content: str


def create_executor() -> ToolExecutor:
    registry = ToolRegistry()
    for tool in calculator_tools:
        registry.register(tool)
    return ToolExecutor(registry)


def create_messages() -> list[Message]:
    return [
        Message(
            role="system",
            content=(
                "You are an assistant with arithmetic tools. "
                "Use exactly one arithmetic tool per step. "
                "Do not calculate intermediate values yourself. "
                "After receiving a tool result, decide whether another "
                "tool is needed before answering the user."
            ),
        ),
        Message(
            role="user",
            content=(
                "Calculate (12 + 7) multiplied by 5. "
                "Use a separate tool call for each arithmetic operation."
            ),
        ),
    ]


def tool_call_signature(call) -> str:
    arguments = json.dumps(
        call.arguments,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return f"{call.name}:{arguments}"


def run_once(
    llm: OllamaClient,
    policy: ReasoningReplayPolicy,
) -> RunMetrics:
    context_builder = CountingContextBuilder(policy)
    agent = AgentLoop(
        llm=llm,
        executor=create_executor(),
        max_steps=MAX_STEPS,
        context_builder=context_builder,
    )

    started = perf_counter()
    result = agent.run(create_messages())
    latency = perf_counter() - started

    responses = [step.model_response for step in result.agent_steps]
    signatures = [
        tool_call_signature(call)
        for response in responses
        for call in response.tool_calls
    ]
    counts = Counter(signatures)
    repeated_calls = sum(count - 1 for count in counts.values())
    final_content = result.response.content.strip()

    return RunMetrics(
        policy=policy,
        completed=result.completed,
        correct=result.completed and "95" in final_content,
        steps=result.steps,
        empty_responses=sum(
            not response.content.strip() and not response.tool_calls
            for response in responses
        ),
        thinking_responses=sum(bool(response.thinking) for response in responses),
        repeated_tool_calls=repeated_calls,
        prompt_tokens=sum(response.prompt_tokens or 0 for response in responses),
        completion_tokens=sum(
            response.completion_tokens or 0 for response in responses
        ),
        replay_count=context_builder.replay_count,
        latency_seconds=latency,
        final_content=final_content,
    )


def print_run(index: int, metrics: RunMetrics) -> None:
    print(
        f"{metrics.policy.value} run {index}: "
        f"correct={metrics.correct}, completed={metrics.completed}, "
        f"steps={metrics.steps}, empty={metrics.empty_responses}, "
        f"repeated_calls={metrics.repeated_tool_calls}, "
        f"replays={metrics.replay_count}, "
        f"prompt_tokens={metrics.prompt_tokens}, "
        f"completion_tokens={metrics.completion_tokens}, "
        f"latency={metrics.latency_seconds:.2f}s"
    )
    print(f"  final={metrics.final_content!r}")


def print_summary(policy: ReasoningReplayPolicy, runs: list[RunMetrics]) -> None:
    print(f"\n=== {policy.value} summary ({len(runs)} runs) ===")
    print(f"success_rate={sum(run.correct for run in runs) / len(runs):.2%}")
    print(f"avg_steps={statistics.mean(run.steps for run in runs):.2f}")
    print(
        "avg_empty_responses="
        f"{statistics.mean(run.empty_responses for run in runs):.2f}"
    )
    print(
        "avg_repeated_tool_calls="
        f"{statistics.mean(run.repeated_tool_calls for run in runs):.2f}"
    )
    print(
        "avg_prompt_tokens="
        f"{statistics.mean(run.prompt_tokens for run in runs):.2f}"
    )
    print(
        "avg_completion_tokens="
        f"{statistics.mean(run.completion_tokens for run in runs):.2f}"
    )
    print(
        "avg_latency_seconds="
        f"{statistics.mean(run.latency_seconds for run in runs):.2f}"
    )
    print(f"total_replays={sum(run.replay_count for run in runs)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs for each replay policy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1.")

    policies = (
        ReasoningReplayPolicy.OFF,
        ReasoningReplayPolicy.LATEST_PENDING,
    )
    results = {policy: [] for policy in policies}

    with OllamaClient(model=MODEL, timeout=180.0) as llm:
        # 交替运行不同策略，降低顺序和模型预热造成的偏差。
        for index in range(1, args.runs + 1):
            for policy in policies:
                metrics = run_once(llm, policy)
                results[policy].append(metrics)
                print_run(index, metrics)

    for policy in policies:
        print_summary(policy, results[policy])


if __name__ == "__main__":
    main()
