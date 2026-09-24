# Phase 12A：Run / Step 标识与计时

## 本阶段的边界

沿用 Phase 5.5 的事件和 Logger，不改变模型决策或工具执行。每次 `AgentLoop.run()`
创建一个新的 `AgentState.run_id`（UUID4 十六进制）；模型尝试的 `step_id` 定义为
`<run_id>:<step number>`。同一个 `AgentLoop` 连续运行也不会复用 run ID。
`AgentRunResult.run_id` 只是 `state.run_id` 的便捷入口；成功返回的 `AgentStep`
也保存对应 step ID。模型在某个 step 抛错时，不会伪造 `AgentStep`，但该尝试的
`ModelCallFailed.step_id` 仍可通过 `state.step_id(number)` 找到。

`observability/events.py` 的 `EventMetadata` 为每种既有事件增加可选的
`run_id`、`step_id`、`timestamp`、`duration_ms`；可选默认值兼容手工构造旧事件。
由 AgentLoop 发出的事件始终具有 run ID 和 UTC aware 时间戳。step 事件同时具有
step ID；run 开始/结束事件没有 step ID。结束事件覆盖 `completed`、`max_steps`
和模型/执行器抛异常路径；Phase 12B 后也覆盖工具发现或上下文构造异常。
`ModelResponse` 的 prompt / completion tokens 仍是
provider 返回的原始计数；缺失值不补零，工具返回业务错误仍是
`ToolExecutionFinished`（其中 `result.success=False`），而不是执行器异常事件。

时间语义：

| 字段 | 含义 |
|---|---|
| `AgentState.started_at` | 创建本轮 state 时的 UTC 墙上时间 |
| `event.timestamp` | 事件发出时的 UTC 墙上时间；开始事件沿用 started_at |
| `ModelResponseReceived` / `ModelCallFailed.duration_ms` | 单次 `llm.chat()` 调用耗时 |
| `ToolExecutionFinished` / `ToolExecutionFailed.duration_ms` | 单次 `executor.execute()` 耗时，包括校验和能力门控 |
| `AgentFinished.duration_ms` | 从开始记录 run 至结束事件的总耗时，包含上下文构造、logger 和调用间开销 |

耗时使用 `perf_counter()` 差值，不由 UTC 时间戳相减；开始、空响应等事件没有
`duration_ms`。多次工具调用共享其模型 step ID，暂不提供单独的 tool-call ID；
它们仍可按事件顺序和 tool call 对应。HumanReadableLogger 显示 run / step ID、
起始时间和各段耗时，并继续保留原来的模型 token 字段。

## 人类可读离线实验

```bash
.venv/bin/python experiments/structured_trace_timing_experiment.py
```

脚本用带 10ms 延迟的模拟模型与工具打印关联 ID、耗时和 token metadata；验证同一
AgentLoop 两次运行 ID 不同、响应 step ID 与事件一致、UTC 时间、模型失败与执行器
异常的关联、`max_steps` 结束原因。无需 Ollama。可对照终端中的模型、工具和整轮
耗时：整轮耗时不要求严格等于各段相加。运行旧的日志实验可检查人类可读格式兼容：

```bash
.venv/bin/python experiments/logging_experiment.py
```

## 后续

Phase 12B 已加入 JSONL v1 和运行指标，详见
[Phase 12B JSONL Trace 与运行指标实现记录](./Phase%2012B%20-%20JSONL%20Trace%20与运行指标实现记录.md)。
之后再将现有实验组织成可汇总的 Evaluation cases；不要用计时小样本推断模型质量。
