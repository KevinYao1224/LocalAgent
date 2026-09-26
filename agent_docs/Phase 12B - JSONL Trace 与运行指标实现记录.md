# Phase 12B：JSONL Trace 与运行指标（Phase 12 完成）

## 结构与边界

Phase 5.5 已建立结构化事件和人类可读日志，Phase 12A 增加 run/step ID、UTC 时间
与单调时钟计时。12B 在同一条 `AgentLoop → AgentEvent → AgentLogger` 链上补充：

1. `observability/metrics.py`：不可变 `RunMetrics` 快照。`model_calls` 是进入
   `llm.chat` 的次数；`tool_calls` 是进入 executor 的次数；准备阶段失败单独计
   `preparation_errors`。模型异常计 `model_errors`，executor 抛异常或其返回
   `ToolResult(success=False)` 均计 `tool_errors`。模型/工具耗时为各次调用耗时之和，
   整轮 wall-clock 耗时仍取 `AgentFinished.duration_ms`。prompt / completion tokens
   分别累加**模型实际报告的数值**；完全未报告为 `null`，部分未报告时是已知值
   的部分合计，不能误当作完整使用量。
2. `AgentFinished.metrics` 将最终快照附在 `completed`、`max_steps` 和异常终止
   事件上；成功返回时 `AgentRunResult.metrics` 也可直接读取。同一次 run 的
   step-preparation（工具发现或上下文构造）异常新增 `StepPreparationFailed`，
   保留 step ID，并有 `AgentFinished(stop_reason="error")`。模型和执行器异常
   的失败事件已有对应耗时与结束事件。业务工具失败仍是
   `ToolExecutionFinished(success=False)`，允许 Agent 继续恢复。
3. `observability/jsonl.py`：`JsonlTraceLogger` 把每个事件写成一行 UTF-8 JSON。
   不改变 Agent 的消息历史或运行决策。`CompositeLogger` 让它与
   `HumanReadableLogger` 同时观察同一事件；默认 `NullLogger` 仍不输出。

## JSONL v1 格式

每行是一个完整 JSON 对象：

```json
{"schema_version":1,"event":"AgentFinished","run_id":"...","step_id":null,"timestamp":"2026-09-24T00:00:00Z","duration_ms":12.3,"data":{"steps":2,"stop_reason":"completed","metrics":{"model_calls":2,"preparation_errors":0,"model_errors":0,"tool_calls":1,"tool_errors":0,"prompt_tokens":42,"completion_tokens":8,"model_duration_ms":10.0,"tool_duration_ms":0.5}}}
```

`event` 是事件类名，`step_id` 在 run 级事件上为 `null`；`timestamp` 为 UTC
ISO-8601 字符串，`duration_ms` 在未计时事件上为 `null`。`data` 随事件类型
变化：模型响应含工具名称、是否有 content/thinking、token 和 done_reason；工具
结束含成功状态/错误类型；结束事件含 steps、stop_reason 和运行汇总。多个 run
追加到同一文件，通过 `run_id` 过滤。schema_version 明确为 1，以后改变字段语义
需要升级版本。此 JSONL 是事件记录，不是恢复 AgentState 的 checkpoint。

默认 `include_content=False`：**不持久化** prompt、回答、thinking、工具参数/结果
值和异常文本；工具名称、消息角色、响应是否为空、错误类型和计数仍会写入，所以
工具名等元数据也可能具有业务含义。显式 `include_content=True` 才会记录原始
消息（含历史 thinking）、模型 content/thinking/tool calls、工具参数/结果、错误
文本和最终回答。JSON 不支持的 payload 在此模式下抛 `TypeError`，NaN/Infinity
拒绝为标准 JSON；不会把任意对象悄悄转成字符串。

写入顺序与失败语义：每条事件**先序列化，后追加**，每个 logger 实例内的写入
有锁，打开/关闭文件确保通常可逐次读取；父目录必须预先存在。序列化或 I/O
失败会传播给调用者，不会悄悄丢失 trace，也不保证整个 run 事务原子性；故障时
可能只有前面的有效记录，操作系统部分写入失败也可能留下不完整行。不同进程
同时写同一路径没有排序/互斥保证。若需要非阻塞或强持久化，应单独设计 sink
策略。`CompositeLogger` 按传入顺序调用 logger，前一个失败时后续不执行。

## 如何运行

无 Ollama 的人类可读实验会打印事件表与最终指标、重新解析 JSONL 并 assert：

```bash
.venv/bin/python experiments/jsonl_trace_experiment.py
.venv/bin/python experiments/structured_trace_timing_experiment.py
```

实验覆盖正常与工具业务失败、thinking-only、部分缺失 token、同文件追加两个 run、
模型/执行器/准备失败、max_steps、默认内容排除、显式内容记录、非法 payload 和
缺失目录的失败传播。临时 JSONL 在 `/tmp/opencode` 下，实验退出自动清理。

当前 `main.py` 入口可与 Ollama 一同运行（目录先存在）：

```bash
.venv/bin/python main.py --prompt '计算 12 + 7' --trace-jsonl /tmp/opencode/agent-trace.jsonl
# 如确需保留原始内容，再显式附加 --trace-content
.venv/bin/python -c 'import json; from pathlib import Path; p=Path("/tmp/opencode/agent-trace.jsonl"); print(*[json.loads(line) for line in p.read_text().splitlines() if json.loads(line)["event"] == "AgentFinished"], sep="\n")'
```

下一条主线是 Phase 13 Evaluation：使用上述 step/run 指标构建可汇总 cases，
把成功率与成本、异常一起比较；Phase 9—11 仍未因 Phase 12 完成而自动完成。
