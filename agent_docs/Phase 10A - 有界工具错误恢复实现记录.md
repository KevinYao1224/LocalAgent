# Phase 10A — 有界工具错误恢复

## 范围与分层

`runtime/recovery.py` 按 `ToolResult.error_type` 分类失败，`AgentLoop` 决定是否给模型**下一次纠正机会**；它不会重放失败的工具调用，也不会写入额外的提示词或把工具输出提升为权限。`RecoveryPolicy(max_corrections=1)` 必须显式传给 `AgentLoop` 才启用；默认 `None` 保持旧行为（工具失败作为 observation 继续，最多运行 `max_steps`）。这是 Phase 10 的**第一小步**，并非模型 Critic 或通用异常重试。

| 失败类型 | 分类 | 启用策略后的行为 |
|---|---|---|
| `validation_error`、`tool_not_found` | 可纠正 | 将现有错误 observation 留在规范对话中，若仍有预算和步骤，让模型重新决定；不直接重试旧参数 |
| `tool_not_available` | 终止 | 不让模型借恢复策略绕过当前 capability |
| `execution_error` | 终止 | handler 可能已经产生副作用，不能安全重放 |

预算按**含失败的模型响应轮数**而非失败工具数量计算；同一响应中的工具仍按原顺序全部执行并记录结果，**不会中途回滚或取消同批后续调用**。如需避免同批副作用，应在权限策略和工具设计上控制，Phase 10A 不提供 sandbox。发生多种错误时终止类优先；首次失败之后有一次纠正机会意味着 `max_corrections=1` 可再向模型询问一次，若第二次失败则停止。`max_steps` 仍是总模型调用数的硬上限：若失败预算未耗尽但已达步骤上限则报告 `max_steps`，不虚报继续；若预算已经用完则优先报告 `recovery_exhausted`。

新增 `StopReason.RECOVERY_EXHAUSTED` 和 `UNRECOVERABLE_TOOL_ERROR` 均表示**未完成**，`AgentSession` 不提交此轮到短期记忆；已经运行的工具不会回滚。模型若在纠正后给出文本回答，原有 `completed` 语义仍只表示完成回答，**不保证答案或工具使用正确**。准备上下文、模型调用或 executor 抛出异常时仍按旧路径记录失败并抛出，不吞掉异常或对不确定操作重试。

`RecoveryDecision` 事件带 run/step ID、错误分类、决定和已用纠正轮数；人类可读日志显示决策，JSONL v1 只新增事件类型，不在默认 trace 中写入工具参数或错误原文；`AgentFinished` 保留已有 metrics 与实际停止原因。已有 trace 消费方应容忍新增事件和新的 stop reason。

## 离线可读实验

从仓库根目录运行（无需 Ollama）：

```bash
.venv/bin/python experiments/recovery_experiment.py
.venv/bin/python experiments/agent_loop_experiment.py
.venv/bin/python experiments/evaluation_experiment.py
.venv/bin/python experiments/planner_experiment.py
.venv/bin/python experiments/jsonl_trace_experiment.py
```

第一个实验会输出每次模型调用、工具结果、恢复决定及 run/step ID；它断言修正、未知工具、零预算、预算耗尽、权限拒绝、handler 执行失败、同批调用、`max_steps`、默认行为以及 Session 不提交未完成轮次。`handler` 示例会先记录一次可见的“side effect”再报错，以证明没有重试；`event_record()` 验证新增 JSONL 事件在默认配置下不包含参数。脚本模型只能证明流程正确，不能证明真实模型能修正错误。其余命令检查旧控制流、评测、规划和 trace 兼容性。

## 下一步 / 重要发现

- 可恢复的是**重新询问模型修正调用**，不是安全重试 handler；即便 handler 报错，它也可能已做了部分工作。非幂等工具引入前仍必须完成 Phase 11 权限、timeout、工作目录和输出边界。
- 同批调用的错误在整批结束后决策，能保持完整工具 observation；这也意味着失败不会阻止同批后续工具的副作用，扩展到副作用工具前须重新审视批处理策略。
- 此处没有基于模型的 Critic、退避/网络重试，也未验证恢复对答案的收益。若要评估收益，用 Phase 13 的同题在线多次评测对比启用与关闭策略，并核对实际工具调用、token 和延迟，不以脚本通过率冒充模型质量。
