# Phase 10B — 自检提示与对照评测（Phase 10 收束）

## 方案与边界

在 Phase 10A 的 `RecoveryPolicy` 上增加 `self_critique=False`。显式传入 `RecoveryPolicy(max_corrections=1, self_critique=True)` 时，**仅**在可纠正工具错误且还剩纠正预算和 `max_steps` 时，为下一次模型请求追加一条固定的 user 级自检提示。提示让模型核对任务、上次工具错误和可用工具；它不替代模型决策，也不是第二个模型/多 Agent Critic。启用与关闭时调用次数上限相同，不增加额外的模型审查轮次。模型仍可能忽略提示或给出错误答案。

实现放在 `agent/loop.py` 的单次请求构造阶段：`ContextBuilder` 先从正式 trajectory 产生规范上下文，再加入暂态提示；**不**向 `AgentState`、`ConversationProjector` 或 `AgentSession` 写入伪造的外部输入。`RecoveryDecision.self_critique_next_step` 同时出现在可读日志与默认 JSONL 事件中，不包含原文；可用工具仍由 `ToolExecutor.available_tools()` 决定，模型及工具内容不能扩大权限。`runtime/recovery.py` 的固定提示是应用策略，不是来自不可信工具输出。默认行为及 Phase 10A 的不自动重试规则不变。

`evaluation.evaluate(..., recovery_policy=...)` 允许用既有逐例答案、工具名/参数、终止原因评分相同的纠正题目；不修改默认评测。单独的在线对照实验先用脚本注入同一个无效工具调用（为了可重复地**触发纠正路径**），从下一步开始交给同一 Ollama 模型；两组都启用同样的 `max_corrections=1`，只改变 `self_critique`。报告逐例 run ID、调用、错误、答案、步骤、耗时及两组真实模型调用数/已报告 token。首个注入响应不是模型生成的，评测总 token 成本为 `unknown`；真实模型 token 另行汇总，有缺失则保持 `unknown`。轮次交替实验顺序以缓解冷启动偏差，不保证消除负载影响。

## 人类可读实验（仓库根目录）

```bash
.venv/bin/python experiments/recovery_experiment.py
.venv/bin/python experiments/recovery_critic_live_experiment.py --help
.venv/bin/python experiments/recovery_critic_live_experiment.py --model qwen3.5:9b --repeats 3
```

第一个实验无需 Ollama：断言提示只在一次纠正请求中出现，原始输入、正式消息、Session 历史与下一模型请求均不含提示，工具集合未变；检查 `RecoveryDecision` 的 JSONL 投影、Phase 13 评分接口，以及模型和 executor 异常不重试。第二条仅显示参数；第三条需要已启动 Ollama 和模型。真实模型在线对照是**小样本条件纠错测试**，不是初始错误率测试，也不能仅凭几题确认自检提示有效。实验用例失败会作为事实报告，不以测试脚本非零退出码伪装成系统异常。

## 重要发现及剩余风险

- 本机 Ollama `qwen3.5:9b` 在线观测（2 题 × 3 轮 × 2 组）：普通纠正 6/6、19 总步骤、13 次真实模型调用、真实响应报告 prompt/completion 8864/625 token、总运行耗时约 13987ms；自检提示 6/6、18 总步骤、12 次真实模型调用、8490/713 token、总运行耗时约 11707ms。两组都执行了 12 次工具调用，其中每组 6 次预置错误。唯一次普通组额外步骤是空模型响应；重复间波动和冷启动可能比策略差异更重要。`EvaluationReport` 的总 token 为 `unknown`，因为每次第一步是无 token 数据的脚本响应；上面单列的仅是真实模型响应 token。此小样本没有准确率提升证据，也不说明真实场景的初始错误率。
- 不做模型/网络异常自动重试：请求可能已经消耗 token，工具异常可能已造成副作用；无法判断幂等性时按原有 `ModelCallFailed` / `ToolExecutionFailed` 记录并抛出，比假装安全恢复更诚实。Phase 11 权限与 timeout 前不扩大副作用工具能力。
- 自检提示可能改变模型措辞或让模型跳过真正需要的工具调用。必须用 Phase 13 的工具参数、答案和耗时/token 一起评估，而非只看是否完成；它不默认启用。
- 这是 bounded prompt-guided reflection，而非独立可信 Critic、语义正确性证明或无限反思。后续若需独立模型审查，应先设计独立预算、状态事实、统计有效的任务集与抗提示注入边界，不把其意见当执行权限。
