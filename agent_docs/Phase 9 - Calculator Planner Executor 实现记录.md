# Phase 9 — Calculator Planner / Executor

## 作用与设计范围

增加一条**显式启用**的 plan-then-execute 路径，不改变默认 `AgentLoop`、`AgentSession` 和 `main.py`。模型仅提出 JSON 计算计划；Python Runtime 在任何工具执行前校验整份计划，再按顺序解析依赖、执行并核验每步结果。适合观察“先规划”和“边看工具结果边决定下一步”两种控制方式的差异。本阶段仅用已注册的四则运算工具；**不引入文件、命令、网络等副作用工具**，它们需先有 Phase 11 权限边界。

## 新增模块和约定

- `agent/planner.py` 的 `CalculatorPlanner.run(task)` 从当前 `ToolExecutor.available_tools()` 选计算器工具，向 `LLM.chat(..., tools=[])` 发送任务和工具描述，请求单个 JSON：`{"steps":[{"tool":"add","arguments":{"a":12,"b":7}},{"tool":"multiply","arguments":{"a":{"from_step":1},"b":5}}]}`。最后一步的数值作为结果，不调用模型合成最终答案。
- `parse_plan()` 拒绝额外字段、重复 JSON 键、未知/不可用工具、空计划或超出 `max_plan_steps` 的计划、非法/非有限数值、布尔值、非法依赖和不匹配的工具 schema。引用只支持先前步骤的整个数值，不能嵌套任意表达式。先完成**整份计划**的校验，再开始执行，避免在发现后面步骤无效时已经执行前面的步骤。
- `Plan` / `PlanStep` / `StepReference` 保存受约束的意图；`PlanState` / `PlanExecution` 保存 run ID、原计划、执行时实际参数、结构化 `ToolResult`、验证结果和计时；`PlanRunResult.render()` 可读地显示 verified / failed / pending。`PlanStopReason` 区分 completed、invalid_plan 和 tool_failed。
- 每步仍走 `ToolExecutor.execute()`，执行时重新检查 capability、JSON Schema 和工具 handler。只把成功且返回**有限数值**的结果传给后续步骤；错误立即停止，不重试也不继续使用失败结果。预期内工具失败返回 `tool_failed`，模型/Runtime 意外异常按现有 Runtime 约定抛出，此时可查看 `planner.last_state` 中已记录的事实。
- `agent/__init__.py` 导出该可选入口。现有 Agent 轨迹、会话历史、Phase 12 JSONL 和 Phase 13 runner **没有被冒充为** Planner 的持久化/恢复机制；Planner 独立的可读结果只在内存中。

## 运行与观察

无需 Ollama，从仓库根目录运行：

```bash
.venv/bin/python experiments/planner_experiment.py
```

实验先调用 Phase 13 `evaluate()` 检查普通 `AgentLoop` 的两工具脚本基线，再运行脚本 Planner 的同一算术题。看逐步 `verified` 与最终 `95`；错误案例覆盖未来依赖/后续步骤格式错误（执行数 0）、除零（停止且后续 `pending`）、非法 JSON/未知工具/重复键/NaN/布尔值/步数上限、规划前禁用工具和执行中撤销工具、返回无穷大以及模型擅自请求 tool call。脚本模型的成功率**不是**真实模型质量；故意提供 `12 + 8` 的合规计划也会得到错误任务答案 `20`，说明“验证”只是结构、能力和运行结果验证，**不能证明计划符合用户意图**。

可选真实模型小样本对照（需运行 Ollama 且模型可用）：

```bash
.venv/bin/python experiments/planner_live_experiment.py --model qwen3.5:9b --repeats 3
```

此脚本对同一题分别调用 Phase 13 普通 Agent 评测和 CalculatorPlanner，输出每次 run ID、答案、执行调用、耗时和严格匹配率。二者提示协议和模型调用次数不同；这些输出只是小样本观察，不能把成功率差异归因为 Planner 单一因素。在线实验不自动写 JSONL。

## 重要发现与后续方案

1. 校验计划和校验工具结果是两件事：即使计划中每个参数合法，除零或数值溢出仍可能在执行时失败；失败后的状态必须保留已执行步骤，而不能给出成功答案。
2. 规划时工具可用不代表执行时始终可用；必须复用 Runtime 的执行时 capability gate。模型输出（包括它假定的工具列表）不能授予权限。
3. 预先生成完整计划可以减少逐步决策的模型调用，但无法在中途得到新信息后自行改计划；Phase 10 若做恢复应明确有界重试、记录失败证据，避免无限反思。
4. 在扩展工具或把 Planner 纳入交互入口前，先做 Phase 11 权限控制、统一 trace/metrics 关联与独立工具调用 ID；不要把此内存状态当成 checkpoint。若要测量实际收益，用相同题集、重复运行并分别报告答案正确性、工具路径、token 与耗时。
