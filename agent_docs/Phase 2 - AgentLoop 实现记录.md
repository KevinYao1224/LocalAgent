# Phase 2：AgentLoop 实现记录

## 1. 本阶段目标

这一阶段把 `main.py` 中固定的：

```text
LLM → Tool → LLM → End
```

抽成可重复运行的控制循环：

```text
ModelResponse
    │
    ├── 没有 tool_calls ──→ 完成
    │
    └── 存在 tool_calls
            │
            ▼
       执行每个工具
            │
            ▼
       追加 observation
            │
            └──────────────→ 再次调用模型
```

实现位置是 `agent/loop.py`。`main.py` 现在只负责组装 `OllamaClient`、`ToolRegistry` 和 `AgentLoop`，不再了解循环内部细节。

## 2. 新增的领域对象

### `StopReason`

`StopReason` 明确区分两种退出原因：

```text
COMPLETED
模型返回普通回答，任务正常结束。

MAX_STEPS
达到最大模型决策次数，Runtime 强制停止。
```

使用枚举比用任意字符串更可靠，调用方也可以明确处理不同状态。

### `AgentRunResult`

一次运行会返回：

```text
response     最后一次模型响应
messages     完整消息轨迹
steps        实际模型调用次数
stop_reason  退出原因
```

`completed` 属性是对正常完成状态的便捷判断。

保留完整 `messages` 很重要。它让实验和未来的 trace 系统能够回答：模型请求了什么、Runtime 执行了什么、结果怎样回到模型。

## 3. 循环怎样工作

`AgentLoop.run()` 每一步依次完成：

1. 把当前历史和可用工具交给 LLM。
2. 把模型响应按 `role="assistant"` 追加到历史。
3. 如果没有 `tool_calls`，把该响应作为最终回答返回。
4. 如果有 `tool_calls`，按顺序通过 `ToolRegistry` 执行。
5. 把每个结果按 `role="tool"` 追加到历史，进入下一步。

输入消息列表会先被浅复制。因此循环可以构建自己的轨迹，又不会悄悄修改调用方传入的列表。

当前 `max_steps` 统计模型决策次数。模型在最后一步请求的工具仍会执行并记录，随后循环以 `MAX_STEPS` 结束。这个语义在未来加入有副作用的工具前需要结合权限策略再次评估。

## 4. 错误为什么要进入消息历史

模型产生的工具名和参数都属于不可信输入。第一版循环处理两类预期错误：

```text
ToolNotFoundError
模型请求了注册表中不存在的工具。

ToolExecutionError
参数无法绑定，或工具 handler 执行失败。
```

这些错误会变成：

```text
Message(role="tool", content="Error: ...")
```

循环不会因此崩溃。模型能在下一步读取错误 observation，并选择修正参数、换工具或向用户解释失败。

`ToolRegistry.get()` 原先会先触发原生 `KeyError`，导致自定义异常实际无法生效；本阶段将它修正为明确的 `ToolNotFoundError`。

## 5. 实验

运行：

```bash
.venv/bin/python experiments/agent_loop_experiment.py
```

该实验使用 `ScriptedLLM`，不需要启动 Ollama。它不模拟语言能力，只固定返回准备好的响应，从而单独检验 Runtime 控制流。

实验覆盖：

1. `multiply(23, 47)` 被执行，结果 `1081` 回到模型，第二次响应正常结束。
2. 不存在的工具被转换为错误 observation，循环继续。
3. 缺失参数被转换为错误 observation，循环继续。
4. 模型持续请求同一工具时，`max_steps=2` 能终止循环。

本阶段还使用真实的本地 `qwen3.5:9b` 执行 `main.py`。端到端结果为：

```text
Final answer: 23 multiplied by 47 is **1081**.
```

这同时验证了 Ollama adapter 生成的 tool schema、assistant tool call 历史和 tool result 消息格式。

## 6. 下一阶段

当前错误仍以字符串表示，`AgentLoop` 也直接依赖 `ToolRegistry.execute()`。Phase 3 应新增：

```text
ToolResult
    success
    value
    error

ToolExecutor
    接收 ToolCall
    查找 Tool
    执行 Tool
    捕获预期异常
    返回 ToolResult
```

依赖关系将变为：

```text
AgentLoop → ToolExecutor → ToolRegistry → Tool
```

这样参数验证、权限、timeout、retry 和日志以后都可以加入执行层，而不需要继续扩大 `AgentLoop`。
