# Phase 3：Runtime 实现记录

## 1. 本阶段目标

Phase 2 的 `AgentLoop` 同时负责控制循环和处理工具异常：

```text
AgentLoop
    ├── 调用模型
    ├── 查找并执行工具
    ├── 捕获工具异常
    └── 继续或终止循环
```

这种结构会让后续的验证、权限、timeout、retry 和日志全部进入 AgentLoop。本阶段把工具执行提取成独立 Runtime：

```text
AgentLoop
    ↓
ToolExecutor
    ↓
ToolRegistry
    ↓
Tool
```

`AgentLoop` 现在只负责控制流程。`ToolExecutor` 负责把不可信的 `ToolCall` 转换为结构化 `ToolResult`。

## 2. ToolResult

`runtime/result.py` 定义：

```python
ToolResult(
    tool_name: str,
    success: bool,
    value: Any = None,
    error: str | None = None,
)
```

其中：

```text
tool_name  标识结果来自哪个工具
success    执行是否成功
value      成功时的原始 Python 返回值
error      失败时的错误说明
```

结果对象是 frozen dataclass。创建后不能被意外修改，适合作为一次已经发生的执行记录。

`__post_init__()` 保证两条不变量：

```text
成功结果不能同时带有 error
失败结果必须带有 error
```

调用方通过 `succeeded()` 和 `failed()` 工厂方法创建结果，不需要反复手写布尔值与空字段组合。

### 模型 observation

结构化结果需要转换为 `Message.content` 才能送回模型。`to_message_content()` 使用以下规则：

```text
失败       → "Error: ..."
字符串值   → 保持原字符串
JSON 值    → json.dumps(..., ensure_ascii=False)
其他对象   → str(value)
```

JSON 序列化让数字、布尔值、列表、字典和 `None` 拥有稳定表示，同时保留非 JSON Python 对象的兼容回退。

## 3. ToolExecutor

`runtime/executor.py` 接收 `ToolRegistry`。它提供两个操作：

```text
available_tools()
返回可提供给模型的工具。

execute(tool_call)
查找并执行一个工具，始终返回 ToolResult。
```

当前执行路径为：

```text
ToolCall
    ↓
ToolRegistry.execute(name, arguments)
    ↓
Tool.execute(arguments)
    ↓
ToolResult.succeeded(...)
```

以下预期异常会转换为失败结果：

```text
ToolNotFoundError
ToolExecutionError
```

编程错误仍然会向外抛出，避免把 Runtime 自身的 bug 伪装成普通工具失败。

`available_tools()` 让 AgentLoop 只依赖 Executor。以后权限策略可以决定哪些注册工具能在某次运行中暴露给模型。

本阶段还把 `LLM.chat()` 的工具参数类型从错误的 `list[dict[str, Any]]` 修正为 `list[Tool]`。Provider adapter 接收领域层 `Tool`，再由 `OllamaClient` 转换成 Ollama JSON schema；这与实际执行路径和既定架构边界一致。

## 4. AgentLoop 的变化

构造函数从：

```python
AgentLoop(llm, registry)
```

变成：

```python
AgentLoop(llm, executor)
```

循环不再包含 `_execute_tool()` 和异常捕获，而是：

```python
tool_result = executor.execute(call)
tool_results.append(tool_result)
history.append(Message(
    role="tool",
    tool_name=call.name,
    content=tool_result.to_message_content(),
))
```

`AgentRunResult` 新增 `tool_results`，因此调用方可以直接分析成功率、原始值和错误，而不必重新解析消息字符串。这也是未来 Evaluation 和 Observability 的数据基础。

## 5. 实验

### ToolExecutor 实验

运行：

```bash
.venv/bin/python experiments/tool_executor_experiment.py
```

覆盖四条路径：

1. `multiply(23, 47)` 返回成功结果，`value=1081`。
2. 未注册工具返回失败结果。
3. 缺失参数返回失败结果。
4. handler 主动抛出异常时，返回失败结果。

每个实验同时打印结构化字段和最终 observation，便于比较 Runtime 内部表示与模型看到的内容。

### AgentLoop 回归实验

运行：

```bash
.venv/bin/python experiments/agent_loop_experiment.py
```

原有四个控制流实验继续通过，并新增对 `result.tool_results` 的断言。这验证了提取 Runtime 后消息顺序、错误恢复和步数限制没有改变。

### Ollama 集成实验

运行 `main.py` 后，本地 `qwen3.5:9b` 仍然完成：

```text
Final answer: 23 multiplied by 47 is **1081**.
```

## 6. 下一阶段

当前 `Tool.execute()` 使用 `inspect.signature(...).bind(...)`。它能发现缺少参数和未知参数，但下面的调用仍可能通过绑定：

```python
multiply(a="twenty-three", b=[47])
```

Phase 4 应在 ToolExecutor 中加入基于 `Tool.parameters` 的 JSON Schema 验证：

```text
ToolCall
    ↓
查找 Tool
    ↓
根据 Tool.parameters 验证 arguments
    ↓
执行 handler
    ↓
ToolResult
```

这样发给模型的 schema 和 Runtime 使用的 schema 将拥有同一个 source of truth。
