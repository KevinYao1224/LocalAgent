# Phase 5.5：Human Readable Logging 实现记录

## 1. 插入本阶段的原因

多工具 AgentLoop 已能运行，但本地实验只能在结束后查看 `AgentRunResult`。为了观察模型每一步看到了什么、请求了什么工具以及 Runtime 返回了什么，本阶段在 Agent State 之前插入一个轻量日志模块。

目标是：

```text
结构化事件负责表达发生了什么
Logger 负责决定怎样输出
AgentLoop 不直接 print
```

## 2. 模块结构

新增：

```text
observability/
    __init__.py
    events.py
    logger.py
```

`events.py` 不负责输出，只定义事实。`logger.py` 接收事件并格式化。这样未来可以让同一批事件同时进入终端、JSONL 或测试收集器。

## 3. 第一版事件

当前事件包括：

```text
AgentStarted
ModelCallStarted
ModelResponseReceived
ModelCallFailed
ToolExecutionStarted
ToolExecutionFinished
ToolExecutionFailed
EmptyModelResponse
AgentFinished
```

事件使用 frozen dataclass，包含日志需要的结构化字段，例如 step、ToolCall、ToolResult、token 数量和停止原因。

`ModelCallFailed` 记录 provider 调用异常。AgentLoop 输出失败事件和 `AgentFinished(stop_reason="error")` 后仍会重新抛出原异常，让应用决定重试或退出。

## 4. Logger 接口

`AgentLogger` 是只有一个方法的 Protocol：

```python
def log(self, event: AgentEvent) -> None:
    ...
```

AgentLoop 依赖这个接口，不依赖具体终端实现。

### NullLogger

未传 logger 时，AgentLoop 自动使用 `NullLogger`。它接收并丢弃事件，因此库的默认行为保持安静，现有实验不会突然出现额外输出。

### HumanReadableLogger

显式传入 `HumanReadableLogger` 后，它会输出：

```text
初始消息和 max_steps
每一步的历史消息数量
可用工具名称
模型文本和 tool calls
模型 thinking（与可见 content 分开）
工具参数、成功结果或错误分类
prompt/completion token 和 done_reason
空响应处理决策
最终停止原因、步数和内容
```

构造函数接受任意文本 stream：

```python
HumanReadableLogger()               # sys.stdout
HumanReadableLogger(stream=buffer)  # StringIO 或文件
```

这使日志既能显示在终端，也能在实验中被完整捕获和断言。

## 5. AgentLoop 集成点

AgentLoop 在下列边界发出事件：

```text
run 开始
    ↓
模型调用开始
    ↓
模型返回 / 模型异常
    ↓
工具执行开始
    ↓
工具执行完成
    ↓
空响应继续 / 正常完成 / max_steps 完成
```

AgentLoop 不包含输出格式字符串。添加 JSONL Logger 时不需要修改控制循环。

## 6. 空响应历史修正

日志集成后的真实 Ollama 实验发现：空的 assistant 响应如果追加到消息历史，连续空响应可能产生无效的消息排列并导致 provider 返回 HTTP 400。

现在：

```text
有 content 或 tool_calls
→ 追加 assistant Message

content 为空且没有 tool_calls
→ 发出 EmptyModelResponse
→ 不追加历史
→ 在 max_steps 范围内重试
```

空响应仍出现在日志中，因此不会丢失诊断信息；它只是不再污染发送给模型的上下文。

## 7. Main 使用方式

`main.py` 显式开启：

```python
AgentLoop(
    llm=llm,
    executor=executor,
    max_steps=6,
    logger=HumanReadableLogger(),
)
```

此前 main 手写的工具结果和最终答案 `print()` 已删除，所有运行信息现在来自统一 Logger。

## 8. 实验

新增：

```bash
.venv/bin/python experiments/logging_experiment.py
```

实验使用 `StringIO` 捕获日志，覆盖：

1. `add → multiply → final answer` 的成功多步轨迹。
2. `divide(10, 0)` 的 `execution_error` 轨迹。
3. 模型 provider 抛出异常时的失败轨迹和重新抛出行为。

另外验证了 `max_steps` 分支会输出工具结果和完整结束事件。

真实 Ollama 运行能够显示工具 schema 选择、参数、token 数和最终回答，例如：

```text
--- Step 1: model call ---
Tool calls: 1
  1. add({"a": 12, "b": 7})
Tool result: add -> 19

=== Agent run finished ===
Stop reason: completed
Steps: 3
Final content: ...95.
```

### Thinking 记录

`ModelResponse` 新增 `thinking` 字段，`OllamaClient` 从原始 `message.thinking` 读取。HumanReadableLogger 会将它和可见回答分开输出：

```text
Model thinking:
  ...模型内部思考文本...
Model content: ...面向用户的回答...
```

thinking 只用于诊断，不参与最终回答判断，也不会被追加到会话历史。AgentLoop 仍然只把 `content` 和 `tool_calls` 转换成 assistant Message。

真实 `qwen3.5:9b` 实验确认了此前空响应的主要原因。工具执行结束后的响应实际为：

```text
Model thinking: The result of (12 + 7) multiplied by 5 is 95.
Model content: <empty>
Tool calls: 0
completion=20
done_reason=stop
```

这说明模型生成的 token 位于 Ollama `message.thinking`，旧 adapter 只读取 `message.content`，所以 Runtime 过去只能看到空响应。即使读取 thinking，也不能把它直接视为用户答案：本次实验中模型之后重新调用了工具，最终以 `max_steps` 结束，说明 thinking 表达的是内部状态而不是可靠的最终输出。

## 9. 与完整 Observability 的边界

本阶段服务本地人工实验，没有提前完成全部 Phase 12。后续仍需要：

```text
event timestamp
model/tool latency
run_id / trace_id
JSONL 持久化
结构化 token 和错误统计
敏感字段过滤
```

当前事件接口为这些能力提供了基础，而终端 Logger 保持简单、同步和可读。
