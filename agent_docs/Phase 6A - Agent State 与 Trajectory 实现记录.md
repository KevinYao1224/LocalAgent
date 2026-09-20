# Phase 6A：Agent State、Trajectory 与 Conversation Projection 实现记录

## 1. 本阶段目标

Phase 5.5 之前，`AgentLoop` 使用分散的局部变量保存消息历史、工具结果和
最后一次模型响应。Phase 6A 首先引入 `AgentState` 与 `AgentStep`，随后在
Phase 6A.1 进一步解决一个职责混合问题：

```text
AgentState 一方面记录运行事实
AgentState 另一方面构造发给模型的 Message
```

最终设计将这两个方向拆开：

```text
AgentState
    只记录运行事实和控制状态

ConversationProjector
    从 Trajectory 构造 canonical conversation messages
```

本阶段仍不实现 reasoning replay、DecisionMemo、历史裁剪或完整
`ContextBuilder`。

## 2. 最终数据流

```text
外部输入 Message
    ↓ record_input
AgentState.trajectory
    ↓ ConversationProjector.project
canonical messages
    ↓
LLM
    ↓ record_model_response
AgentStep
    ↓ 执行 ToolCall
ToolExecution(call, result)
    ↓ record_tool_execution
AgentState.trajectory
```

每次调用模型前，`AgentLoop` 都从当前 Trajectory 重新投影 canonical
messages。AgentLoop 不再创建 assistant 或 tool message，AgentState 也不再
存储或构造 messages。

## 3. Trajectory 的事实类型

### InputMessage

`InputMessage` 表示从当前 Agent 运行外部传入的 conversation message：

```python
InputMessage(message: Message)
```

它可以保存 system、user，也可以保存调用方提供的既有 assistant/tool
历史。`AgentState.from_messages()` 会按输入顺序逐条调用 `record_input()`，
因此原始历史不会丢失。

### AgentStep

`AgentStep` 表示一次实际返回的模型响应：

```python
AgentStep(
    step: int,
    model_response: ModelResponse,
    tool_executions: list[ToolExecution],
)
```

每个 `ModelResponse` 都会进入 AgentStep，包括：

```text
普通最终回答
工具调用响应
thinking-only 响应
content、thinking 和 tool_calls 全为空的响应
```

### ToolExecution

`ToolExecution` 显式保存一次工具调用及其结果：

```python
ToolExecution(
    call: ToolCall,
    result: ToolResult,
)
```

只保存 `ToolResult` 无法稳定表达结果属于哪个调用，特别是在同一个模型响应
包含多个调用，或者未来加入 call ID 和并发执行时。显式关联避免调用方通过
工具名称或列表下标猜测。

`AgentState.record_tool_execution()` 会检查：

```text
当前 step 已经存在 ModelResponse
ToolCall 确实属于当前 ModelResponse
ToolResult.tool_name 与 ToolCall.name 一致
```

## 4. AgentState 的职责

当前 `AgentState` 的实际存储字段为：

```text
max_steps
step
trajectory
metadata
```

状态改变通过四个明确操作完成：

```text
record_input(message)
begin_step()
record_model_response(response)
record_tool_execution(call, result)
```

下面的数据不再重复存储，而是从 Trajectory 派生：

```text
current_task
steps
tool_results
```

`current_task` 是最后一条外部 user input；`steps` 是全部 AgentStep；
`tool_results` 是所有 ToolExecution 的结果汇总。这使 Trajectory 成为事实源，
避免 messages、steps 和 tool_results 三份可变列表之间出现不一致。

AgentState 仍持有 `step` 和 `max_steps`，因为它们是 Runtime 控制状态，不是
conversation 表示。

## 5. ConversationProjector

`agent/conversation.py` 定义了纯粹的确定性投影：

```python
ConversationProjector.project(trajectory) -> list[Message]
```

转换规则为：

```text
InputMessage
    → 原始 Message

AgentStep 有 content 或 tool_calls
    → assistant Message

thinking-only 或全空 AgentStep
    → 不生成 Message

ToolExecution
    → role="tool" Message
```

相同 Trajectory 始终生成相同的 canonical messages。Projector 不修改
Trajectory，也不负责：

```text
reasoning replay
token budget
历史截断
摘要
memory
工具过滤
```

未来 Phase 6C 的 `ContextBuilder` 可以把 canonical projection 作为基础，再
决定哪些信息真正发送给模型：

```text
Trajectory
    ↓
ConversationProjector
    ↓ canonical conversation
ContextBuilder
    ↓ policy-aware model context
LLM
```

## 6. AgentLoop 的职责

`AgentLoop` 继续负责控制顺序，因为只有控制循环知道何时收到了输入、模型
响应和工具结果。它现在执行的是状态转换，而不是消息构造：

```text
记录输入
    ↓
投影当前 conversation
    ↓
调用模型并记录 ModelResponse
    ↓
执行工具并记录 ToolExecution
    ↓
完成、继续或 max_steps
```

这种依赖是有意保留的：AgentLoop 是 orchestrator，应该触发状态变化；需要
移除的是它对 assistant/tool message 构造规则的了解。

Provider 或 Runtime 异常继续原样向外抛出。`AgentLoop.last_state` 保留异常
发生前的状态。Provider 没有返回响应时，`step` 会记录调用尝试，但不会虚构
一个 AgentStep。

## 7. AgentRunResult 兼容性

`AgentRunResult.state` 仍是事实来源。原有接口继续可用：

```python
result.messages       # 由 ConversationProjector 动态生成
result.steps          # 模型调用尝试次数
result.tool_results   # 从 ToolExecution 派生
```

轨迹相关接口现在明确区分：

```python
result.trajectory     # InputMessage | AgentStep 的完整时间顺序
result.agent_steps    # 仅 AgentStep
```

完整 Trajectory 包含外部输入，因此不能再假设 `result.trajectory[0]` 一定是
AgentStep。只分析模型步骤的调用方应使用 `result.agent_steps`。

## 8. 实验

### Conversation Projection 实验

```powershell
.\.venv\Scripts\python.exe experiments\conversation_projection_experiment.py
```

验证：

1. system/user 输入按原顺序投影。
2. tool call assistant message 和 tool result message 正确生成。
3. 全空与 thinking-only 响应保留在 Trajectory，但不会生成 conversation message。
4. 在后续 step 前追加的新 user input 保持时间顺序。
5. `current_task` 随最后一条外部 user input 更新。
6. 两个同名 ToolCall 仍分别保留与各自 ToolResult 的显式关联。

### Trajectory 实验

```powershell
.\.venv\Scripts\python.exe experiments\trajectory_experiment.py
```

验证正常完成、工具失败、空响应、thinking-only、max_steps 和 provider
异常，并额外检查 ToolExecution 保存的是原始 ToolCall 与对应 ToolResult。

## 9. 回归验证

以下命令全部通过：

```powershell
.\.venv\Scripts\python.exe experiments\conversation_projection_experiment.py
.\.venv\Scripts\python.exe experiments\trajectory_experiment.py
.\.venv\Scripts\python.exe experiments\logging_experiment.py
.\.venv\Scripts\python.exe experiments\multiple_tools_experiment.py
.\.venv\Scripts\python.exe experiments\agent_loop_experiment.py
.\.venv\Scripts\python.exe experiments\schema_validation_experiment.py
.\.venv\Scripts\python.exe experiments\tool_executor_experiment.py
.\.venv\Scripts\python.exe -m compileall -q agent llm runtime tools observability main.py experiments
```

现有日志文本、消息角色顺序、工具错误恢复和 max_steps 行为保持不变。本阶段
没有运行真实 Ollama，因为修改位于 provider-neutral 的状态和 projection
层；Phase 6B 调研 reasoning continuation 时再做真实模型实验。

## 10. 下一阶段

Phase 6B 将在当前事实模型上设计：

```text
ReasoningBlock
provider reasoning replay capability
```

`ConversationProjector` 继续保持 policy-free。Reasoning replay 是否启用、
回放哪一块 reasoning，以及如何降级，必须由之后的 ContextBuilder 和 provider
capability 共同决定。
