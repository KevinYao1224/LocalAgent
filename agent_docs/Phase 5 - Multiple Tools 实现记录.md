# Phase 5：Multiple Tools 实现记录

## 1. 本阶段目标

此前只有 `multiply`，无法验证模型的工具选择和跨步骤数据依赖。本阶段扩展 calculator，并测试：

```text
(12 + 7) × 5
```

目标轨迹为：

```text
add(12, 7)
    ↓
19
    ↓
multiply(19, 5)
    ↓
95
    ↓
最终回答
```

## 2. Calculator 工具集合

`tools/calculator.py` 现在定义：

```text
add(a, b)       → a + b
subtract(a, b)  → a - b
multiply(a, b)  → a × b
divide(a, b)    → a ÷ b
```

`subtract` 和 `divide` 的工具描述明确说明参数顺序有意义，帮助小模型正确选择 `a` 和 `b`。

四个工具共享相同形状的二元数字 schema。`_binary_number_parameters()` 每次创建一个新的字典，避免多个 Tool 共享同一个可变 schema 对象。

`divide()` 在 `b == 0` 时抛出 `ValueError`。零是合法的 JSON number，因此它会通过 schema validation，再由 handler 报告业务执行错误，最终分类为 `execution_error`。

模块还导出不可变顺序集合：

```python
calculator_tools = (
    add_tool,
    subtract_tool,
    multiply_tool,
    divide_tool,
)
```

入口和实验可以循环注册全部 calculator 工具，不需要在多处重复工具名称。

## 3. 包级公共接口

`tools/__init__.py` 新增导出所有函数、Tool 对象和 `calculator_tools`。调用方可以选择：

```python
from tools import add
from tools import add_tool
from tools import calculator_tools
```

分别用于直接计算、单独注册工具或注册完整 calculator 集合。

## 4. Main 集成流程

`main.py` 现在循环注册 `calculator_tools`，并把示例问题改为：

```text
Calculate (12 + 7) multiplied by 5.
Use a separate tool call for each arithmetic operation.
```

系统消息要求模型一次只调用一个算术工具，并使用返回的中间结果决定下一步。这里的约束用于学习和观察顺序依赖，不是通用 Agent 必须采用的策略。

运行结束后，入口会先打印结构化工具轨迹，再打印最终答案：

```text
Tool 1: add -> 19
Tool 2: multiply -> 95
Final answer: ...95.
```

## 5. Final response detection 修正

真实 `qwen3.5:9b` 首次集成运行时产生：

```text
add → 19
multiply → 95
assistant(content="", tool_calls=[])
```

原来的 AgentLoop 只检查 `tool_calls`，因此把这个空响应误判成最终答案。

完成条件现在是：

```python
not response.tool_calls
and response.content.strip()
```

无工具调用但内容为空时，循环会继续下一步，并仍受 `max_steps` 限制。这样可以恢复偶发空响应，同时避免无限重试。

## 6. 实验

新增：

```powershell
.\.venv\Scripts\python.exe experiments\multiple_tools_experiment.py
```

覆盖四种行为。

### 依赖型多步调用

脚本模型依次产生：

```text
add(12, 7) → 19
multiply(19, 5) → 95
final answer
```

实验检查工具名称、结果、模型调用步数和完整消息角色顺序。

### 同一响应中的多个调用

一个模型响应同时请求：

```text
subtract(10, 4)
divide(20, 5)
```

Runtime 按模型给出的顺序执行，结果为 `6` 和 `4.0`。这验证了 multiple tool calls 的消息处理，但当前没有并发执行。

### 除零恢复

`divide(10, 0)` 返回：

```text
Error [execution_error]:
Tool 'divide' failed: Cannot divide by zero.
```

错误作为 observation 返回模型，下一步模型可以解释失败并正常结束。

### 空响应恢复

脚本模型先返回空响应，再返回正式答案。实验确认第一次不会被视为完成，第二次非空响应才结束循环。

## 7. 真实模型验证

本地 `qwen3.5:9b` 实际运行结果：

```text
Tool 1: add -> 19
Tool 2: multiply -> 95
Final answer: The result of (12 + 7) multiplied by 5 is 95.
```

这同时验证了工具选择、中间结果传递、连续模型调用和最终回答检测。

## 8. 下一阶段

Phase 6 将引入 `AgentState`，逐步收纳当前散落在 AgentLoop 中的：

```text
messages
step
max_steps
tool_results
metadata
```

统一状态对象将为 ContextBuilder、Memory、事件 trace 和状态持久化建立边界。
