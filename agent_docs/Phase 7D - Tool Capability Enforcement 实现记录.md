# Phase 7D：Tool Capability Enforcement

## 问题与威胁模型

2026-09-23：在准备进入长期记忆阶段前，发现 `ToolExecutor` 原有接口存在潜在的
授权分叉：

```text
available_tools() -> 告诉模型当前有哪些工具
execute(call)     -> 直接从 Registry 查找并执行
```

当前实现尚未加入权限过滤，所以两者都使用 Registry 全集，暂时不会产生分叉。但若
未来权限层只从 `available_tools()` 移除工具，而工具仍注册在 Registry 中，模型可能
从历史 tool call、旧提示或自行构造的输出中恢复工具名称与参数。原执行路径会找到
该工具并运行。

因此，tool schema 是否发给模型不能作为安全边界。模型输出始终是不可信输入：

```text
未向模型暴露工具
≠
模型不可能请求工具
```

## 修复后的不变量

本阶段把以下规则固化到 `ToolExecutor`：

> 一个工具只有同时存在于 Registry，并且存在于执行时刻的
> `available_tools()` 中，才可以执行。

执行顺序现在是：

```text
ToolCall（不可信）
    ↓
Registry existence check
    ↓
Current available_tools capability check
    ↓
JSON Schema validation
    ↓
Tool handler
```

Capability 检查发生在参数验证和 handler 之前。被禁用工具不会因为参数正确、历史中
曾成功运行或仍在 Registry 中而获得执行机会。

`execute()` 特意调用与模型广告相同的 `available_tools()`，而不是复制一份近似的
判断。这样未来权限实现即使通过 subclass 或动态策略过滤 `available_tools()`，执行
路径也会遵守同一视图，避免“广告列表过滤了、执行检查忘了改”的漏洞。

## ToolExecutor API

构造器新增可选参数：

```python
ToolExecutor(
    registry,
    enabled_tool_names={"read_file", "calculator"},
)
```

并新增：

```text
set_enabled_tools(names | None)
disable_tool(name)
enable_tool(name)
```

语义：

- `None`：兼容旧行为，Registry 中当前工具全部可用。
- 集合/其他名称 iterable：进入受限模式，只有集合内已注册工具可用。
- 空集合：所有已注册工具都保留，但不广告、不可执行。
- `disable_tool()`：从当前 capability 集合移除工具，不修改 Registry。
- `enable_tool()`：只允许重新启用已注册工具。
- 配置未注册名称、空名称或把单个字符串误当名称集合时立即报错。

新增 `ToolErrorType.TOOL_NOT_AVAILABLE`。它与以下错误分开：

```text
tool_not_found       Registry 中不存在
tool_not_available   Registry 中存在，但当前 context 无 capability
validation_error     capability 允许，但参数无效
execution_error      capability 和参数通过，但 handler 失败
```

拒绝会作为普通 tool observation 返回模型，因此 AgentLoop 可以继续生成解释或改选
允许的工具；handler 不会被调用。

## 越权回归实验

新增：

```text
experiments/tool_capability_experiment.py
```

运行：

```bash
.venv/bin/python experiments/tool_capability_experiment.py
```

实验流程刻意模拟用户提出的场景：

1. `read_secret` 注册在 Registry，并在旧轮次成功运行一次。
2. 正式历史保留旧 assistant tool call、参数和 tool result。
3. Runtime 调用 `disable_tool("read_secret")`，但不从 Registry 删除工具。
4. 模拟模型根据历史再次产生完全正确的 `read_secret(key="ALPHA")`。
5. 两次模型调用收到的当前 advertised tools 都为空。
6. Runtime 返回 `tool_not_available`，handler 计数没有从 1 增加到 2。

人类可读输出：

```text
registry_contains=read_secret
advertised_tools=[]
model_requested=read_secret(key='ALPHA')
runtime_result="Error [tool_not_available]: ..."
handler_execution_count=1
verdict=PASS
```

`experiments/tool_executor_experiment.py` 还验证了：

- 构造时空 capability 集合会同时隐藏并禁止工具。
- enable 后可以执行，disable 后立即再次禁止。
- 未注册配置和错误 iterable 被拒绝。
- 模拟未来权限 subclass 仅覆盖 `available_tools()` 返回空列表时，`execute()` 仍拒绝
  Registry 中存在的工具。

## 安全边界与尚未完成的工作

这次修复解决的是 **AgentLoop 通过 ToolExecutor 执行时的 capability 一致性**，不
等于 Phase 11 的完整 PermissionPolicy：

- 尚无按用户、资源、路径、读写操作或参数内容进行授权的 Policy 对象。
- 尚无 capability 变更事件、审计主体、批准流程或并发锁。
- 应用代码若绕过 ToolExecutor 直接调用 `ToolRegistry.execute()`，不受本层保护。
  Agent Runtime 必须继续坚持 `AgentLoop -> ToolExecutor -> Registry` 依赖方向。
- 工具在一次模型响应的多个 call 之间被禁用时，后续每个 call 都会重新检查当前
  `available_tools()`；但完整并发语义仍留到 Phase 11/14 设计。
- 历史 tool result 本身可能包含敏感数据。禁用工具只能阻止新副作用，不能擦除已经
  进入 Memory 的秘密；敏感结果还需要独立的 retention/redaction 策略。

## 是否可以进入 Phase 8

**可以进入，但建议限定为 Phase 8A 的非执行型长期记忆基础。**

理由：

1. Phase 7 的完整轮次短期记忆、预算和真实实验已经完成。
2. 本次发现的“根据历史恢复旧工具调用并越权执行”路径已在 ToolExecutor 层封闭。
3. Phase 8 第一小步可以只实现 Memory interface、SQLite 文本/metadata 持久化和
   确定性检索，不需要引入 filesystem/shell 等高风险 action tools。

进入 Phase 8 时必须保持以下约束：

- 检索出的 memory 是不可信 data，不是 system instruction，也不能授予 capability。
- 长期记忆中出现工具名称或参数，不得影响 ToolExecutor 的当前可用集合。
- 第一小步不要同时加入 embedding、自动写入所有对话、跨用户共享或敏感工具结果。
- 当前路线中 Structured Trace + Evaluation 仍然重要。若直接开始 Phase 8A，应至少
  为 write/retrieve 命中和来源保留可观察字段，随后补齐统一 trace/eval。

因此结论不是“权限系统已经完成”，而是：当前已具备安全开始 **Phase 8A：SQLite
文本记忆与显式读写策略** 的最小前提；完整 PermissionPolicy 仍必须在引入真实世界
副作用工具前完成。
