# Phase 4：Schema Validation 实现记录

## 1. 本阶段目标

模型产生的 `ToolCall.arguments` 属于不可信输入。此前 `Tool.execute()` 只使用 `inspect.signature().bind()`，能发现缺少参数和额外参数，但不能阻止：

```python
multiply(a="twenty-three", b=[47])
```

本阶段让同一个 `Tool.parameters` 同时服务两个方向：

```text
Tool.parameters
    ├──→ OllamaClient → 模型看到的工具 schema
    └──→ Runtime → handler 执行前的参数验证
```

## 2. 依赖选择

项目新增两个直接依赖并记录在 `requirements.txt`：

```text
httpx>=0.28,<1
jsonschema>=4.23,<5
```

`httpx` 是已有 Ollama adapter 的依赖；`jsonschema` 是本阶段新增依赖。

第一版采用 JSON Schema Draft 2020-12。项目已有手写的 `Tool.parameters`，直接验证它可以保持单一事实来源。当前没有引入 Pydantic，以免同时维护 Pydantic model 和手写 schema。

## 3. ToolArgumentsValidator

`runtime/validation.py` 新增：

```text
ToolArgumentsValidator
ToolArgumentsValidationError
ToolSchemaError
```

验证分为两个层次：

1. `Draft202012Validator.check_schema()` 检查工具作者提供的 schema 是否合法。
2. `validator.iter_errors(arguments)` 检查模型参数是否符合 schema。

模型参数错误会汇总并带上类似 JSONPath 的位置：

```text
$.retries
$.tasks[0].priority
```

这比只返回“参数无效”更有利于模型在下一轮修正调用。

无效 schema 属于开发配置错误，不是模型可以修正的输入错误。因此 `ToolSchemaError` 会直接向外抛出，不会转换成失败的 `ToolResult`。

## 4. ToolExecutor 执行顺序

新的执行顺序是：

```text
查找工具
    ↓ 找不到
tool_not_found

验证 arguments
    ↓ 不符合 schema
validation_error

执行 handler
    ↓ handler 抛出异常
execution_error

成功
    ↓
ToolResult.succeeded(...)
```

验证发生在 handler 之前。实验会记录 handler 调用次数，确认非法参数没有产生业务副作用。

`ToolExecutor` 支持注入自定义 validator；默认创建 `ToolArgumentsValidator`。这为以后增加 schema 缓存或不同验证策略保留了替换位置。

## 5. ToolResult 错误分类

`ToolResult` 新增 `error_type`，使用 `ToolErrorType` 枚举：

```text
tool_not_found
validation_error
execution_error
```

成功结果必须同时满足：

```text
error is None
error_type is None
```

失败结果必须同时包含非空 `error` 和 `error_type`。

发送给模型的 observation 现在为：

```text
Error [validation_error]: Invalid arguments for tool ...
```

模型可以同时看到错误类别、具体字段路径和验证原因。Runtime 和 Evaluation 则可以直接读取枚举，不需要解析字符串。

## 6. Tool.execute() 的异常边界修正

原实现把签名绑定和 handler 调用放在同一个 `try` 中，因此 handler 自己抛出的 `TypeError` 会被错误描述成参数绑定失败。

现在分成两个阶段：

```text
signature.bind(**arguments)
    → 参数与 Python 签名不一致

handler(**arguments)
    → 工具业务执行失败
```

JSON Schema 负责模型输入验证，Python signature binding 继续作为 schema 与 handler 定义不一致时的最后防线。

## 7. 实验

运行：

```powershell
.\.venv\Scripts\python.exe experiments\schema_validation_experiment.py
```

实验工具包含 enum、整数范围、数组和嵌套对象，覆盖：

1. 字段类型错误。
2. `additionalProperties` 额外字段。
3. enum 非法值。
4. 超出 maximum。
5. 嵌套数组对象中的字段错误。
6. 合法的嵌套参数成功执行。
7. 无效工具 schema 直接抛出 `ToolSchemaError`。

同时运行原有 ToolExecutor 和 AgentLoop 实验，确认错误分类变化没有破坏循环恢复。真实 Ollama 的 `multiply(23, 47)` 调用也继续返回 `1081`。

## 8. 下一阶段

Phase 5 将扩展 calculator：

```text
add
subtract
multiply
divide
```

重点不是增加算术函数，而是验证：

```text
模型能否选择正确工具
前一步工具结果能否成为下一步输入
多步 AgentLoop 是否正确终止
执行错误能否反馈给模型并恢复
```
