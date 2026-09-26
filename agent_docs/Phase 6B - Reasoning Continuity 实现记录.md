# Phase 6B：Reasoning Continuity 实现记录

## 1. 本阶段目标

Phase 6A 已经能完整保存模型的 `thinking`，但字符串本身无法回答：

```text
它来自哪个 provider 和 model
provider 是否能在下一轮接收它
它是否应该进入下一轮上下文
```

Phase 6B 引入两部分：

```text
LLMCapabilities / ReasoningReplaySupport
ReasoningBlock
```

本阶段只建立类型、来源、关联和安全降级，不自动启用 reasoning replay，
也不实现完整 ContextBuilder。

## 2. Ollama capability 调查

2026-09-20 查阅 Ollama 当前官方文档，得到两个明确事实。

### Chat API message schema

Ollama Chat API 将 `thinking` 列为 message object 的字段，并说明 thinking
model 的 reasoning trace 位于 `message.thinking`，最终答案位于
`message.content`。

### Tool calling history

Ollama 官方 tool-calling 示例会把完整的 `response.message` 加入下一轮
messages。流式示例更明确要求：累计每个 chunk 的
`thinking/content/tool_calls`，然后把三者一起作为 assistant message 放回
后续请求。

因此可以确认：

```text
Ollama chat API 的传输层支持 reasoning replay
```

但不能推出：

```text
每个模型都生成 thinking
每个模型都正确消费历史 thinking
replay 一定提高任务成功率
应该默认启用 replay
```

因此 Ollama adapter 声明 transport capability 为 supported，而默认上下文
策略继续保持 replay off。收益和模型差异留给 Phase 6C A/B 实验。

参考资料：

- Ollama Chat API：https://docs.ollama.com/api/chat
- Ollama Thinking：https://docs.ollama.com/capabilities/thinking
- Ollama Tool Calling：https://docs.ollama.com/capabilities/tool-calling

## 3. Provider capability

`llm/base.py` 新增：

```python
ReasoningReplaySupport
    UNSUPPORTED
    SUPPORTED

LLMCapabilities
    reasoning_replay
```

`LLM.capabilities` 的默认值是 `UNSUPPORTED`。任何没有显式声明的 provider
都会安全降级，不能因为返回了 thinking 就自动允许回放。

LLM 同时提供来源信息：

```python
provider_name
model_name
```

默认 provider name 使用 adapter 类名，model 为 `None`。具体 adapter 可以
覆盖它们。

## 4. ReasoningBlock

`ReasoningBlock` 保存：

```text
provider
model
raw_thinking
replayable
provider_state
```

`LLM.create_reasoning_block(response)` 负责把原始 `ModelResponse.thinking`
转换为带 provenance 的 ReasoningBlock：

```text
没有 thinking
    → None

有 thinking + provider unsupported
    → ReasoningBlock(replayable=False)

有 thinking + provider supported
    → ReasoningBlock(replayable=True)
```

`raw_thinking` 与原始 response 保持一致，不混入 final content。
`provider_state` 为未来无法只用文本表达的 provider continuation state 预留，
当前 Ollama 使用 `None`。

AgentLoop 收到每个 ModelResponse 后，会立即通过当前 LLM adapter 创建
ReasoningBlock，并与 ModelResponse 一起记录到对应 AgentStep：

```text
AgentStep
    model_response
    reasoning: ReasoningBlock | None
```

## 5. Ollama adapter

`OllamaClient` 现在明确提供：

```text
provider_name = "ollama"
model_name = 构造 client 时使用的 model
reasoning_replay = supported
```

`Message` 新增可选 `thinking` 字段。Ollama 的 message converter 只在该字段
非空时写入 wire format：

```json
{
  "role": "assistant",
  "content": "",
  "thinking": "...",
  "tool_calls": []
}
```

当前 `ConversationProjector` 不会把 AgentStep.reasoning 写入 Message，因此
正常 AgentLoop 仍不会自动发送 thinking。这个 converter 能力只是为 Phase
6C 的显式 replay 策略准备传输路径。

## 6. 默认 replay-off

必须区分：

```text
ReasoningBlock.replayable == True
    表示 provider adapter 有能力传输

上下文策略启用 replay
    表示 Runtime 决定本轮实际发送
```

Phase 6B 只实现前者。即使测试 provider 或 Ollama 返回 replayable block，
第二轮模型调用仍只收到 `ConversationProjector` 产生的 canonical messages。

这保持以下不变量：

```text
默认行为与 Phase 6A 相同
thinking 不等于最终答案
thinking 不参与权限和参数验证
provider capability 不会隐式改变上下文
```

## 7. 实验

新增：

```bash
.venv/bin/python experiments/reasoning_continuity_experiment.py
```

不依赖 Ollama server，覆盖：

1. 未声明 capability 的 ScriptedLLM：保存 ReasoningBlock，但 replayable 为 false。
2. 声明 supported 的测试 provider：block replayable 为 true，但第二轮仍不自动收到 thinking。
3. Ollama adapter：验证 capability、provider/model provenance 和 thinking wire conversion。

## 8. 回归验证

阶段完成时上述实验及相关轨迹/工具回归与编译检查通过；当前实验入口见
[代码与实验阅读指南](./代码与实验阅读指南.md)。

现有日志、消息角色顺序、工具错误恢复、空响应处理和 max_steps 行为不变。

## 9. 暂缓 DecisionMemo

早期 Phase 6B 曾加入 DecisionMemo 数据结构，但它没有明确的生成来源和消费
策略，也没有参与 AgentLoop 或 ContextBuilder。为避免维护尚未被验证的抽象，
当前实现已删除相关类型、AgentStep 字段、State 方法和实验。

等 Reasoning Replay 策略通过真实模型实验并稳定后，再根据实际缺口决定是否
需要 provider-neutral memo，以及它应由谁生成、何时生成和如何进入上下文。

## 10. 下一阶段

Phase 6C 实现 ContextBuilder，第一版策略仍应只有：

```text
reasoning_replay = off | latest_pending
default = off
```

`latest_pending` 只能选择最近一个仍与当前 tool observation 相关的 replayable
ReasoningBlock。它必须同时检查 provider 和 model 是否与当前 LLM 一致。

之后对本地 `qwen3.5:9b` 做 A/B 实验，比较：

```text
最终答案成功率
空 content 响应数
重复工具调用数
平均 step 数
prompt/completion token
latency
```

实验稳定证明有收益之前，不改变默认策略。
