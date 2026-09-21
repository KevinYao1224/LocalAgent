# Phase 6C：ContextBuilder 实现记录

## 1. 本阶段目标

Phase 6A 已经把外部输入、模型响应和工具执行统一记录为 `Trajectory`；
Phase 6B 又为 provider reasoning 加入了 capability 与 provenance。Phase 6C
要解决的问题是：

```text
Trajectory 是完整事实
        ↓
本轮究竟把哪些 Message 发给模型？
```

本阶段引入 `ContextBuilder`，把“正式对话投影”和“本轮上下文策略”分开。
第一版只实现两种 reasoning replay 策略：

```text
off                 默认，不回放 reasoning
latest_pending      只回放最近一个仍等待 continuation 的 reasoning
```

本阶段没有实现 memory、历史摘要、token-budget manager、长期存储、JSONL trace
或 DecisionMemo。这些能力不能和 reasoning replay 一次性混入，否则很难判断
每个策略究竟改变了什么。

## 2. 最终职责边界

现在的模型输入链路为：

```text
AgentState / Trajectory
        │
        ├── ConversationProjector
        │       └── 逐条生成 canonical messages
        │
        └── ContextBuilder
                ├── 读取 replay policy
                ├── 检查 adapter capability
                ├── 检查 provider / model provenance
                ├── 判断 reasoning 是否仍然 pending
                └── 生成本次 LLM 调用使用的 messages
```

三个组件的职责保持明确。

### AgentState

只保存运行事实和控制状态：

```text
InputMessage
AgentStep(ModelResponse, ReasoningBlock, ToolExecution[])
step / max_steps / metadata
```

它不构造 provider messages，也不选择 replay 策略。

### ConversationProjector

把 Trajectory 确定性投影为正式对话：

```text
外部输入                 → 原 Message
带 content/tool_calls    → assistant Message
ToolExecution            → tool Message
空响应或 thinking-only   → 不产生 canonical Message
```

它保持 policy-free：不回放 thinking、不裁剪、不摘要、不注入 memory。

### ContextBuilder

从同一份 Trajectory 构造某一次模型调用使用的上下文。它可以先复用 canonical
projection，再对被策略选中的一个 `AgentStep` 做瞬时增强。增强结果不会反写
Trajectory，也不会改变正式 conversation history。

这个区分很重要：

```text
AgentRunResult.messages
    = canonical conversation

LLM.chat(messages=...)
    = ContextBuilder 为该次调用生成的策略上下文
```

因此调用者、日志和后续持久化不会误把一次实验性 reasoning replay 当成新的
历史事实。

## 3. ConversationProjector 的调整

`agent/conversation.py` 新增：

```python
ConversationProjector.project_entry(entry)
```

`project()` 现在按顺序调用 `project_entry()` 并合并结果。这个小改动让
`ContextBuilder` 能准确知道某个 `AgentStep` 投影出了哪些消息，然后只增强
那个 step 对应的 assistant message。

没有采用“先投影整张列表，再根据列表下标或对象 identity 猜测对应消息”的
方式，因为以下情况会让位置关系不稳定：

```text
thinking-only step       不产生 canonical message
inert empty step         不产生 canonical message
一个 tool step           产生一条 assistant 和多条 tool messages
```

逐 entry 投影使这个关联保持显式，同时没有把 replay policy 塞进
`ConversationProjector`。

## 4. 新增的 Context 类型

`agent/context.py` 新增三个公开类型。

### ReasoningReplayPolicy

```python
class ReasoningReplayPolicy(str, Enum):
    OFF = "off"
    LATEST_PENDING = "latest_pending"
```

构造 `ContextBuilder` 时会通过枚举转换校验传入值。未知策略会立即抛出
`ValueError`，避免拼写错误被静默当成关闭或开启。

### ContextBuildResult

```python
ContextBuildResult
    messages: list[Message]
    replayed_reasoning_step: int | None
    reasoning_replayed: bool
```

除了本次模型输入，它还报告实际回放了哪个 step。这个诊断信息使实验、日志
或未来 trace 能区分“策略配置为 latest_pending”和“本次调用确实找到并回放
了合格 reasoning”。二者并不等价；capability 或 provenance 检查失败时，
配置存在但不会发生回放。

### ContextBuilder

默认构造为：

```python
ContextBuilder(reasoning_replay=ReasoningReplayPolicy.OFF)
```

它可以接收自定义 `ConversationProjector`，但通常使用内部默认实例。

## 5. `off` 策略

`off` 是默认值，也是当前推荐的生产行为。

构建过程仍逐条调用 `ConversationProjector.project_entry()`，但不会选择任何
reasoning step。因此结果与 canonical projection 等价：

```text
ReasoningBlock 仍保存在 Trajectory 中
Message.thinking 不进入模型输入
ContextBuildResult.reasoning_replayed == False
```

这保证引入 `ContextBuilder` 本身不会隐式改变旧行为。

## 6. `latest_pending` 的精确选择规则

`latest_pending` 不是“把最后一段 thinking 放回去”，而是从最近到最早检查
`AgentStep`，并且最多选择一个 block。

### 6.1 Adapter capability

当前 LLM 必须明确声明：

```python
llm.capabilities.reasoning_replay
    is ReasoningReplaySupport.SUPPORTED
```

默认或 `UNSUPPORTED` adapter 直接安全降级为 canonical messages。

### 6.2 Pending / awaits continuation

带工具调用的 step 只有在每个 tool call 都已经得到 execution/observation 后，
才允许把 reasoning 放入下一次模型调用：

```text
tool call 数量 == tool execution 数量
```

这避免在 observation 尚不完整时构造看似可继续的上下文。

没有 tool call、content 为空、但保存了 reasoning 的 thinking-only step，也被
视为等待 continuation。它表示模型产生了内部推理，却还没有给出最终回答。

普通非空最终回答不属于 pending。AgentLoop 正常情况下会在该处结束，本规则
也防止该 reasoning 被后续错误地重新使用。

### 6.3 Inert empty step

满足以下条件的 step 被视为一次没有产生新决策的空重试：

```text
content 为空
tool_calls 为空
reasoning 为 None
```

选择器会跳过这种 step，继续检查它之前的 pending reasoning。否则偶发的模型
空响应会让刚刚完成工具 observation 的 reasoning 无故失效。

### 6.4 Stale reasoning barrier

只要遇到一个不是 inert empty 的新 step，就不会越过它去捞取更旧的 reasoning。
例如，新 step 已经作出新的 tool decision，但没有产生可回放 reasoning，那么
旧 reasoning 会被视为陈旧，而不是继续注入。

这形成了一个明确屏障：

```text
旧 reasoning + tool observation
        ↓
新决策 step（无可用 reasoning）
        ↓
不得回放旧 reasoning
```

### 6.5 Provenance 与 replayable

候选 `ReasoningBlock` 必须同时满足：

```text
replayable == True
block.provider == llm.provider_name
block.model 非空
llm.model_name 非空
block.model == llm.model_name
```

provider 匹配但 model identity 未知也不允许回放。这里选择保守降级，因为
不同模型或模型版本未必能安全解释另一个模型生成的 raw thinking。

## 7. Reasoning 如何进入消息

如果被选中的 tool step 已经投影出 assistant tool-call message，
`ContextBuilder` 会创建一条等价 assistant message，并把：

```text
content
tool_calls
thinking = ReasoningBlock.raw_thinking
```

合并在同一条消息中。随后原有 tool result messages 保持顺序不变：

```text
assistant(thinking + tool_calls)
tool(result 1)
tool(result 2)
```

如果选中的是 thinking-only step，canonical projection 原本没有 assistant
message，builder 会创建：

```text
assistant(content="", thinking="...")
```

整个过程只创建本次调用使用的新 `Message`，不会修改原始 `ModelResponse`、
`ReasoningBlock`、`AgentStep` 或 Trajectory。

## 8. AgentLoop 集成

`AgentLoop.__init__()` 新增可选依赖：

```python
context_builder: ContextBuilder | None = None
```

未传入时使用默认 `ContextBuilder()`，也就是 replay off。每次模型调用前，
AgentLoop 现在执行：

```python
context = self._context_builder.build(state, self._llm)
response = self._llm.chat(
    messages=context.messages,
    tools=available_tools,
)
```

AgentLoop 仍负责控制循环、记录响应和执行工具；它不再直接决定当前消息投影
策略。模型响应和工具结果依然先进入 AgentState，下一轮 ContextBuilder 再从
已记录的事实构造输入。

`AgentRunResult.messages` 继续通过 `ConversationProjector` 生成 canonical
conversation，故启用 replay 也不会改变公开的正式历史。

## 9. 离线实验

新增：

```powershell
.\.venv\Scripts\python.exe experiments\context_builder_experiment.py
```

该实验不需要 Ollama，覆盖：

1. 无效 policy 会抛出 `ValueError`。
2. 默认 `off` 与 canonical conversation 一致。
3. `latest_pending` 把 reasoning 合并进对应 tool-call assistant message。
4. adapter 不支持 replay 时安全降级。
5. provider 不匹配时安全降级。
6. model 不匹配或 model identity 未知时安全降级。
7. block 标记为不可 replay 时安全降级。
8. tool observation 尚未完整产生时安全降级。
9. inert empty step 不会错误屏蔽之前的 pending reasoning。
10. 新决策 step 会阻止陈旧 reasoning 回放。
11. thinking-only step 能生成 reasoning-only assistant message。
12. AgentLoop 注入 builder 后，第二次模型调用确实收到第一步 thinking，且
    tool result 顺序正确。

## 10. 真实 Ollama A/B 实验

新增：

```powershell
.\.venv\Scripts\python.exe experiments\reasoning_replay_ab_experiment.py --runs 3
```

实验使用本地 `qwen3.5:9b` 完成两步算术工具任务，并交替执行 `off` 与
`latest_pending`，以降低固定执行顺序和模型热启动造成的偏差。记录指标包括：

```text
完成与答案正确性
step 数
空响应数
产生 thinking 的响应数
重复工具调用数
prompt / completion tokens
实际 replay 次数
端到端延迟
最终回答
```

模型预热后的 3 × 2 次小样本结果为：

| policy | success | avg steps | avg empty | avg repeated calls | avg prompt tokens | avg completion tokens | avg latency | total replays |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 100% | 3.00 | 0 | 0 | 2071.00 | 218.33 | 3.41s | 0 |
| latest_pending | 100% | 3.00 | 0 | 0 | 2071.00 | 219.67 | 3.41s | 6 |

`latest_pending` 每次 run 实际回放两次，说明 replay 路径确实被执行，而不是
只配置了 policy。可是在这个简单任务上，两组的成功率、step、空响应、重复
调用和平均延迟没有差异；completion token 的微小差异也不足以形成结论。

两组报告的 prompt token 完全相同是一个尚未解释的观察。它可能涉及 provider
模板处理、缓存或指标口径，但当前证据不足，不能据此断言 replay 不占 token，
也不能断言模型忽略了 thinking。后续需要更复杂任务、更多样本和必要时的请求
级检查。

因此本阶段的实验结论是：

```text
传输路径和选择策略可用
当前小样本没有证明 replay 带来收益
默认策略继续保持 off
```

## 11. 完整回归

Phase 6C 完成后，以下检查全部通过：

```powershell
.\.venv\Scripts\python.exe experiments\reasoning_continuity_experiment.py
.\.venv\Scripts\python.exe experiments\context_builder_experiment.py
.\.venv\Scripts\python.exe experiments\conversation_projection_experiment.py
.\.venv\Scripts\python.exe experiments\trajectory_experiment.py
.\.venv\Scripts\python.exe experiments\logging_experiment.py
.\.venv\Scripts\python.exe experiments\multiple_tools_experiment.py
.\.venv\Scripts\python.exe experiments\agent_loop_experiment.py
.\.venv\Scripts\python.exe experiments\schema_validation_experiment.py
.\.venv\Scripts\python.exe experiments\tool_executor_experiment.py
.\.venv\Scripts\python.exe -m compileall -q agent llm runtime tools observability main.py experiments
git diff --check
```

现有工具调用、失败 observation、schema validation、日志、max_steps、空响应、
canonical projection 和 trajectory 行为均未回归。

## 12. 设计结论与下一阶段

Phase 6C 证明了一个关键架构边界：

```text
Trajectory 保存发生过什么
ConversationProjector 定义正式协议历史
ContextBuilder 决定模型这一次看到什么
```

下一阶段进入 Phase 7 短期记忆。第一步应保持小范围：先确定 memory 的数据
边界、生命周期以及它如何作为 ContextBuilder 的输入，再实现最小的 append /
retrieve 行为。不要在第一步同时加入复杂摘要、精确 token budget、向量检索或
长期记忆。

DecisionMemo 继续暂缓。只有未来 reasoning replay 或 memory 实验显示确实需要
provider-neutral 决策摘要时，才根据已观察到的缺口重新设计它。
