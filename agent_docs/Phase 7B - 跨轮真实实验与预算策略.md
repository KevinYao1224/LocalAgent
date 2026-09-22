# Phase 7B：跨轮真实实验与预算策略

## 阶段范围

2026-09-22：本阶段不修改 `AgentSession` 或 `ConversationMemory` 的运行语义，
先新增真实 Ollama 实验，回答 Phase 7A 尚未验证的问题：真实模型能否使用跨轮
历史、事实更新和完整轮次淘汰是否符合预期，以及长工具结果对上下文成本的影响。

新增：

```text
experiments/conversation_memory_live_experiment.py
```

运行方式：

```bash
.venv/bin/python experiments/conversation_memory_live_experiment.py --runs 3
```

可用 `--model` 和 `--base-url` 指定模型与 Ollama 地址。该实验具有模型随机性，
用于探索和记录指标，不作为确定性单元测试。

### 可视化核验入口

`main.py` 提供同一组场景的人类可读版本。它不绕过正式组件，而是为每个场景
创建标准调用链：

```text
AgentSession
    → ConversationMemory
    → AgentLoop
    → HumanReadableLogger
    → OllamaClient
```

运行全部场景或单个场景：

```bash
.venv/bin/python main.py
.venv/bin/python main.py --scenario recall
.venv/bin/python main.py --scenario fact-update
.venv/bin/python main.py --scenario eviction
.venv/bin/python main.py --scenario long-tool
```

每个 user turn 后会显示 run 是否完成、step 数、prompt tokens，以及实际提交后的
Memory canonical message 快照。最后输出场景 PASS/FAIL；存在失败时进程返回非零
状态，便于人工观察之外也被脚本调用。

为了避免长工具结果淹没终端，`HumanReadableLogger` 新增可选
`max_text_chars`。`main.py` 设为 500 字符，Memory 快照另显示前 100 字符。
截断只发生在格式化输出层，event、模型上下文和 Memory 中的数据保持完整；默认
`max_text_chars=None` 仍完整输出，因此现有 Logger 行为不变。

## 实验设计

每个场景使用新的 `AgentSession`，避免历史相互污染：

1. `recall`：第一轮保存精确项目代码，第二轮引用。
2. `fact_update`：先保存旧 region，再更新，最后要求当前值。
3. `whole_turn_eviction`：`max_turns=2`，用两个新轮次淘汰旧事实，要求缺失时
   回答 `UNKNOWN`。
4. `long_tool_result`：仅本场景暴露实验工具 `load_archive`。工具返回约 1.67 万
   字符，标记位于中间；下一轮要求引用该标记。

长输出由实验专用工具产生，不加入正式 `tools` 包。前三个场景使用空 Registry，
因为评测初版把 `load_archive` 暴露给所有场景后，模型曾在淘汰实验中无关调用该
工具，使一次 prompt 异常升至 2598 tokens。该污染样本已废弃；工具可见性隔离后
重新运行正式样本。这也再次说明工具选择本身是上下文和评测条件的一部分。

每个场景记录：

```text
semantic correctness
completed status
turn count
final query prompt tokens
all turns prompt tokens
retained message count
retained content characters
wall-clock latency
final content
```

正确性检查关注目标事实是否出现；更新和淘汰场景同时拒绝旧事实。它不是严格的
格式遵循评分，因为本阶段研究的是 Memory 是否提供了正确事实。

## qwen3.5:9b 结果

环境：本地 Ollama、`qwen3.5:9b`、reasoning replay 默认 `off`，每个场景 3 次。

| scenario | success | avg query prompt | avg retained chars | avg latency |
|---|---:|---:|---:|---:|
| recall | 3/3 | 92 | 124 | 16.32s |
| fact_update | 3/3 | 117 | 194 | 38.90s |
| whole_turn_eviction | 3/3 | 132 | 169 | 59.07s |
| long_tool_result | 3/3 | 2182 | 16765.67 | 3.69s |

观察：

- 模型在此小样本中正确引用旧事实，并以更新后的事实覆盖旧值。
- 完整轮次淘汰后，模型正确回答 `UNKNOWN`，说明测试使用的旧标记确实未继续
  出现在正式消息历史中。
- 长工具结果仍可被引用，但最终查询 prompt 是普通 recall 的约 23.7 倍。
  两者工具 schema 条件不同，不能把全部差值归因于工具结果；但 1.67 万字符历史
  显然已证明“5 个轮次”不等于“可控上下文”。
- 延迟与 prompt token 不呈单调关系。本轮受本地模型 warm state、thinking 长度、
  系统负载等影响，不应用这些数据声称长上下文更快或更慢。
- 两次回答把字面 `</think>` 泄漏到 `content`，但仍包含正确标记。这是模型输出
  格式/adapter 诊断问题，不是 ConversationMemory 丢失事实；后续 Evaluation 应
  将“语义正确”和“格式遵循”分开计分。

样本量仍然很小，因此这里只证明调用链和风险，不宣称统计意义上的模型能力。

## 预算策略设计

实验已经给出实现预算策略所需的最小证据，但不支持立即引入自动摘要。下一小步
建议实现 **完整轮次的 history character budget**：

```text
ConversationMemory 保存最近 max_turns 个完整轮次
        ↓
检索时从最新轮次向前选择完整 suffix
        ↓
累计 history content characters 不超过预算
        ↓
ContextBuilder / AgentSession 仍只收到结构完整的消息
```

设计边界：

1. 第一版应命名为 character budget，而不是 token budget。Ollama 的
   `prompt_eval_count` 在调用后才返回，不能用于调用前硬限制；字符数只是可解释、
   provider-neutral 的近似量。
2. 淘汰单位仍是完整 turn，不能截断 assistant tool call 与 tool observation。
3. 预算应作用于“本次 retrieve 选择”，不要销毁更多已保存轮次。这样调用者可用
   不同预算构造上下文，Memory 存储与 Context 选择职责不混淆。
4. system prompt、当前 user message 和 tools schema 不在 history character budget
   内。API 必须明确这是 history budget，不声称限制整个 provider prompt。
5. 最近一个轮次本身超预算时，第一版应明确排除它并报告选择结果，而不是静默突破
   上限或截断工具结果。摘要/压缩是后续独立策略。
6. 需要让调用者能观察 selected turn count、dropped turn count 和 selected
   characters，才能解释模型为何忘记某项事实。
7. 精确 token budget 留到 adapter 提供 tokenizer/预估接口之后；自动摘要还需要
   来源、更新、失败和错误传播策略，继续暂缓。

建议将上述能力作为 Phase 7C，而不是在本实验脚本中偷加生产行为。

## 结论

Phase 7B 完成了真实跨轮基线和预算策略设计。Phase 7A 的完整轮次窗口语义在本次
小样本中工作正常；当前最明确的缺口不是事实摘要，而是长工具 observation 可让
上下文成本在单轮内失控。下一阶段应先实现可观察、完整轮次粒度的 history
character budget，再用确定性实验验证选择算法，并复跑本文件中的真实场景。
