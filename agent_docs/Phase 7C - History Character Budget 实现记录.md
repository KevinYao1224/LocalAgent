# Phase 7C：完整轮次 History Character Budget

## 阶段范围

2026-09-23：根据 Phase 7B 的真实实验结果，为 `ConversationMemory` 增加检索级
history character budget。目标不是精确限制 provider token，而是在模型调用前用
一种可解释、provider-neutral 的近似量，避免一个很长的工具 observation 自动进入
后续每一轮上下文。

本阶段没有实现 tokenizer、自动摘要、消息截断、向量检索或长期记忆。reasoning
replay 默认值仍为 `off`。

## API 与职责

`ConversationMemory` 仍用 `max_turns` 管理实际保存的完整轮次，并保留原有：

```python
messages = memory.retrieve()
```

`retrieve()` 的兼容语义不变：返回当前轮数窗口中的全部历史。新增：

```python
selection = memory.select(history_character_budget=4000)
```

返回的 `ConversationMemorySelection` 包含：

```text
messages
selected_turns
dropped_turns
selected_characters
history_character_budget
```

`selected_characters` 只累计所选历史中每条 `Message.content` 的 Python 字符数，
明确不包括 system prompt、当前 user message、tools schema、provider 模板与协议开销。
因此它叫 character budget，不叫 token budget，也不承诺限制完整 prompt。

`AgentSession` 新增可选构造参数：

```python
session = AgentSession(
    agent=agent,
    memory=ConversationMemory(max_turns=5),
    history_character_budget=4000,
)
```

每次 `run()` 前调用 `memory.select()`，并通过
`session.last_history_selection` 暴露该轮实际使用的选择指标。默认预算为 `None`，
所以既有调用者继续携带 `max_turns` 窗口中的全部历史。

## 选择算法

算法从最新轮次向前累计，但最终仍按时间正序返回：

```text
最新 turn
    ↓ 能放入预算
继续检查前一 turn
    ↓ 不能放入预算
立即停止
    ↓
返回已选中的完整、连续 suffix
```

遇到第一个放不下的轮次后不会跳过它再选择更老的小轮次。否则返回值就不再是
最新连续 suffix，并可能让被新事实覆盖的旧事实绕过新事实重新进入上下文。

轮次是最小选择单位。包含 assistant tool call、一个或多个 tool observation 和
最终回答的轮次，要么整体进入，要么整体排除。若最新一个轮次本身已超过预算，
选择结果为空；Runtime 不静默突破预算，也不切断工具交换。

预算只影响一次 `select()`，不会从 `ConversationMemory` 删除轮次。这样同一个
Memory 可以由不同调用者用不同预算读取，选择策略也不会破坏保存状态。真正的
存储淘汰仍只由 `max_turns` 控制。

## 为什么选择字符数

Phase 7B 中普通 recall 的最终查询约为 92 prompt tokens，而保留约 1.67 万字符
工具结果后达到 2182 prompt tokens。`prompt_eval_count` 只能在 Ollama 调用完成后
得到，当前 adapter 也没有调用前 tokenizer。因此字符数虽然不是 token 数，却能在
调用前确定计算、跨 provider 使用，并让“为什么忘记了某轮”可以通过指标解释。

未来如果 adapter 提供 tokenizer 或 prompt 估算接口，应新增独立的 token budget
策略，而不是把当前字段改名后假装字符数等于 token 数。

## 离线实验

新增 `experiments/history_character_budget_experiment.py`。运行：

```bash
.venv/bin/python experiments/history_character_budget_experiment.py
```

实验以人类可读方式打印选择结果，并确定性验证：

1. 从三个轮次中选择预算可容纳的最新两个完整轮次。
2. `selected_turns`、`dropped_turns` 和 `selected_characters` 准确。
3. 最新工具轮次超预算时返回空 suffix，不拆分工具交换，也不选择更老轮次。
4. 预算选择不删除保存的轮次，`retrieve()` 仍能读到完整窗口。
5. `AgentSession` 实际只把 system、所选 suffix 和当前 user 发送给模型。
6. 当前轮完成后仍正常提交到 Memory。
7. 精确等于预算时允许进入，零预算选择空历史。
8. 负数、布尔值、浮点数和字符串预算被拒绝。
9. 修改 selection 返回的消息不会污染 Memory 快照。

运行输出中的核心可观察信息类似：

```text
Suffix selection: selected=2, dropped=1, characters=42 (passed)
Oversized newest turn: selected=0, dropped=2; ... (passed)
```

这组实验验证的是 Runtime 的确定性上下文选择，不代表模型在被排除事实后一定按
指定格式回答。真实 Ollama 实验仍需把 semantic correctness、格式遵循、工具可见性
和上下文选择指标分开观察。

## 真实 Ollama 验证

2026-09-23 第一次复跑 Phase 7B 时，首个请求在 180 秒后触发
`httpx.ReadTimeout`。随后确认 `/api/version`、`/api/tags` 正常，但 `/api/ps`
最初没有驻留模型。通过空 `/api/generate` 请求预热 `qwen3.5:9b`，模型约 4.3 秒
完成加载；简单 READY 请求在约 0.24 秒内完成。超时不能完全归因于冷加载，更像是
一次服务端推理卡住。

预热后用 `main.py` 拆分复跑四个 Phase 7B 场景，全部通过：

| scenario | final answer | final query prompt tokens | verdict |
|---|---|---:|---:|
| recall | `CEDAR-731` | 92 | PASS |
| fact-update | `AMBER-42` | 117 | PASS |
| eviction | `UNKNOWN` | 132 | PASS |
| long-tool | `VAULT-48291` | 2182 | PASS |

这些 prompt token 与 Phase 7B 的三次样本平均值一致，说明默认
`history_character_budget=None` 保持了原有行为。

随后新增真实预算实验：

```bash
.venv/bin/python experiments/history_character_budget_live_experiment.py
```

实验使用 4000 字符预算和一个只能成功读取一次的 `load_archive`。第一轮保存
16677 个 content 字符；第二轮选择结果及模型行为为：

```text
selected_turns=0
dropped_turns=1
selected_characters=0
query_prompt_tokens=708
reacquisition_attempts=2
successful_reacquisitions=0
final_content='UNKNOWN'
marker_still_stored=True
verdict=PASS
```

这里的 708 是第二轮两次模型调用的累计 prompt tokens：第一次为 309，模型在同一
响应中发出两个 `load_archive` 调用；一次性工具均返回“archive no longer
available”，第二次模型调用为 399 tokens 并回答 `UNKNOWN`。这证明超预算轮次没有
进入上下文，且预算没有从 Memory 删除原始标记。

实验设计过程中还发现：如果 `load_archive` 可以重复成功，模型会在历史被排除后
主动再次调用工具，并重新得到 `VAULT-48291`。这不是 Memory 泄漏。最初版本因此
得到语义上的 FAIL（累计 2466 prompt tokens），随后改为一次性工具，才把“历史
检索”与“外部事实重取”隔离。工具是否可见、是否可重复调用必须作为评测条件。

复现默认行为与预算行为：

```bash
.venv/bin/python experiments/conversation_memory_live_experiment.py --runs 1
.venv/bin/python experiments/history_character_budget_live_experiment.py
```

## 重要边界与后续发现

- `max_turns` 是存储窗口；`history_character_budget` 是单次检索策略，两者不能
  合并为一个概念。
- `dropped_turns` 表示本次没有进入上下文的已保存轮次，不表示数据已被删除。
- `last_history_selection` 在模型异常时仍描述该次尝试使用的历史，因为选择发生在
  provider 调用前；失败运行仍不会提交当前轮。
- 字符预算无法计入工具 schema、provider 模板、system 和当前问题，因此还不是
  完整的上下文窗口管理器。
- 超大最新轮次被整体排除会损失其中的有用事实。后续若要解决，应单独设计有来源、
  失败语义和更新规则的摘要/压缩策略，不能在本阶段静默截断。
- 历史被排除不等于事实必然不可获得：模型可能再次调用仍可用的工具。Memory
  evaluation 必须隔离或显式记录这种重新获取路径。

## 下一阶段建议

Phase 7 的最小短期记忆链路已经具备完整轮次存储、真实跨轮基线与可观察字符预算。
下一步优先进入 Structured Trace + Evaluation：先为 run/step 增加稳定标识、时间戳、
延迟和 JSONL 输出，再把现有确定性实验逐步组织成可汇总的 evaluation cases。这样
以后比较摘要、token budget 或模型变更时，才能用同一套指标证明效果，而不是只看
单次回答。
