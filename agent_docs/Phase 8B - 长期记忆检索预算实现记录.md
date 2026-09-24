# Phase 8B：长期记忆检索字符预算

## 选择这个小阶段的原因

Phase 8A 的 `retrieval_limit` 限制的是条数。一条很长的记录仍可能占据大量模型上下文；Phase 7C 的短期历史字符预算也不覆盖临时检索消息。本阶段在 `AgentSession` 的显式检索入口加入可选 `retrieval_character_budget`，默认 `None` 保留原行为。SQLite 中的记录不被截断或删除。

## 数据路径与计数语义

```text
应用显式传入 memory_query
    → LongTermMemory.search(query, limit=retrieval_limit)
    → 按检索顺序逐条尝试加入 JSON array
    → 只注入预算内的完整记录（独立 user 消息，非可信参考数据）
    → last_retrieval / last_retrieval_selection
```

预算计算 `len(json.dumps(payload, ensure_ascii=False))`：包括 JSON 括号、分隔符、转义、记录 id、创建时间、文本和 metadata；它是 Python 字符数，不是 token、字节或整次模型请求的长度。标签、system prompt、短期历史、当前用户问题和工具 schema 不在此预算内。值须为非负整数，`0` 不注入记录也不发送空的检索消息。

一条记录放不下时整条跳过，再尝试后续较短的命中；这与 Phase 7C 的“最新连续轮次 suffix”不同：长期检索没有对话轮次的时间连续性约束。`retrieval_limit` 仍先限制搜索返回的候选条数；预算不能让搜索自动扩展到 limit 之外。没有显式查询时 `last_retrieval_selection is None`，有查询但无命中或预算为零时报告候选数、选中数和排除数。`last_retrieval` 仅包含实际进入上下文的记录，供调用者审计；检索消息仍不进入短期 ConversationMemory。

调用示例：

```python
session = AgentSession(
    agent,
    long_term_memory=store,
    retrieval_limit=5,
    retrieval_character_budget=2000,
)
result = session.run("Cedar 的情况？", memory_query="Cedar")
print(session.last_retrieval_selection)
print([record.id for record in session.last_retrieval])
```

## 人类可读实验

```bash
.venv/bin/python experiments/long_term_retrieval_budget_experiment.py
.venv/bin/python experiments/long_term_memory_experiment.py
```

离线模拟模型记录实际收到的消息，无需 Ollama。实验构造一个新且超长的命中和一个较短的旧命中，按较短记录的**完整 JSON**长度设预算，打印 candidates / selected / dropped / JSON chars 并断言真实模型上下文仅含短记录 id；检查 `0` 预算、不检索的下一轮、短期记忆未混入来源、SQLite 内容未被删除、默认不限预算及非法参数。实验只证明送给模型的上下文和指标，不代表模型一定正确使用事实。Phase 8A 的能力拒绝实验可单独验证被检索内容无法恢复已禁用工具的执行权。

### 真实 Ollama 实验（2026-09-24）

```bash
.venv/bin/python experiments/long_term_retrieval_budget_live_experiment.py
# 服务重启后可单独补跑：
.venv/bin/python experiments/long_term_retrieval_budget_live_experiment.py --scenario zero_budget
```

实验使用本机 `qwen3.5:9b`（Q4_K_M），每个场景新建 `AgentSession`，带 `HumanReadableLogger` 的预览日志；临时 SQLite 写入较短项目记录和较长归档记录。先断言 `result.messages` 中确实只含被选中的来源 id，再单独评价模型答案。该次较短记录的 JSON 长 168 字符，较长记录长 18381 字符（JSON 包含来源和 metadata）。已完成的观测如下，token 来自 Ollama `prompt_eval_count`：

| 场景 | 预算 | 候选/选中/排除 | 注入 JSON 字符 | prompt tokens | 延迟 | 模型答案 / 预期 |
|---|---:|---:|---:|---:|---:|---|
| 查询归档，全部注入 | 无 | 2/2/0 | 18549 | 2379 | 14.26s | `VAULT-48291` / `VAULT-48291` |
| 查询项目，排除超长归档 | 168 | 2/1/1 | 168 | 172 | 5.21s | `CEDAR-731` / `CEDAR-731` |
| 查询归档，归档被排除 | 168 | 2/1/1 | 168 | 173 | 68.88s | `CEDAR-731` / `UNKNOWN`（错误） |

已完成的三个场景中，上下文选择断言均通过，模型回答为 2/3 正确。第三场景只送了项目记录，**没有送归档记录或归档 code**；模型却把项目 code 误答为归档 code。它还产生 4658 completion tokens（主要是 thinking），因此该场景耗时明显更长。预算降低输入规模不保证模型正确识别缺失的事实，也不保证输出延迟下降。

随后单独补跑 `zero_budget` 场景时，模型调用立即返回 `httpx.ConnectError: [Errno 111] Connection refused`；`curl --noproxy '*' http://127.0.0.1:11434/api/version` 也无法连接。该在线场景**未完成**，不能计入通过率；离线实验已验证零预算不产生检索消息。上表为单次探索性观察，不能据此推出稳定的成功率或性能提升。

服务恢复后，同日通过 `/api/version` 确认 Ollama `0.34.3`，重新运行**全部四个场景**，而非只补跑零预算。新的临时库 id 和创建时间会改变少量 prompt tokens；本次结果：

| 场景 | 候选/选中/排除 | JSON 字符 | prompt tokens | completion tokens | 延迟 | 答案 / 预期 |
|---|---:|---:|---:|---:|---:|---|
| unbounded_archive | 2/2/0 | 18549 | 2376 | 296 | 9.26s | `VAULT-48291` / `VAULT-48291` |
| bounded_project | 2/1/1 | 168 | 171 | 312 | 4.59s | `CEDAR-731` / `CEDAR-731` |
| bounded_archive | 2/1/1 | 168 | 172 | 5376 | 79.58s | `UNKNOWN` / `UNKNOWN` |
| zero_budget | 2/0/2 | 0 | 48 | 2310 | 33.52s | `UNKNOWN` / `UNKNOWN` |

本次上下文选择 4/4、模型答案 4/4；与前一次 `bounded_archive` 答错形成对照：同一模型和策略对缺失事实的回答可能波动，不应把单次 4/4 解释为稳定成功率。`zero_budget` 的模型输入仅有 system 和当前 user，没有检索消息；prompt token 为 48。`bounded_archive` 和 `zero_budget` 的 completion token / 延迟仍明显偏高，提示即便压低检索输入，模型仍可能花大量生成预算判断缺失事实。

## 后续发现与建议

- 字符预算只限制检索 JSON 本体，不限制附加标签或整个 prompt；精确 token 预算需 provider 层测量或估算。
- 如果超长记录经常被排除，应先考虑应用控制的分段写入和质量评估，再研究 embedding / semantic retrieval。检索顺序目前仍是字面匹配后的创建时间顺序，不代表语义相关性。
- 对“事实缺失”的评测不能只检查有没有注入被排除记录，还要检查模型是否把另一条可见记录错误关联到目标；这次真实模型实验已出现这一误归因。
- 统一 trace 阶段可记录本次候选、选中、排除和来源 id；目前这些指标只在当前会话实例上可读，未写 JSONL。
