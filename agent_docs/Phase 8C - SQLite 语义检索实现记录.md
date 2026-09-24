# Phase 8C：SQLite 语义检索最小版

## 范围与依赖方向

8A 是 namespace 内字面检索；8B 给注入上下文的记录加字符预算。8C 增加**显式写入时 embedding、SQLite 持久化向量、查询时余弦相似度排序**。保留原有 `SQLiteLongTermMemory` 行为；应用明确选择 `SQLiteSemanticMemory` 并传入 `TextEmbedder`，其 `search()` 仍符合 `LongTermMemory` 接口，可直接接入 `AgentSession`。不自动写入对话、工具结果或 thinking。

```text
应用 → SQLiteSemanticMemory(namespace, embedder)
          ├── write(text, metadata) → embed(text) → 原文 + 归一化向量同一事务写入
          └── search(query, limit, metadata_filter)
                 → embed(query) → namespace/model_id 过滤
                 → metadata 过滤 → cosine 排序 → top-k
      → AgentSession.run(memory_query=...) → 8B JSON 字符预算 → LLM
```

`memory/semantic.py` 定义 provider-neutral `TextEmbedder` 协议、`ScoredMemory`（可查看相似度）和 `SQLiteSemanticMemory`。写入前验证文本、metadata、向量非空/有限/非零，并归一化；相同 namespace 和 model_id 下的向量必须同维。文本与向量在同一 SQLite 事务提交，失败不留无向量的半条记录。搜索仅扫描当前 namespace 且匹配 model_id 的向量，先过滤 metadata，再按 cosine 降序取 top-k；同分时按 created_at、id 降序保证确定性。`search_with_scores()` 用于实验和诊断；`search()` 只返回原有 `MemoryRecord`。旧的纯文本记录没有向量，不会被语义搜索命中；8A 的字面搜索仍可读取同库文本。

`llm/embedding.py` 的 `OllamaEmbedder` 调用 `/api/embed`，返回单条 embedding；模型调用和网络错误传播给应用，不会静默退回字面检索。向量在 Memory 层统一验证。模型 id 目前取 `ollama:<model tag>`；**模型 tag 被重新指向不同权重但名称未变时，需要换用新 id 或重建向量库**，不能混用不同嵌入空间。SQLite 扫描适用于小规模学习实验，不是向量索引。`metadata_filter` 为已有精确键值过滤；namespace 是应用提供的隔离范围，不是认证。

## 人类可读离线实验

```bash
.venv/bin/python experiments/semantic_memory_experiment.py
```

无需 Ollama。使用几何上可解释的确定性二维向量（不是预置搜索答案），演示不含字面查询词的文本依然按 cosine 命中；模拟 HTTP 检查 `/api/embed` 的单输入请求与返回；重新打开数据库验证持久性，并检查 metadata、namespace/model 隔离、维度与零向量校验、写入原子性、AgentSession 的显式检索和 8B 预算。人类可读输出会打印来源 id、相似度和 PASS。它验证算法及接线，不证明真实 embedding 的语义质量。

可用 embedding 服务时的应用示例：

```python
from llm.embedding import OllamaEmbedder
from memory import SQLiteSemanticMemory

with OllamaEmbedder("<已安装的 embedding 模型>") as embedder:
    store = SQLiteSemanticMemory("memories.sqlite3", "user-123", embedder)
    store.write("Cedar 部署在东京", {"kind": "project"})
    print(store.search_with_scores("Cedar 在哪里运行？", limit=2))
    # session = AgentSession(agent, long_term_memory=store,
    #                        retrieval_character_budget=2000)
    # session.run("Cedar 在哪里运行？", memory_query="Cedar 在哪里运行？")
```

## 本机 Ollama 可用性检查（2026-09-24）

本机 `/api/tags` 仅列出 `qwen3.5:9b`（capabilities: tools/thinking/completion），无 embedding 模型。用本机现有模型 POST `/api/embed` 实测 HTTP **501**：`This server does not support embeddings. Start it with --embeddings`。因此本次未宣称完成真实 embedding 的效果验证；要做在线检索实验，需先启用支持 embeddings 的服务并安装相应模型，然后用**同一模型身份**完成写入和搜索。在线评测还应比较字面检索、语义检索的命中率和被检索内容的模型采纳率。

服务准备好后可运行 `.venv/bin/python experiments/semantic_memory_live_experiment.py --model <embedding-model>`：临时库写三条不同类别的记录、重新打开并显示两条同类候选的 cosine 和正确来源的排名；这是可观察的在线 smoke test，单个查询不是准确率评测。

### 真实 embedding 服务恢复与小规模检索评估（2026-09-24）

服务现在列出 `qwen3-embedding:0.6b`（Q8_0，1024 维、capabilities 含 embedding）。运行：

```bash
.venv/bin/python experiments/semantic_memory_live_experiment.py --model qwen3-embedding:0.6b
.venv/bin/python experiments/semantic_retrieval_eval_experiment.py --chat
```

第一项真实 smoke test 中，询问 `Where is the project hosted?`，东京记录 cosine=0.6517、颜色干扰记录 cosine=0.5090，正确来源排名 1/2。第二项临时写入四条 Cedar 事实（当前部署东京、颜色蓝色、负责人 Maya、旧部署巴黎），对四个英文改写式问题做 full-question 字面子串与语义 top-k 的观察：

| 问题 | 预期来源排名 | 语义 top-2（cosine） | 字面整句命中 |
|---|---:|---|---:|
| Where is Cedar hosted now? | 1/4 | current 0.789, archive 0.747 | 0 |
| What shade is the Cedar product? | 1/4 | color 0.745, owner 0.570 | 0 |
| Who is responsible for Cedar? | 1/4 | owner 0.620, current 0.565 | 0 |
| Where did Cedar run before? | 1/4 | archive 0.673, current 0.591 | 0 |

在这个预先定义的四题小集合中，语义 top-1 为 **4/4**。字面基线是 8A 的**完整问题子串**匹配，并非关键词、全文索引或词项扩展，因此 0/4 不能推断语义检索全面优于所有传统检索算法。当前与旧部署两条的分数接近（0.789 / 0.747），增加相似干扰项后应复测。相似度只用于同次查询的排序，不是“答案正确概率”。

`--chat` 又选用第一题做端到端实验：通过 `AgentSession` 显式检索选中东京记录，JSON payload 182 字符，Ollama chat prompt 167 tokens；`qwen3.5:9b` 完成 1 步、返回 `Tokyo`，来源和答案均正确。这证明该次运行的 retrieval → budget → context → generation 链路连通；不意味着模型对所有问题都可靠。单次运行没有跨模型/多次样本统计意义。

## 下一步发现

- 长文本仍是一条向量；遇到 8B 大记录排除时，后续应评估有来源的分段写入与父记录关联，而不是无来源截断。
- 写入量大时逐条扫描、Python metadata 过滤效率有限，可在测量后考虑索引/批量处理；当前不引入向量数据库。
- 相似度是排序信号，不等于事实正确性或可信度；Memory 内容仍按 8A 的非可信数据注入，工具执行继续受 capability gate 约束。
- 当前 `AgentSession.run(memory_query=...)` 尚不支持传入 metadata_filter；需要应用选择隔离 namespace 或在 store 层明确过滤策略。四题实验未使用 metadata 过滤，端到端运行 top-1 也可能因相似记录导致误选。
