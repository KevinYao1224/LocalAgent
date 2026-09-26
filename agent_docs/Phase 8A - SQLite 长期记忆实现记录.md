# Phase 8A：SQLite 文本长期记忆与显式读写

## Phase 8 的总体目标与当前进度

Phase 8 在 Phase 7 的短期完整轮次记忆之外提供跨进程持久化与检索。**8A 完成**文本与 metadata 持久化、限定范围的字面子串检索、显式读写和可选的会话注入；检索预算与语义 embedding / cosine top-k 后续分别由 8B、8C 实现。自动写入和跨用户共享仍未实现。

## 依赖方向与 API

```text
Application ──显式 write/search──→ SQLiteLongTermMemory(namespace)
Application ──run(memory_query=...)──→ AgentSession
                                     ├──→ LongTermMemory.search()
                                     └──→ AgentLoop → ContextBuilder → LLM
```

`memory/long_term.py` 定义 `LongTermMemory` Protocol（write/search）、不可变 `MemoryRecord`（id、text、UTC created_at、metadata）和 `SQLiteLongTermMemory`。调用者明确选择数据库路径和 namespace；每次 SQL 查询都限制 namespace。这个范围隔离依赖调用者正确分配 namespace，**不是身份认证**。数据库连接按操作打开、事务结束后关闭；写入先校验非空文本和 JSON object metadata，采用参数化 SQL，避免把查询当作 SQL。metadata 采用 JSON 文本保存和恢复；无额外依赖。

检索为文本 **大小写不敏感的字面子串**（SQLite 内置 `lower()` 对非 ASCII 的大小写折叠有限），不按词义检索；`%` 等通配符没有特殊意义。按 `created_at DESC, id DESC` 排序，在 metadata 精确键值过滤后取 limit 条，确保新记录的其他类型不会挤掉较早的合格记录。结果带来源 id 和时间；这是确定性的排序规则，但同一时间戳下按 id 而非写入先后打破平局。当前小规模实验采用逐条过滤 JSON metadata，记录量大时应改进索引和过滤效率。

`agent/session.py` 增加可选 `long_term_memory`、`retrieval_limit` 和 `run(..., memory_query=...)`。默认不检索；只有调用者传入 query 才检索。命中内容以 JSON 编码、带来源 id 和时间的 **untrusted reference data** 标签，作为独立 user 消息放在当前用户问题之前；不进入 system prompt，不配置工具列表。`last_retrieval` 提供最近一次运行的命中记录与来源，未检索/未命中为空。该临时消息出现在本轮 trajectory / `AgentRunResult.messages` 中供审计，但提交短期 ConversationMemory 时从当前用户消息开始切片，避免它在下一轮被无意重复注入。`ToolExecutor` 仍是当前执行 capability 的唯一入口。

需要明确：文字标签并不能保证模型永远不受 prompt injection 影响；真正的动作限制由 ToolExecutor 的 capability gate 执行。长期记录可能包含敏感数据；当前写入由应用显式控制，不会自动持久化会话、tool result 或 raw thinking。

## 使用示例

```python
from memory import SQLiteLongTermMemory
from agent import AgentSession

store = SQLiteLongTermMemory("local_memories.sqlite3", namespace="user-123")
record = store.write("Project Cedar uses blue widgets", {"kind": "project"})
matches = store.search("Cedar", limit=3, metadata_filter={"kind": "project"})

session = AgentSession(agent_loop, long_term_memory=store, retrieval_limit=3)
result = session.run("What do you know about Cedar?", memory_query="Cedar")
print([item.id for item in session.last_retrieval])
```

应用代码决定写入哪些经过选择的事实、使用哪个 namespace、何时检索和检索哪个词；没有配置 store 的会话沿用 Phase 7 的默认行为。

## 可复现实验（人类可读）

```bash
.venv/bin/python experiments/long_term_memory_experiment.py
```

实验在临时目录创建 SQLite 文件，无需 Ollama。输出写入 id、scope、metadata、命中数和来源，以及每组 PASS。它断言重开进程式连接后数据仍在、metadata 过滤先于 top-k、namespace 隔离、字面查询及错误输入；随后把包含旧工具名称和越权指令的记录显式检索到模型上下文，模拟模型提出该工具调用，检查当前广告工具为空、handler 无权执行、结果是 `tool_not_available`。最后检查临时检索消息不被提交到短期记忆，下一轮不提供 query 时不自动读写。

## 后续设计记录

- 按 namespace 的删除/更新与保留期限仍待研究。8B 已增加检索字符预算，但 `retrieval_limit` 本身只控制条数；精确 token 预算仍未实现。
- 大规模记录需要索引/全文检索与明确的 Unicode 规范化，再考虑 embedding / 相似度评估。评测应区分检索命中、模型是否采纳正确事实和是否误执行记忆中的指令。
- 与 Phase 12/13 的结构化 trace 和 evaluation 对接时，保留命中 id、namespace（注意访问控制）、query、筛选条件和数量；当前仅提供 `last_retrieval` 和可审计 trajectory，不声称已完成统一 trace。
