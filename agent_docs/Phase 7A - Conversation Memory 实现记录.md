# Phase 7A：会话短期记忆

## 范围与决策

2026-09-21：进入 Phase 7 的第一个小阶段。6C 的离线验证和已记录的真实模型
A/B 支持保留默认 replay off。本阶段的验证目标是跨轮上下文正确性，不需要
重复简单算术 A/B，也不能据此宣称模型的记忆能力已经获得统计意义上的提升。

新增 `memory/conversation.py` 的 `ConversationMemory` 和 `agent/session.py`
的 `AgentSession`。只保存进程内最近 N 个完整、已完成的用户轮次。默认 N=5。
Phase 7 整体尚未完成；token budget、摘要和检索继续留待后续。

## 生命周期与依赖

```text
调用者持有 AgentSession
    → 固定 system prompt + memory.retrieve() + 当前 user
    → AgentLoop.run(messages)
    → 新 AgentState / Trajectory
    → ContextBuilder → 模型上下文
    → completed 后只把当前轮追加到 memory
```

`AgentLoop` 管一次运行，`AgentSession` 管跨轮会话；`ConversationMemory` 只管
完整消息轮次的存储与读取，不依赖 AgentLoop 或 Ollama。每次运行独立计算
step/max_steps，历史作为 InputMessage 进入 Trajectory，不冒充本轮 AgentStep。

ContextBuilder 的 build 接口无需改动：历史已经是本轮 State 的输入事实。
这样 result.messages 仍能完整表示本轮收到的正式输入及产生的回答。Memory
是受限窗口而不是审计存储；如需完整历史，应由调用者另行保存运行结果。

## ConversationMemory

- `append(messages)`：接收一个 user 开始、最终 assistant answer 结束的完整轮次。
  校验成功后深拷贝保存并清除 thinking；无效输入不会淘汰已有轮次。
- `retrieve()`：按时间顺序展平历史，返回深拷贝，包含 tool_calls 嵌套 arguments。
- `turn_count`：当前保存的轮次数量。
- `clear()`：清空当前实例的历史。
- `max_turns`：构造时指定正整数；超过容量后淘汰最旧完整轮次。

采用完整轮次作为窗口单位，避免留下孤立的 tool result 或缺失 observation 的
assistant tool_calls。验证按当前同步执行器的工具调用顺序核对 tool_name，
支持同名多次调用；目前协议没有 call ID，未来并发/外部历史导入需要补充 ID
关联，不能把名称与顺序验证当成通用工具协议验证器。

轮次中不接受 system message、第二个 user、缺失/多余/顺序错误的工具结果，
也不接受空最终回答。工具失败产生的正常 observation 可以被保存；只有运行
完成才提交，并不要求每次工具执行都成功。

## AgentSession

`AgentSession(agent, system_prompt="", memory=None)` 默认创建独立 Memory。
调用者显式传入同一 memory 实例时会共享历史；独立会话应使用独立实例。

`run(user_content)` 接收非空用户文本，固定 system prompt 每次置于最前面，
不存入窗口，不被历史淘汰。调用者仅传本轮文本，不必手工拼接历史。

成功完成后，用调用前的 history_length 从 canonical result.messages 中提取
本轮新增部分。这一偏移来自明确的输入前缀长度；投影器逐条保留 InputMessage，
所以不依赖模型产生了多少 step。若直接 append 全部 result.messages，则会
递归重复存储旧历史，本实现刻意避免这一点。

`max_steps` 返回未完成结果时不提交 memory；异常传播时也不提交。未完成运行
可通过返回结果或 AgentLoop.last_state 诊断。这里的“未提交”仅涉及会话历史，
已经执行的工具副作用不会撤销。下一次调用是新一轮，不是失败运行的恢复。

仅 canonical messages 跨轮保存，不保存 ReasoningBlock 或 raw thinking。
因此上一轮 reasoning 不会因 memory 绕过 replay provenance 检查。本轮原始
Trajectory 仍保留 reasoning，ContextBuilder 的显式 latest_pending 仍可正常工作。

## 使用方式

```python
from agent import AgentSession
from memory import ConversationMemory

# agent 是已配置好模型和执行器的 AgentLoop。
session = AgentSession(
    agent=agent,
    system_prompt="Use tools when needed.",
    memory=ConversationMemory(max_turns=3),
)
first = session.run("My project is Cedar.")
second = session.run("What is my project name?")
session.memory.clear()
```

默认 max_turns=5 表示本轮开始时最多带入五轮历史，加上当前轮；成功提交后
只保留最近五轮。它不保证 token 或字节上限，单个巨大工具结果仍可能撑满上下文。
本阶段也不支持持久化、并发会话调用或失败续跑。

## 离线实验与发现

运行：

```powershell
.\.venv\Scripts\python.exe experiments\conversation_memory_experiment.py
```

实验使用确定性模拟模型，其中 RecallLLM 根据实际收到的历史是否包含项目名
来回答，而不是无条件返回预设的正确答案。验证：

1. 同会话跨轮保留信息；新会话和 clear 后无法读取旧信息。
2. 每轮 State 与 step budget 独立；current_task 是新请求。
3. 只追加本轮增量，旧历史没有重复累积。
4. 同名多工具调用及其结果按完整轮次保存与淘汰，system prompt 始终在前。
5. 修改旧结果的嵌套工具参数或 retrieve 返回值不会污染保存的快照。
6. 显式启用 replay 时当前轮可回放；下一用户轮收到的历史不含 thinking。
7. max_steps 与 provider 异常不提交；下一次调用只带之前已完成历史。
8. 格式错误、缺失工具结果或不完整轮次在修改 memory 前被拒绝。

以上证明框架送出的上下文符合约定，不代表真实模型一定准确使用其中的信息。
后续真实模型实验可以比较跨轮引用任务与窗口淘汰后的行为。

本阶段已通过上述新增实验，以及 context_builder、conversation_projection、
trajectory、reasoning_continuity、agent_loop、multiple_tools、logging、
schema_validation、tool_executor 九组既有实验。agent/memory/experiments 的
compileall 与 git diff --check 也通过。本次未重新运行真实 Ollama A/B。

## 下一阶段建议

Phase 7B 先做跨轮真实模型实验和预算策略设计。以长工具输出、超过窗口的旧事实、
用户更新信息等任务观察准确性与 prompt token，再确定是否需要 token budget 或摘要。
摘要需要明确来源、更新和错误传播策略后再实现。DecisionMemo 继续暂缓。
