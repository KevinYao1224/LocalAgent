# Phase 12 后待开发清单与下一阶段建议

> 2026-09-24，根据《后续开发路线》《新会话开发交接》及 Phase 7C、8B、8C、12B
> 记录整理。这是**计划中的剩余工作**，不表示全部都是当前阶段的阻塞项；
> 阶段编号也不代表 9—11 已完成。已完成基线：1—8C（含 7D）、12A—12B。

## 尚未实现的独立阶段

| 阶段 | 要解决的问题 | 最小可验收增量 / 顺序关系 |
|---|---|---|
| Phase 9：Planner / Executor | 将多步任务显式分解、验证计划并交给现有工具循环执行 | 先做结构化计划、状态和步骤执行的最小闭环；不要直接开放文件或 shell 工具。若例子要执行此类工具，应先满足 Phase 11 的权限前提。 |
| Phase 10：Reflection / Error Recovery | 判断哪些失败可恢复、何时停止 | 先做 Runtime 控制的错误分类、有限重试、次数和终止条件；再评估模型 Critic/Reflection 的实际收益。规划稳定后再推进。 |
| Phase 11：PermissionPolicy / Sandbox | 控制潜在副作用工具的执行权和资源边界 | 已有 7D capability gate，但尚无路径/命令权限、timeout、环境与输出限制；必须先于 filesystem/shell/Python/HTTP/browser 等工具落地。 |
| Phase 13：Evaluation | 将成功率、工具选择、参数准确率、空/重复调用、成本和延迟按 case 汇总 | 先建立离线、确定性的小型 cases 与 runner；之后再引入 Ollama 多次运行和对比报告。与 Phase 12 事件/指标衔接。 |
| Phase 14：Parallel Tool Calling | 同一模型响应的独立工具并发执行并按原顺序提交观察 | 需要判断独立性、保持结果关联和故障语义；有依赖的 A→B→C 仍必须顺序执行。 |
| Phase 15：Multi-Agent | 在成熟单 Agent Runtime 上进行 supervisor/worker 编排 | 依赖前面的状态、策略、trace、evaluation 和权限能力，不应现在引入。 |

## 已实现模块的后续改进（不等于上述阶段未完成）

1. **上下文与短期记忆（Phase 6/7）**：统一 prompt 的预算核算（不只历史与检索 JSON
   的字符数，还需 system、当前问题、工具 schema 等）；有 provider token 估算依据后
   再做 token budget。评估完整轮次摘要及信息损失。`DecisionMemo` 仍暂缓，不开启
   默认 reasoning replay；必要时在更难任务上作 A/B 证据比较。
2. **长期记忆（Phase 8）**：研究带来源和父记录关联的长文分段；显式的读写/自动写入
   策略及其评估；让会话级检索能表达 metadata 过滤，同时保持 namespace 隔离；
   对相似干扰项、缺失事实、预算排除后的误归因做可重复评测。数据量扩大后再测量
   SQLite 全量向量扫描和 Python metadata 过滤是否需要索引/批处理。跨用户共享
   不作为默认功能。
3. **可观测性增量（在 Phase 12 基线之上）**：当前 JSONL 记录模型、工具、结束
   指标，但 Phase 7C/8B 的历史与检索候选/选中/排除数量、来源 ID 仍主要在
   `AgentSession` 可读，尚未关联到 JSONL 的 run；需要时可增加 session 事件。
   同一步内多个同名工具调用目前共用 step ID，后续做并发时需独立 call ID。
   JSONL 不是可恢复 checkpoint，不能把它当成状态持久化。
4. **工程化与设计记录**：随着新增 policy/planner/provider，再按需补架构决策记录
   和有意义的边界实验；不提前建立空目录或引入大型框架。

## 推荐下一阶段：Phase 13A（离线评测基线）

**理由**：Phase 12 已提供 run/step、失败和 token/耗时的事实来源，但尚不能回答
“改动是否让 Agent 更好”。8B 的真实预算实验甚至出现“只看上下文选择通过，答案
却把可见项目 code 错归因给被排除归档”的案例；此时直接叠加 Planner 或自动写入
很难区分新功能收益与回归。评测先行能为 Phase 9/10/11 的改动提供可重复基线。

建议把 13A 限定为：

```text
现有 ScriptedLLM / ToolExecutor / AgentLoop
    → 少量声明式 case（计算、工具选择、参数错误/未知工具、无需工具、max_steps）
    → 用最终答案、工具调用及参数、stop_reason 判定结果
    → 输出人类可读的逐例结果与汇总（成功率/步骤/调用/错误）
    → 将 Phase 12 已有 token/耗时当作辅助指标（缺失值保留 unknown）
```

首批离线 case 证明 runner 的判定和汇总逻辑，不以预设脚本结果冒充真实模型能力；
随后 Phase 13B 对 Ollama 跑多次、保留 run ID 并报告波动，特别加入 8B 的缺失事实
误归因案例。可选 JSONL 保存独立 run 的事件，评价结果不要混入 Agent 的决策逻辑。

**另一合理选项**：如果优先按编号补齐，就先做 Phase 9A 最小结构化 Planner
（仅现有计算工具，不接文件/shell），再用 Phase 13 的评测度量它。但当前推荐
先 13A、再 9A：评测小步投入低，能够为之后的每次架构改动提供对照。

## CLI 用户界面安排

当前 `main.py` 接受 `--scenario` 等参数，实际运行预置的 Memory 实验，**尚不是**
允许用户任意输入任务的通用 CLI。通用 CLI v1 无需等待 Phase 9 Planner、Phase 11
完整 Sandbox 或 Phase 15 Multi-Agent；它可以作为 Phase 13A 后的一个独立小阶段
（若优先考虑可用性，也可以把这一小阶段提前到 13A 之前）。

最小可验收目标：

```text
python -m ... 或 python cli.py [--model ...] [--base-url ...]
    → 用户连续输入自然语言任务
    → 同一进程内 AgentSession 保留短期会话轮次
    → 每轮显示答案与必要的错误/停止原因
    → :help / :clear / :exit；Ctrl-C / EOF 可正常退出
    → --verbose 打开 HumanReadableLogger
    → 可选 --trace-jsonl，沿用 Phase 12 JSONL 指标
```

首版复用 `AgentLoop`、`AgentSession`、`OllamaClient` 和现有 calculator 工具；默认不
自动写入长期记忆、不执行文件或 shell。非交互的单次 `--prompt` 可作为同一 CLI
的简单脚本用法。需要通过模拟 LLM 的离线交互实验和一次真实 Ollama 手动运行，
检查连续轮次、退出和失败处理。用户若优先需要可操作界面，可直接以“CLI v1”
作为下一阶段，不必等路线表上的 Planner 或 Multi-Agent。
