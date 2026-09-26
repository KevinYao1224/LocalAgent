# agent_docs 导航

先读本页，再按任务进入对应文档；不要把按日期撰写的阶段记录当作当前操作说明。

| 需要了解 | 权威入口 |
|---|---|
| 当前状态、下一步、开发约束 | [后续开发路线](./后续开发路线.md) |
| 一次 run 的模块与数据边界 | [当前架构导览](./当前架构导览.md) |
| 源码阅读顺序、离线/在线实验命令 | [代码与实验阅读指南](./代码与实验阅读指南.md) |
| 开新任务的最短检查单 | [新会话开发交接](./新会话开发交接.md) |
| WSL 环境或 Ollama 连接问题 | [WSL2 迁移与开发指南](./WSL2%20迁移与开发指南.md) |

## 历史实现记录（按主题查阅）

- 控制与工具：[Phase 2](./Phase%202%20-%20AgentLoop%20实现记录.md) → [3](./Phase%203%20-%20Runtime%20实现记录.md) → [4](./Phase%204%20-%20Schema%20Validation%20实现记录.md) → [5](./Phase%205%20-%20Multiple%20Tools%20实现记录.md)；[7D capability gate](./Phase%207D%20-%20Tool%20Capability%20Enforcement%20实现记录.md)。
- 轨迹与推理：[5.5 日志](./Phase%205.5%20-%20Human%20Readable%20Logging%20实现记录.md) → [6A 状态](./Phase%206A%20-%20Agent%20State%20与%20Trajectory%20实现记录.md) → [6B reasoning](./Phase%206B%20-%20Reasoning%20Continuity%20实现记录.md) → [6C context](./Phase%206C%20-%20Context%20Builder%20实现记录.md)。
- 记忆：[7A 短期](./Phase%207A%20-%20Conversation%20Memory%20实现记录.md) → [7B 在线观察](./Phase%207B%20-%20跨轮真实实验与预算策略.md) → [7C 历史预算](./Phase%207C%20-%20History%20Character%20Budget%20实现记录.md)；[8A 文本](./Phase%208A%20-%20SQLite%20长期记忆实现记录.md) → [8B 检索预算](./Phase%208B%20-%20长期记忆检索预算实现记录.md) → [8C 语义](./Phase%208C%20-%20SQLite%20语义检索实现记录.md)。
- 可观测性：[12A 标识和计时](./Phase%2012A%20-%20Run%20Step%20标识与计时实现记录.md) → [12B JSONL](./Phase%2012B%20-%20JSONL%20Trace%20与运行指标实现记录.md) → [交互入口与文件日志](./交互入口与文件调试日志实现记录.md)。
- 任务评测：[Phase 13 离线与在线 Evaluation](./Phase%2013%20-%20Evaluation%20实现记录.md)。
- 规划执行：[Phase 9 Calculator Planner / Executor](./Phase%209%20-%20Calculator%20Planner%20Executor%20实现记录.md)。
- 带日期的环境/模型实验：[2026-09-22 Ollama 集成记录](./2026-09-22%20Ollama%20WSL%20Mirror%20集成测试记录.md)。其延迟、token 和成功率仅反映当时的小样本。

阶段记录保留当时的设计推理与测试结果；其中“下一阶段”不代表现状。实际入口以仓库根目录的 `README.md`、本目录的阅读指南和当前代码为准。

## 文档核验（无需 Ollama）

从仓库根目录运行：

```bash
.venv/bin/python main.py --help
.venv/bin/python experiments/conversation_memory_live_experiment.py --help
.venv/bin/python experiments/interactive_cli_experiment.py
git diff --check
```

对照 `main.py --help` 确认交互入口支持 `--prompt` 而不再支持 `--scenario`；
第二条显示独立在线实验的参数但不连接模型；第三条以可读轨迹验证交互、日志和 JSONL；
最后一条检查文档空白格式。需要真实模型的验证另见[阅读指南](./代码与实验阅读指南.md)。
