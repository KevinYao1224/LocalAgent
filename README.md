# 本地 AI Agent 学习项目

这是一个以理解 AI Agent Runtime 架构为目标的 Python 项目。它从 Ollama
模型接口出发，逐步实现 Agent 循环、工具执行、会话与长期记忆，以及可观察的
运行轨迹。项目优先展示每一层的职责和依赖方向，而不是封装成大型通用框架。

## 从哪里开始

1. 从[agent_docs 导航](./agent_docs/README.md)选择当前状态、架构、实验或阶段记录。
2. 第一次阅读可从[当前架构导览](./agent_docs/当前架构导览.md)和[代码与实验阅读指南](./agent_docs/代码与实验阅读指南.md)开始。

`agent_docs/` 中按阶段保留设计边界、实现细节、实验结果和重要发现；
其中旧“下一步”以当前[路线图](./agent_docs/后续开发路线.md)为准。

## 目录概览

| 目录 | 职责 |
|---|---|
| `agent/` | Agent 会话、控制循环、运行状态、轨迹、上下文构建及可选计算器 Planner |
| `llm/` | provider-neutral 模型类型，以及 Ollama 聊天和 embedding 适配器 |
| `runtime/` | 工具能力门控、参数校验、执行和统一结果 |
| `tools/` | 工具定义、注册表和计算器示例 |
| `memory/` | 短期会话历史、SQLite 长期记忆和语义检索 |
| `observability/` | 结构化事件、人类可读日志、JSONL trace 和运行指标 |
| `experiments/` | 确定性离线实验及需要本地 Ollama 的集成实验 |
| `evaluation/` | Phase 13 声明式用例、运行后评分和可读汇总 |
| `agent_docs/` | 开发路线、阶段记录、架构导览和开发交接文档 |

## 环境准备

项目使用 Python、`httpx` 和 `jsonschema`。在 Linux / WSL 环境中，可在仓库根目录
创建并安装依赖：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

部分实验会连接本机 Ollama；下面的离线实验不需要 Ollama 服务：

```bash
.venv/bin/python experiments/agent_loop_experiment.py
```

它使用脚本化模型检查 AgentLoop 的控制流程。更多离线实验及每项实验观察重点见
[代码与实验阅读指南](./agent_docs/代码与实验阅读指南.md)。

## 与 Agent 对话

启动 Ollama 并准备默认模型 `qwen3.5:9b` 后，从项目根目录运行：

```bash
.venv/bin/python main.py
```

输入自然语言连续对话；`:help` 显示命令，`:clear` 清空当前短期会话记忆，`:exit`
或 Ctrl-D / Ctrl-C 退出。只提问一次可用 `--prompt "计算 (12 + 7) * 5"`。
`--model`、`--base-url` 和 `--max-turns` 可调整模型、地址及保留的完整历史轮数。
应用仅注册四则运算工具。

详细的 run/step、模型 thinking、工具参数和结果、错误及耗时会追加到
`logs/agent-debug.log`（自动创建目录，已被 git 忽略），终端默认只显示回答。
可以通过 `--log-file /tmp/opencode/my-agent.log` 指定文件，或添加 `--verbose`
同时在终端查看详细轨迹。调试日志含原始对话和 thinking；请自行管理日志文件。

还可以将事件追加到独立的 JSONL 文件（需先创建父目录）：

```bash
mkdir -p /tmp/opencode
.venv/bin/python main.py \
  --prompt "计算 12 + 7" \
  --trace-jsonl /tmp/opencode/agent-trace.jsonl
```

默认 trace 记录事件和指标，不保存对话原文。只有明确需要保存原文时才使用
`--trace-content`；它会把消息、thinking、工具参数和结果写入 trace。
不依赖 Ollama 的交互和文件日志实验：

```bash
.venv/bin/python experiments/interactive_cli_experiment.py
```

## 当前开发位置

Phase 9 计算器 Planner / Executor 已作为独立可选路径实现：
`.venv/bin/python experiments/planner_experiment.py` 可离线观察规划、验证和逐步执行；
在 Ollama 可用时运行 `.venv/bin/python experiments/planner_live_experiment.py --repeats 3`
与 Phase 13 基线做小样本对照。默认交互入口仍使用 `AgentLoop`；详情见路线文档。
