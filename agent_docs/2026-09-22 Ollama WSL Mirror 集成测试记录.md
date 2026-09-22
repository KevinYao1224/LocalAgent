# 2026-09-22：Ollama 与 WSL Mirror 集成测试

## 1. 测试背景

此前项目位于 WSL2 的 NAT 网络模式时，Windows 上的 Ollama 虽然已经启动，且
Windows 本机 `127.0.0.1:11434` 可以访问，但 WSL 中的以下地址均不可用：

```text
127.0.0.1:11434
localhost:11434
172.20.144.1:11434
```

Windows `netstat` 显示当时 Ollama 只监听 `127.0.0.1:11434`。因此之前曾经：

1. 在 Windows 本机直接调用 `/api/chat`，确认服务和模型本身正常。
2. 在 Windows Python 临时依赖环境中运行仓库的 `main.py`，绕过 WSL 网络完成了一次
   端到端验证。

这两项验证不是最终目标，因为项目的主工作区和 Linux `.venv` 都在 WSL 中。

之后将 WSL 网络切换为 **Mirror** 模式，使 WSL 可以通过回环地址访问 Windows
服务。本记录补齐切换后的正式验证。

## 2. Ollama 服务检查

在 WSL 中执行：

```bash
curl --noproxy '*' --fail http://127.0.0.1:11434/api/version
curl --noproxy '*' --fail http://127.0.0.1:11434/api/tags
```

实测结果：

```text
Ollama version: 0.34.2
Model: qwen3.5:9b
Parameter size: 9.0B
Quantization: Q4_K_M
Context length: 262144
Capabilities: tools, thinking, completion
Digest: a450a8f34d854a7fdc63581eddda0ebba1434f2f278821c1cab386eee7bcb9bd
```

因此当前 `main.py` 中的：

```python
MODEL = "qwen3.5:9b"
```

与 Ollama 实际安装的模型标签一致。

## 3. WSL 主工作区端到端测试

运行：

```bash
.venv/bin/python main.py
```

结果：

```text
Step 1: add(a=12, b=7)       -> 19
Step 2: multiply(a=19, b=5)  -> 95
Step 3: final answer          -> ... equals 95.
Stop reason: completed
Steps: 3
```

这证明 Linux `.venv` 中的项目代码已经通过真实 `OllamaClient` 接入
`qwen3.5:9b`，并完整复现了：

```text
模型决策 → 工具调用 → 工具结果回传 → 下一次模型决策 → 最终回答
```

同时观察到模型返回了 `thinking`。适配器将它保存到 `ModelResponse.thinking`，
但没有把它误当成最终答案；最终完成判断仍使用非空 `content`。

## 4. Reasoning replay A/B 测试

运行：

```bash
.venv/bin/python experiments/reasoning_replay_ab_experiment.py --runs 3
```

本次在 WSL Mirror 模式中实际运行的结果如下：

| policy | success | avg steps | avg empty | avg repeated | avg prompt | avg completion | avg latency | replays |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 100% | 3.00 | 0 | 0 | 2071.00 | 218.67 | 3.58s | 0 |
| latest_pending | 100% | 3.00 | 0 | 0 | 2071.00 | 209.33 | 3.33s | 6 |

两种策略都完成了 3/3 次任务，且没有空响应或重复工具调用。
`latest_pending` 每次运行实际回放两次，说明 reasoning replay 传输路径确实执行。

该样本仍然只是简单算术任务，不能据此证明 replay 对复杂任务有收益；默认策略
继续保持 `ReasoningReplayPolicy.OFF`。

## 5. 结论与后续

当前已确认：

```text
WSL Mirror → 127.0.0.1:11434       ✅
Ollama /api/version 和 /api/tags    ✅
qwen3.5:9b 模型标签和能力          ✅
OllamaClient 普通响应解析           ✅
thinking 捕获                      ✅
tool_calls 解析                    ✅
ToolExecutor + calculator           ✅
AgentLoop 多步调用                  ✅
最终回答和 completed 状态            ✅
ContextBuilder reasoning replay     ✅
```

此前的网络问题不是项目适配器或模型安装问题，而是 WSL NAT 模式下 Ollama 只监听
Windows 回环地址导致的连通性问题。以后在本机 WSL 环境中应优先使用 Mirror 模式
和项目默认的 `http://127.0.0.1:11434`。

Phase 7B 仍可按原计划继续：使用真实 `AgentSession` 做跨轮事实引用、事实更新、
窗口淘汰和长工具结果实验，再决定是否需要 token budget 或摘要策略。
