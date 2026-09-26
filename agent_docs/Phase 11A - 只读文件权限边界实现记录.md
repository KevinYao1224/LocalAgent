# Phase 11A：只读文件权限边界

## 范围与调用链

这一阶段只允许调用方**显式**注册一个受约束的 `read_text_file` 工具；默认 `main.py` 仍仅注册计算器。没有开放写文件、shell、Python 执行、HTTP 或 browser。它是最小的本地文件读取权限练习，**不是进程级 sandbox 或通用 PermissionPolicy**。

```text
调用方选择可信的绝对目录及 max_bytes
  → ReadTextPolicy → create_read_text_tool(policy) → ToolRegistry
  → ToolExecutor (可用工具、schema) → Tool handler → policy.read_text(path)
  → ToolResult / ToolExecutionFinished (可读日志、既有 JSONL 和指标)
```

`runtime/permissions.py` 把策略与不可信的模型路径隔开：只接受非空相对路径、禁止 `.` / `..` / 空路径段及反斜杠；从授权目录开始逐段用目录 fd 和 `O_NOFOLLOW` 打开，不在检查后重新按路径打开目标；仅读取普通文件，按 `max_bytes + 1` 判断是否超出字节限制，再解码 UTF-8。目录 symlink、文件 symlink、管道等被拒绝。配置时拒绝不存在、相对或末段为 symlink 的 root。文件不存在或编码错误是执行失败；越权和大小超限返回单独的 `permission_denied`，它在已有 `RecoveryPolicy` 中按 terminal 处理，不由模型纠正、更不会重放 handler。`tools/filesystem.py` 工厂把固定策略绑定到工具，`runtime/executor.py` 将预期内的拒绝转换为 `ToolResult`；工具参数 schema 仍负责结构验证，不能替代路径授权。

示例（仅在调用方明确同意授予目录内容读取时）：

```python
from pathlib import Path
from runtime import ReadTextPolicy, ToolExecutor
from tools.filesystem import create_read_text_tool
from tools.registry import ToolRegistry

registry = ToolRegistry()
registry.register(create_read_text_tool(ReadTextPolicy(Path('/absolute/trusted/docs'), max_bytes=4096)))
executor = ToolExecutor(registry)
```

## 人类可读实验（无需 Ollama）

从仓库根目录运行：

```bash
.venv/bin/python experiments/read_text_permission_experiment.py
```

实验仅用临时目录：一项正常嵌套读取；父目录逃逸、绝对路径、符号链接、FIFO、超限文件均被拒绝；编码错误/不存在与权限拒绝分类不同；还检查 schema、capability 和不可恢复终止。末尾 `HumanReadableLogger` 展示真实 run/step 事件、权限错误和 `corrections_used=0`，最终 `verdict=PASS`。测试结果验证的是本工具的边界，不表示真实模型不会被误导。

## 重要限制 / 后续方案

- 只在 Linux/支持 `dir_fd`、`O_NOFOLLOW` 的平台使用；受信任的 root 及其父目录必须由调用方管理，不应把可被恶意本地进程改写的目录当作隔离边界。已有硬链接、挂载点和拥有进程权限的其他代码不受此策略隔离；需要抵抗恶意本地代码时应使用 OS 级沙箱/权限隔离。读取内容及工具调用路径可能出现在可读日志中，JSONL 原文仍由显式选项控制；授权前须考虑数据敏感度。
- 本阶段没有统一执行任意工具的权限策略，也没有输出 token 预算、超时、命令白名单或环境隔离。未来引入命令/网络/副作用工具前，单独定义允许动作与资源、工作目录/环境、deadline、输出限制及失败语义，并以隔离测试验证；不要用当前 capability gate 代替沙箱。对可变授权目录及不同平台还需更深入威胁模型和测试。
