# Phase 11B：固定命令的有界执行

## 决策与边界

Phase 11A 仅提供受约束的只读文件；本阶段增加**调用方显式注册的固定命令动作**，不开放模型可填写的命令字符串、shell、任意参数或 `main.py` 默认命令权限。此工具仍可运行有副作用的程序，绝不是沙箱：只应把调用方信任的可执行文件和**完整** argv 配置为授权动作，并先审查其潜在副作用及可访问资源。

```text
调用方 → FixedCommandPolicy(argv, cwd, environment, timeout_seconds, max_output_bytes)
       → create_fixed_command_tool(policy, name) → ToolRegistry
模型 → 已注册的动作名 + 空参数对象 → ToolExecutor 校验/可用性检查
     → handler(policy.run) → ToolResult / 既有事件与 RunMetrics
```

`runtime/commands.py` 构造时验证绝对可执行文件、可信绝对工作目录、固定 argv、显式环境变量及正数上限，并拷贝配置；`tools/commands.py` 创建的 schema 要求空参数对象，模型不能覆写命令、目录、环境或预算。`Popen` 用 `shell=False`、无 stdin、显式环境、独立进程组；selector 同时消费 stdout/stderr，**合计**最多 `max_output_bytes` 字节，超限或超时就终止该进程组并回收进程。非零退出、超限或超时作为已有 `execution_error`，恢复策略将其视为 terminal，不重试可能产生副作用的动作。成功时返回退出码、stdout 和 stderr；输出按 UTF-8 解码并替换非法字节，仍是**不可信数据**。

示例（只有明确同意授予该动作时才注册；路径须按本机实际情况填写）：

```python
from pathlib import Path
from runtime import FixedCommandPolicy, ToolExecutor
from tools.commands import create_fixed_command_tool
from tools.registry import ToolRegistry

policy = FixedCommandPolicy(
    argv=("/absolute/path/to/trusted/program", "--fixed-option"),
    cwd=Path("/absolute/trusted/working-directory"),
    environment={},
    timeout_seconds=2,
    max_output_bytes=1024,
)
registry = ToolRegistry()
registry.register(create_fixed_command_tool(policy, "approved_action"))
executor = ToolExecutor(registry)
```

## 离线可读实验

在仓库根目录运行：

```bash
.venv/bin/python experiments/fixed_command_permission_experiment.py
```

实验只执行调用方固定的短 Python 片段：打印固定工作目录、显式环境和 stdin 状态；再检查模型注入参数、禁用 capability、超时、stdout/stderr 超限、非零退出、超时后同组子进程清理。最后借 `HumanReadableLogger` 展示 run/step、工具错误、恢复终止与 `corrections_used=0`，输出 `verdict=PASS`。它不是在真实模型或不可信命令上进行的安全证明。

## 重要发现与后续方案

- `cwd` 不限制进程对其他目录的访问；清空环境不撤销进程本身的 OS 权限；固定命令仍可访问网络、写文件、启动脱离进程组的子进程。受信任的可执行文件及其依赖/配置/目录、父目录必须由调用方保护，不得把模型可修改的程序或脚本当作固定授权动作。
- selector 约束**捕获的管道输出**，并不限制磁盘写入、CPU/内存、额外文件描述符、进程数或系统调用。超时是运行期间的协作式清理机制，不是应对不可中断内核操作的硬期限。退出后脱离进程组的子进程未受本策略控制。日志/模型可能接触命令输出，需自行审查敏感性。
- 未来若要引入模型参数化命令、可写文件或不可信代码，必须另行设计参数到资源的授权映射，以及 OS 级身份/容器/namespace/seccomp 等隔离和针对竞态、子进程、文件系统、网络的测试。不要将本工具的上限或 Phase 7D capability 误称为通用 sandbox。
