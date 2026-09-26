# Phase 11C：精确文件资源授权

## 增量与层次

在 11A 的 `ReadTextPolicy` 增加可选的 `allowed_paths`：由调用方配置**精确的相对文件路径**。`None` 保持 11A 原有的整个 root 内读取范围；空集合拒绝所有文件。配置时使用和模型输入相同的路径语法规则，并拷贝为不可变集合，避免原列表之后的修改扩大授权。模型给出路径时，先校验语法，再比较精确名称；不匹配时在打开任何文件前返回 `permission_denied`。匹配后仍逐段用 fd 打开、禁止 symlink、检查普通文件、字节上限及 UTF-8。不存在的**已授权**文件属于 `execution_error`。

这仍是工具自身的资源策略，不是通用 PermissionPolicy：动作授权由注册工具及 `ToolExecutor` 当前 capability 共同控制，固定命令依旧只能使用调用方绑定的完整 argv。`main.py` 没有注册文件或命令工具；本增量没有新增写入、网络或执行不可信代码的能力。

```python
from pathlib import Path
from runtime import ReadTextPolicy
from tools.filesystem import create_read_text_tool

policy = ReadTextPolicy(
    Path('/absolute/trusted/docs'), max_bytes=4096,
    allowed_paths=['public/guide.txt', 'faq.txt'],
)
tool = create_read_text_tool(policy)  # 仅在调用方明确授权后注册到 ToolRegistry
```

## 离线可读实验

在仓库根目录运行（无需 Ollama）：

```bash
.venv/bin/python experiments/read_text_allowlist_experiment.py
.venv/bin/python experiments/read_text_permission_experiment.py
```

第一条用临时目录验证：准许的文件可读；同目录兄弟文件、目录内未授权路径和配置后追加的路径被拒绝；允许列表中的 symlink、超限文件仍被拒绝；允许列表中不存在的文件是执行错误；非法配置提前报错；空白名单无文件可读。最后的人类可读 run/step 轨迹显示权限拒绝和不进行模型纠正的终止，`verdict=PASS`。第二条回归验证未启用白名单时 11A 的行为。

## 发现与后续方案

- 精确的路径名匹配不是文件身份保证：硬链接、挂载、受信任目录被其他进程修改，以及有 OS 权限的进程都可能改变实际资源。root 及其父目录、列入白名单的目标和日志内容都须由调用方管理；对恶意本地进程仍需 OS 隔离。
- 这是针对只读文件的收紧选项，不自动缩减已有调用方的授权范围；敏感应用应显式提供最小白名单。未来引入模型可参数化的命令、写入或网络前，需要单独完成动作/资源映射、隔离与相应竞态测试；不要把白名单或 capability gate 当沙箱。
