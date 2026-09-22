# WSL2 迁移与开发指南

## 1. 本次迁移结论

迁移日期：2026-09-21。

项目从 Windows 工作目录：

```text
D:\kyee\projects\private\Local Agent
```

复制到 WSL2 的 Linux 原生文件系统：

```text
/home/kyee/code/local-agent
```

使用的发行版：

```text
Ubuntu-26.04 (WSL 2)
```

迁移采用“复制、验证、再切换”的方式。Windows 原目录保留为回退副本，不在
迁移阶段删除。复制包含 `.git`、已提交文件以及未提交工作树，因此本地领先远端
的提交和 Phase 7A 的未提交代码都不会因重新克隆而丢失。

以下内容不复制：

```text
.venv/
__pycache__/
*.pyc
```

它们包含 Windows 专用解释器、绝对路径或可重建缓存，不能在 Linux 中复用。
WSL 目标目录会重新创建 `.venv` 并安装 `requirements.txt`。

## 2. 为什么项目放在 `/home/kyee` 下

不要长期在 `/mnt/d/kyee/projects/...` 下运行 Python、Git、Codex 或 OpenCode。
Windows 挂载盘可以用来搬运文件，但大量小文件访问、权限、符号链接和文件监听
通常不如 Linux 原生文件系统稳定。

推荐数据流：

```text
Windows 原目录（只作回退）
        │
        │ 一次性复制
        ▼
/home/kyee/code/local-agent
        │
        ├── VS Code Remote - WSL
        ├── Codex（在 WSL 中运行）
        ├── OpenCode（在 WSL 中运行）
        └── Python .venv（Linux 版本）
```

迁移后不要同时修改 Windows 和 WSL 两份工作树，否则它们会立刻分叉。确认 WSL
版本工作正常后，把 `/home/kyee/code/local-agent` 视为主工作区；跨机器或回退
应使用 Git commit/branch，而不是反复双向复制目录。

## 3. Linux 文件名兼容性修正

原仓库使用 `AGENTs.md`。Windows 文件系统通常不区分大小写，但 WSL 的 Linux
文件系统会区分大小写，而 Codex 和 OpenCode 约定的项目规则文件名是：

```text
AGENTS.md
```

迁移前已将文件规范为 `AGENTS.md`，以保证两种 Agent 都能在 WSL 中自动发现
项目说明。以后不要改回混合大小写形式。

## 4. Python 环境

本次迁移已经创建并验证：

```text
Python 3.14.4
pip 26.2.1
/home/kyee/code/local-agent/.venv
```

每次新开 WSL 终端：

```bash
cd ~/code/local-agent
source .venv/bin/activate
python --version
```

首次创建环境：

```bash
cd ~/code/local-agent
sudo apt update
sudo apt install python3.14-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Ubuntu 初始环境没有 `python3.14-venv`，自动迁移无法代输 `sudo` 密码，因此
本次实际使用 `python3 -m venv --without-pip`，再从 PyPA 官方
`https://bootstrap.pypa.io/get-pip.py` 只向项目 `.venv` 引导安装 pip。当前环境
已经可用，不需要为了继续开发立刻安装系统包。以后若需要删除并重建 `.venv`，
推荐先执行上面的 apt 安装命令。

Windows 的 `.venv\Scripts\python.exe` 命令在 WSL 中必须改成：

```bash
.venv/bin/python
```

或激活环境后直接使用：

```bash
python
```

离线回归实验：

```bash
python experiments/conversation_memory_experiment.py
python experiments/reasoning_continuity_experiment.py
python experiments/context_builder_experiment.py
python experiments/conversation_projection_experiment.py
python experiments/trajectory_experiment.py
python experiments/logging_experiment.py
python experiments/multiple_tools_experiment.py
python experiments/agent_loop_experiment.py
python experiments/schema_validation_experiment.py
python experiments/tool_executor_experiment.py
python -m compileall -q agent memory llm runtime tools observability main.py experiments
git diff --check
```

真实模型实验仍需要 Ollama 可访问，见第 8 节。

## 5. VS Code 工作方式

Windows 侧安装：

1. Visual Studio Code。
2. Microsoft 的 `WSL` 扩展。
3. OpenAI 的 Codex 扩展（扩展 ID：`openai.chatgpt`）。

从 WSL 终端打开项目：

```bash
cd ~/code/local-agent
code .
```

正确打开后，VS Code 左下角应显示 `WSL: Ubuntu-26.04`，集成终端中的路径应为
`/home/kyee/code/local-agent`，而不是 `D:\...` 或 `/mnt/d/...`。

在 VS Code 的用户设置 JSON 中建议启用：

```json
{
  "chatgpt.runCodexInWindowsSubsystemForLinux": true,
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python"
}
```

第一项来自 OpenAI 官方 Codex WSL 指南，会让 Windows 上的 Codex IDE 扩展在
可用时把 Agent 放进 WSL2；第二项让 Python 扩展使用项目的 Linux 虚拟环境。

常用检查：

```bash
echo "$WSL_DISTRO_NAME"
pwd
which python
git status --short --branch
```

## 6. Codex 使用方式

### VS Code 扩展

在 WSL Remote 窗口中打开 Codex 侧栏。若图标不可见，打开命令面板并运行：

```text
Codex: Open Codex Sidebar
```

开始任务前先让 Codex 读取 `AGENTS.md` 和 `agent_docs/新会话开发交接.md`。功能
开发继续遵循“每阶段实现、实验、记录”的项目约定。

本次迁移发生在一个已经绑定 Windows 原目录的 Codex Desktop 任务中；复制目录
不会自动改变现有任务的工作区。完成本任务后，应从 WSL 项目新建后续任务，或在
项目选择器中使用：

```text
\\wsl.localhost\Ubuntu-26.04\home\kyee\code\local-agent
```

不要继续让旧任务修改 `D:\kyee\projects\private\Local Agent`，否则两份工作树会
分叉。

### Codex CLI

必须在 WSL 内安装 Linux 版本，不要依赖 Windows `PATH` 中的可执行文件：

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
codex
```

安装后检查：

```bash
which codex
codex --version
```

`which codex` 应指向 `/home/kyee/...` 下的 Linux 路径，而不是 `/mnt/c/...`。

## 7. OpenCode 使用方式

WSL 中已经安装 Linux 版本 OpenCode，安装目录为：

```text
/home/kyee/.opencode/bin/opencode
```

重新打开一个交互式 WSL shell 后检查：

```bash
which opencode
opencode --version
```

`which opencode` 应指向 `/home/kyee/.opencode/bin/opencode`，不应落到
`/mnt/c/Users/...`。进入项目后运行：

```bash
cd ~/code/local-agent
opencode
```

仓库已经有人工维护的 `AGENTS.md`，不要直接运行 `/init` 覆盖它。如果需要让
OpenCode 补充规则，应先备份或提交当前文件，再人工合并建议。

OpenCode 的 Linux 配置和会话默认位于：

```text
~/.local/share/opencode/
```

它和 Windows 侧 OpenCode 的配置是两套独立状态。API key 应通过 OpenCode 的
连接流程或环境变量配置，不要写入仓库。

## 8. 从 WSL 访问 Windows 上的 Ollama

先在 WSL 中测试 Windows Ollama 是否通过 localhost 转发可达：

```bash
curl http://localhost:11434/api/tags
```

若成功，项目默认的 `http://localhost:11434` 可以继续使用。若失败：

1. 确认 Windows Ollama 正在运行。
2. 检查 Ollama 是否允许 WSL 访问以及 Windows 防火墙规则。
3. 获取 Windows 主机地址后测试：

   ```bash
   ip route show | awk '/default/ {print $3}'
   curl http://<windows-host-ip>:11434/api/tags
   ```

不要为了临时连通而把 Ollama 无认证地暴露到局域网。若必须修改监听地址，应只
开放必要范围并同步检查防火墙。

迁移初期的 NAT 网络模式下，实测 `localhost:11434` 不可达；Windows Ollama
当时只监听 `127.0.0.1:11434`。2026-09-22 已将 WSL 网络切换为 Mirror 模式，
现在 WSL 中的以下检查已经成功：

```bash
curl --noproxy '*' http://127.0.0.1:11434/api/version
curl --noproxy '*' http://127.0.0.1:11434/api/tags
```

并已在 Linux `.venv` 中重新运行 `main.py` 和
`experiments/reasoning_replay_ab_experiment.py --runs 3`。详细结果见：
[Ollama WSL Mirror 集成测试记录](./2026-09-22%20Ollama%20WSL%20Mirror%20集成测试记录.md)。

## 9. 日常开发流程

```bash
wsl -d Ubuntu-26.04              # 此命令在 PowerShell/Windows Terminal 运行
cd ~/code/local-agent
source .venv/bin/activate
git status --short --branch
code .                            # VS Code 工作流
# 或 codex                        # Codex CLI 工作流
# 或 opencode                     # OpenCode TUI 工作流
```

每一阶段完成后：

1. 运行与本阶段相关的实验。
2. 运行全部离线回归、`compileall` 和 `git diff --check`。
3. 在 `agent_docs` 添加实现记录和重要发现。
4. 更新 `后续开发路线.md` 与 `新会话开发交接.md`。
5. 检查 diff 后再提交 Git checkpoint。

当前下一阶段仍是 Phase 7B：真实跨轮实验与预算策略设计。迁移本身不改变
`reasoning_replay=off` 的默认策略，也不提前实现摘要、向量检索或长期记忆。

## 10. 迁移验证记录

迁移完成时已确认：

```text
WSL 发行版              Ubuntu-26.04 / WSL 2
Git HEAD                3df0a2173c934df66f894318c49eb345bb25ce57
Linux 项目路径           /home/kyee/code/local-agent
Windows 访问路径         \\wsl.localhost\Ubuntu-26.04\home\kyee\code\local-agent
Windows .venv/缓存       未复制
Linux .venv             已创建，依赖已安装
VS Code WSL 扩展         已安装
OpenAI Codex 扩展        已安装
WSL 原生 Codex CLI       尚未安装
WSL 原生 OpenCode        已安装；/home/kyee/.opencode/bin/opencode
旧 OpenCode 数据         Linux/Windows 配置、缓存、状态、日志和项目快照已清理
仓库权限                 目录 755、普通文件 644，Git core.filemode=true
文本换行                 已统一为 LF，并加入 .gitattributes
Ollama localhost:11434   Mirror 模式下可达（2026-09-22 已验证）
```

以下检查在 WSL 中全部通过：

```text
10 组离线 experiments
python -m compileall
git diff --check
```

## 11. 官方参考

- OpenAI Codex WSL 指南：https://learn.chatgpt.com/docs/windows/wsl
- OpenAI Codex IDE 扩展：https://learn.chatgpt.com/docs/codex/ide
- Codex approvals 与 WSL2 sandbox：https://learn.chatgpt.com/docs/agent-approvals-security#os-level-sandbox
- VS Code Remote - WSL：https://code.visualstudio.com/docs/remote/wsl-tutorial
- OpenCode 入门：https://opencode.ai/docs/zh-cn/
- OpenCode Windows/WSL：https://opencode.ai/docs/zh-cn/windows-wsl
