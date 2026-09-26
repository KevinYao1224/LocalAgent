# WSL2 开发环境（迁移记录与当前用法）

项目在 2026-09-21 从 Windows 复制到 WSL2 的 Linux 原生文件系统；当前工作区以仓库根目录为准，不要在 Windows 旧副本和 WSL 两处同时开发。`AGENTS.md` 大小写必须准确，Linux 文件系统区分大小写。

## 环境与实验

在 Linux/WSL 仓库根目录安装依赖；Windows `.venv` 不可直接复用：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python experiments/agent_loop_experiment.py
.venv/bin/python experiments/interactive_cli_experiment.py
```

若 Python 缺少 `venv`，先安装与本机 Python 对应的发行版软件包。更多按模块可读实验见[阅读指南](./代码与实验阅读指南.md)；阶段状态见[路线图](./后续开发路线.md)。密钥不要写入仓库，打开项目时确认编辑器/终端指向 WSL 的项目目录与 Linux `.venv`。

## 从 WSL 连接本机 Ollama

先确认 Ollama 正在运行并有相应模型，再测试 loopback：

```bash
curl --noproxy '*' http://127.0.0.1:11434/api/tags
.venv/bin/python main.py --prompt '计算 (12 + 7) * 5'
```

本机曾在 WSL Mirror 模式下通过 `127.0.0.1:11434` 访问 Windows Ollama；不同主机/网络模式不保证相同。如果失败，先检查服务监听和 WSL 网络设置，再用 `main.py --base-url` 指定实际可达地址；不要为排障把无认证服务开放到公网或局域网。2026-09-22 的运行样本见[集成测试记录](./2026-09-22%20Ollama%20WSL%20Mirror%20集成测试记录.md)，不代表今天的网络或模型状态。
