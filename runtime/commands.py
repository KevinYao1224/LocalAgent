"""显式授权的固定命令执行；限制资源，但不是操作系统沙箱。"""

import os
import selectors
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from types import MappingProxyType
from typing import Mapping

from tools.base import ToolExecutionError


@dataclass(frozen=True, slots=True)
class FixedCommandPolicy:
    """调用方固定完整 argv；模型不能添加参数或选择可执行文件。"""

    argv: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float = 3.0
    max_output_bytes: int = 4096

    def __post_init__(self) -> None:
        argv = tuple(self.argv)
        if not argv or any(
            not isinstance(arg, str) or "\x00" in arg for arg in argv
        ):
            raise ValueError("argv must contain non-null-terminated strings.")
        executable = Path(argv[0])
        if (not executable.is_absolute() or not executable.is_file()
                or executable.is_symlink() or not os.access(executable, os.X_OK)):
            raise ValueError("argv[0] must be an absolute executable file, not a symlink.")
        cwd = Path(self.cwd)
        if not cwd.is_absolute() or not cwd.is_dir() or cwd.is_symlink():
            raise ValueError("cwd must be an existing absolute directory, not a symlink.")
        env = dict(self.environment)
        if any(
            not isinstance(key, str) or not key or "=" in key or "\x00" in key
            or not isinstance(value, str) or "\x00" in value
            for key, value in env.items()
        ):
            raise ValueError("environment must contain valid string keys and values.")
        if (isinstance(self.timeout_seconds, bool)
                or not isinstance(self.timeout_seconds, (int, float))
                or not 0 < self.timeout_seconds < float("inf")):
            raise ValueError("timeout_seconds must be finite and positive.")
        if (isinstance(self.max_output_bytes, bool)
                or not isinstance(self.max_output_bytes, int)
                or self.max_output_bytes < 1):
            raise ValueError("max_output_bytes must be a positive integer.")
        object.__setattr__(self, "argv", argv)
        object.__setattr__(self, "cwd", cwd)
        object.__setattr__(self, "environment", MappingProxyType(env))

    def run(self) -> dict[str, str | int]:
        """只执行调用方配置的动作；stdout 与 stderr 共用字节上限。"""

        try:
            process = subprocess.Popen(
                self.argv,
                cwd=self.cwd,
                env=dict(self.environment),  # 不继承调用进程的环境变量
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                start_new_session=True,
                close_fds=True,
            )
        except OSError as exc:
            raise ToolExecutionError("Unable to start the authorized command.") from exc

        deadline = monotonic() + self.timeout_seconds
        output: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
        total = 0
        try:
            with selectors.DefaultSelector() as selector:
                assert process.stdout is not None and process.stderr is not None
                selector.register(process.stdout, selectors.EVENT_READ, "stdout")
                selector.register(process.stderr, selectors.EVENT_READ, "stderr")
                while selector.get_map():
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        raise ToolExecutionError("Command timed out.")
                    ready = selector.select(remaining)
                    if not ready:
                        raise ToolExecutionError("Command timed out.")
                    for key, _ in ready:
                        # 最多多读一个字节以识别超限，不能无限缓存子进程输出。
                        chunk = os.read(key.fd, self.max_output_bytes - total + 1)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        total += len(chunk)
                        if total > self.max_output_bytes:
                            raise ToolExecutionError("Command output exceeds the byte limit.")
                        output[key.data].extend(chunk)
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise ToolExecutionError("Command timed out.")
            try:
                code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                raise ToolExecutionError("Command timed out.") from exc
            if code != 0:
                raise ToolExecutionError(f"Command exited with status {code}.")
            return {
                "exit_code": code,
                "stdout": output["stdout"].decode("utf-8", errors="replace"),
                "stderr": output["stderr"].decode("utf-8", errors="replace"),
            }
        except BaseException:
            # 包括超时、输出超限及取消：清理同一进程组中的子进程。
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            raise
        finally:
            process.wait()
            assert process.stdout is not None and process.stderr is not None
            process.stdout.close()
            process.stderr.close()
