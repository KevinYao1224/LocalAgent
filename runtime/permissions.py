"""显式授权的本地文本读取；不是通用 OS 沙箱。"""

import os
import stat
import errno
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tools.base import PermissionDeniedError, ToolExecutionError


@dataclass(frozen=True, slots=True)
class ReadTextPolicy:
    """仅允许读取受信任目录下的普通 UTF-8 文件，按字节限制结果。

    root 是调用方授予的目录，不从模型输入推导。allowed_paths 可将授权
    缩小到精确相对文件名；None 保持整个 root 内的原有读取范围，空集合拒绝所有文件。
    路径逐段以目录 fd 打开，不跟随符号链接。
    """

    root: Path
    max_bytes: int = 4096
    allowed_paths: Iterable[str] | None = None

    def __post_init__(self) -> None:
        root = Path(self.root)
        if not root.is_absolute() or not root.is_dir() or root.is_symlink():
            raise ValueError("root must be an existing absolute directory, not a symlink.")
        if (
            isinstance(self.max_bytes, bool)
            or not isinstance(self.max_bytes, int)
            or self.max_bytes < 1
        ):
            raise ValueError("max_bytes must be a positive integer.")
        if self.allowed_paths is not None:
            if isinstance(self.allowed_paths, (str, bytes)):
                raise ValueError("allowed_paths must be an iterable of relative file paths.")
            try:
                configured_paths = tuple(self.allowed_paths)
            except TypeError as exc:
                raise ValueError("allowed_paths must be iterable.") from exc
            if any(not _is_relative_file_path(path) for path in configured_paths):
                raise ValueError("allowed_paths must contain only valid relative file paths.")
            # 拷贝成不可变集合；调用方之后修改原列表不应悄悄扩大授权。
            object.__setattr__(self, "allowed_paths", frozenset(configured_paths))
        object.__setattr__(self, "root", root)

    def read_text(self, path: str) -> str:
        """模型给出相对路径；越权返回明确拒绝，I/O 错误保持执行错误。"""

        if not _is_relative_file_path(path):
            raise PermissionDeniedError("Only normalized relative file paths are permitted.")
        parts = path.split("/")
        if self.allowed_paths is not None and path not in self.allowed_paths:
            raise PermissionDeniedError("File is not in the authorized path list.")

        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        opened: list[int] = []
        try:
            directory_fd = os.open(self.root, directory_flags)
            opened.append(directory_fd)
            for part in parts[:-1]:
                directory_fd = os.open(part, directory_flags, dir_fd=directory_fd)
                opened.append(directory_fd)
            file_fd = os.open(parts[-1], file_flags, dir_fd=directory_fd)
            opened.append(file_fd)
            if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                raise PermissionDeniedError("Only regular files are permitted.")
            chunks: list[bytes] = []
            remaining = self.max_bytes + 1
            while remaining:
                chunk = os.read(file_fd, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            if len(data) > self.max_bytes:
                raise PermissionDeniedError("File exceeds the configured byte limit.")
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ToolExecutionError("File is not valid UTF-8 text.") from exc
        except OSError as exc:
            # 不把本机绝对路径及异常细节送回模型；不存在的文件属于普通 I/O 失败。
            if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                raise PermissionDeniedError(
                    "Symbolic links and non-directory components are denied."
                ) from exc
            raise ToolExecutionError("Unable to read the requested file.") from exc
        finally:
            for fd in reversed(opened):
                os.close(fd)


def _is_relative_file_path(path: object) -> bool:
    """配置与模型输入使用同一规则；不接受可绕过精确匹配的路径写法。"""

    return (
        isinstance(path, str) and bool(path)
        and "\x00" not in path and "\\" not in path
        and not path.startswith("/")
        and all(part not in ("", ".", "..") for part in path.split("/"))
    )
