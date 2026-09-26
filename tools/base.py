import inspect

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class ToolExecutionError(Exception):
    pass


class PermissionDeniedError(ToolExecutionError):
    """工具自身施加的资源授权拒绝；允许 Runtime 单独分类。"""


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema Draft 2020-12 参数定义
    handler: Callable[..., Any]

    def execute(self, arguments: dict[str, Any]) -> Any:
        try:
            signature = inspect.signature(self.handler)
            signature.bind(**arguments)
        except TypeError as exc:
            raise ToolExecutionError(
                f"Invalid arguments for tool '{self.name}': {exc}"
            ) from exc

        try:
            return self.handler(**arguments)
        except PermissionDeniedError:
            # 只放行权限拒绝；其他 handler 异常仍保留原有包装行为。
            raise
        except Exception as exc:
            raise ToolExecutionError(
                f"Tool '{self.name}' failed: {exc}"
            ) from exc
