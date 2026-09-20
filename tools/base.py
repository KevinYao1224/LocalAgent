import inspect

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class ToolExecutionError(Exception):
    pass


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema Draft 2020-12
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
        except Exception as exc:
            raise ToolExecutionError(
                f"Tool '{self.name}' failed: {exc}"
            ) from exc
