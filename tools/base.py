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
    parameters: dict[str, Any] # Use JSON Schema
    handler: Callable[..., Any]

    def execute(self, arguments: dict[str, Any]) -> Any:
        try:
            signature = inspect.signature(self.handler)

            signature.bind(**arguments)

            return self.handler(**arguments)

        except TypeError as exc:
            raise ToolExecutionError(
                f"Invalid arguments for tool '{self.name}': {exc}"
            ) from exc

        except Exception as exc:
            raise ToolExecutionError(
                f"Tool '{self.name}' failed: {exc}"
            ) from exc