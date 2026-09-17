from typing import Any

from tools.base import Tool


class ToolNotFoundError(Exception):
    pass


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered."
            )
        
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(
                f"Tool '{name}' is not registered."
            ) from exc

    def execute(
        self,
        name: str,
        arguments: dict[str, Any]
    ) -> Any:
        tool = self.get(name)

        return tool.execute(arguments)
    
    def all(self) -> list[Tool]:
        return list(self._tools.values())
        
