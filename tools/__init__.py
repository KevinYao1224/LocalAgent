"""Public interfaces and built-in tools for the tools package."""

from tools.base import Tool, ToolExecutionError
from tools.calculator import multiply, multiply_tool
from tools.registry import ToolNotFoundError, ToolRegistry

__all__ = [
    "Tool",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolRegistry",
    "multiply",
    "multiply_tool",
]
