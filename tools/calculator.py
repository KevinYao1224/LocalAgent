from typing import Any

from tools.base import Tool


def add(a: float, b: float) -> float:
    """返回两个数的和。"""

    return a + b


def subtract(a: float, b: float) -> float:
    """用 a 减去 b。"""

    return a - b


def multiply(a: float, b: float) -> float:
    """返回两个数的乘积。"""

    return a * b


def divide(a: float, b: float) -> float:
    """用 a 除以 b。"""

    if b == 0:
        raise ValueError("Cannot divide by zero.")

    return a / b


def _binary_number_parameters() -> dict[str, Any]:
    """为二元运算创建一份独立的 JSON Schema。"""

    return {
        "type": "object",
        "properties": {
            "a": {
                "type": "number",
                "description": "The first number.",
            },
            "b": {
                "type": "number",
                "description": "The second number.",
            },
        },
        "required": ["a", "b"],
        "additionalProperties": False,
    }


add_tool = Tool(
    name="add",
    description="Add a and b and return their sum.",
    parameters=_binary_number_parameters(),
    handler=add,
)

subtract_tool = Tool(
    name="subtract",
    description="Subtract b from a and return a - b. Argument order matters.",
    parameters=_binary_number_parameters(),
    handler=subtract,
)

multiply_tool = Tool(
    name="multiply",
    description="Multiply a and b and return their product.",
    parameters=_binary_number_parameters(),
    handler=multiply,
)

divide_tool = Tool(
    name="divide",
    description="Divide a by b and return a / b. Argument order matters.",
    parameters=_binary_number_parameters(),
    handler=divide,
)


calculator_tools = (
    add_tool,
    subtract_tool,
    multiply_tool,
    divide_tool,
)
