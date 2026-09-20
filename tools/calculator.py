from typing import Any

from tools.base import Tool


def add(a: float, b: float) -> float:
    """Return the sum of two numbers."""

    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract b from a."""

    return a - b


def multiply(a: float, b: float) -> float:
    """Return the product of two numbers."""

    return a * b


def divide(a: float, b: float) -> float:
    """Divide a by b."""

    if b == 0:
        raise ValueError("Cannot divide by zero.")

    return a / b


def _binary_number_parameters() -> dict[str, Any]:
    """Create an independent JSON Schema for a binary operation."""

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
