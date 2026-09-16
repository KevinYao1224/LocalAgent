from tools.base import Tool


def multiply(a: float, b: float) -> float:
    return a * b


multiply_tool = Tool(
    name="multiply",

    description=(
        "Multiply two numbers and return the result. "
        "Use this tool when exact multiplication is needed."
    ),

    parameters={
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
    },

    handler=multiply,
)