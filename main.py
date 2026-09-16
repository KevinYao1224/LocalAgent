from llm.base import Message
from llm.ollama import OllamaClient

from tools.calculator import multiply_tool
from tools.registry import ToolRegistry


MODEL = "qwen3.5:9b"


def main():
    registry = ToolRegistry()

    registry.register(
        multiply_tool
    )

    messages = [
        Message(
            role="system",
            content=(
                "You are an assistant with tools. "
                "Use tools when appropriate. "
                "After receiving a tool result, "
                "use that result to answer the user."
            ),
        ),

        Message(
            role="user",
            content="What is 23 multiplied by 47?",
        ),
    ]

    with OllamaClient(
        model=MODEL,
    ) as llm:
        response = llm.chat(
            messages=messages,
            tools=registry.all(),
        )

        print("Model response:", response.content)

        messages.append(Message(
            role="assistant",
            content=response.content,
            tool_calls=response.tool_calls
        ))

        if not response.tool_calls:
            print(
                "Model did not request a tool."
            )
            return

        for call in response.tool_calls:
            print(
                "Tool requested:",
                call.name,
                call.arguments,
            )

            result = registry.execute(
                call.name,
                call.arguments,
            )

            print("Tool result:", result)

            messages.append(Message(
                role = "tool",
                tool_name = call.name,
                content = str(result),
            ))
        
        final_response = llm.chat(
            messages=messages,
            tools=registry.all(),
        )

        print("------")
        print(
            "Final answer:",
            final_response.content,
        )


if __name__ == "__main__":
    main()