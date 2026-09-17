from agent.loop import AgentLoop, StopReason
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
        agent = AgentLoop(
            llm=llm,
            registry=registry,
            max_steps=5,
        )
        result = agent.run(messages)

        if result.stop_reason is StopReason.MAX_STEPS:
            print(f"Agent stopped after {result.steps} steps.")
        else:
            print("Final answer:", result.response.content)


if __name__ == "__main__":
    main()
