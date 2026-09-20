from agent.loop import AgentLoop
from llm.base import Message
from llm.ollama import OllamaClient
from observability import HumanReadableLogger
from runtime.executor import ToolExecutor

from tools.calculator import calculator_tools
from tools.registry import ToolRegistry


MODEL = "qwen3.5:9b"


def main():
    registry = ToolRegistry()

    for tool in calculator_tools:
        registry.register(tool)
    executor = ToolExecutor(registry)

    messages = [
        Message(
            role="system",
            content=(
                "You are an assistant with tools. "
                "Use one arithmetic tool per step. "
                "Do not calculate intermediate values yourself. "
                "After receiving a tool result, decide whether another "
                "tool is needed before answering the user."
            ),
        ),

        Message(
            role="user",
            content=(
                "Calculate (12 + 7) multiplied by 5. "
                "Use a separate tool call for each arithmetic operation."
            ),
        ),
    ]

    with OllamaClient(
        model=MODEL,
    ) as llm:
        agent = AgentLoop(
            llm=llm,
            executor=executor,
            max_steps=6,
            logger=HumanReadableLogger(),
        )
        agent.run(messages)


if __name__ == "__main__":
    main()
