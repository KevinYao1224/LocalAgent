from agent.loop import AgentLoop, StopReason
from llm.base import Message
from llm.ollama import OllamaClient
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
        )
        result = agent.run(messages)

        for index, tool_result in enumerate(
            result.tool_results,
            start=1,
        ):
            print(
                f"Tool {index}: {tool_result.tool_name} -> "
                f"{tool_result.to_message_content()}"
            )

        if result.stop_reason is StopReason.MAX_STEPS:
            print(f"Agent stopped after {result.steps} steps.")
        else:
            print("Final answer:", result.response.content)


if __name__ == "__main__":
    main()
