from llm.base import Message
from llm.ollama import OllamaClient


MODEL = "qwen3.5:9b"


def main():
    messages = [
        Message(
            role="system",
            content="You are a concise local AI assistant.",
        ),
        Message(
            role="user",
            content="计算 23 * 47，只给出结果。",
        ),
    ]

    with OllamaClient(model=MODEL) as llm:
        response = llm.chat(messages)

    print("content:", response.content)
    print("prompt tokens:", response.prompt_tokens)
    print("completion tokens:", response.completion_tokens)
    print("tool calls:", response.tool_calls)


if __name__ == "__main__":
    main()