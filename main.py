"""与本地 Ollama Agent 连续对话：python main.py（或 --prompt 单次提问）。"""

import argparse
from collections.abc import Callable
from pathlib import Path

from agent import AgentLoop, AgentSession
from llm.base import LLM
from llm.ollama import OllamaClient
from memory import ConversationMemory
from observability import (
    AgentLogger,
    CompositeLogger,
    FileDebugLogger,
    HumanReadableLogger,
    JsonlTraceLogger,
)
from runtime import ToolExecutor
from tools import ToolRegistry, calculator_tools


DEFAULT_LOG_FILE = Path("logs/agent-debug.log")
SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the available calculator tools when arithmetic "
    "is needed. Answer the user's question using the conversation and tool results."
)
HELP_TEXT = (
    ":help   显示命令\n"
    ":clear  清除本次会话的短期记忆\n"
    ":exit   退出（也可按 Ctrl-C 或 Ctrl-D）"
)


def create_session(llm: LLM, logger: AgentLogger, max_turns: int) -> AgentSession:
    """一个进程持有一个 Session；每次提问由 Session 创建独立的 Agent run。"""

    registry = ToolRegistry()
    for tool in calculator_tools:
        registry.register(tool)
    agent = AgentLoop(
        llm=llm,
        executor=ToolExecutor(registry),
        max_steps=6,
        logger=logger,
    )
    return AgentSession(
        agent=agent,
        system_prompt=SYSTEM_PROMPT,
        memory=ConversationMemory(max_turns=max_turns),
    )


def run_turn(session: AgentSession, prompt: str) -> bool:
    """仅将完成的回答展示给用户；异常和步骤耗尽均视为本轮失败。"""

    try:
        result = session.run(prompt)
    except Exception as exc:
        print(f"本轮失败：{type(exc).__name__}: {exc}")
        return False
    if not result.completed:
        print(f"本轮未完成：{result.stop_reason.value}（{result.steps} 步）。")
        return False
    print(f"助手：{result.response.content.strip()}")
    return True


def run_interactive(
    session: AgentSession,
    read_input: Callable[[str], str] = input,
) -> None:
    """命令在应用层处理，不发送给模型，也不进入会话记忆。"""

    print("Agent 已就绪。输入 :help 查看命令。")
    while True:
        try:
            text = read_input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            return

        command = text.lower()
        if command == ":exit":
            print("再见！")
            return
        if command == ":help":
            print(HELP_TEXT)
        elif command == ":clear":
            session.memory.clear()
            print("短期会话记忆已清除。")
        elif command.startswith(":"):
            print("未知命令，输入 :help 查看命令。")
        elif text:
            try:
                run_turn(session, text)
            except KeyboardInterrupt:
                print("\n再见！")
                return


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地 Ollama Agent 交互入口")
    parser.add_argument("--model", default="qwen3.5:9b", help="Ollama 聊天模型")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--max-turns", type=int, default=5, help="保留的完整历史轮数")
    parser.add_argument("--prompt", help="单次提问后退出")
    parser.add_argument(
        "--log-file", type=Path, default=DEFAULT_LOG_FILE,
        help="追加完整可读调试轨迹的文件",
    )
    parser.add_argument("--verbose", action="store_true", help="也在终端展示完整轨迹")
    parser.add_argument(
        "--trace-jsonl", type=Path, help="可选 JSONL 事件文件（父目录需存在）",
    )
    parser.add_argument("--trace-content", action="store_true", help="在 JSONL 中也记录原文")
    args = parser.parse_args(argv)
    if args.max_turns < 1:
        parser.error("--max-turns must be a positive integer")
    if args.trace_content and args.trace_jsonl is None:
        parser.error("--trace-content requires --trace-jsonl")
    if args.prompt is not None and not args.prompt.strip():
        parser.error("--prompt must not be empty")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with FileDebugLogger(args.log_file) as file_logger:
        loggers: list[AgentLogger] = [file_logger]
        if args.verbose:
            loggers.append(HumanReadableLogger(max_text_chars=500))
        if args.trace_jsonl is not None:
            loggers.append(JsonlTraceLogger(
                args.trace_jsonl, include_content=args.trace_content,
            ))
        logger = CompositeLogger(*loggers)

        with OllamaClient(model=args.model, base_url=args.base_url, timeout=180.0) as llm:
            session = create_session(llm, logger, args.max_turns)
            print(f"模型：{args.model} | 调试日志：{args.log_file}")
            if args.prompt is not None:
                return 0 if run_turn(session, args.prompt) else 1
            run_interactive(session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
