"""无需 Ollama：.venv/bin/python experiments/interactive_cli_experiment.py。"""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.base import LLM, ModelResponse, ToolCall
from main import create_session, main as cli_main, run_interactive, run_turn
from observability import FileDebugLogger


class DemoLLM(LLM):
    """用输入历史驱动确定性回复，方便看见跨轮记忆与工具链。"""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def chat(self, messages, tools=None) -> ModelResponse:
        question = messages[-1].content
        if question == "计算 12 + 7":
            return ModelResponse(
                thinking="选择加法工具",
                tool_calls=[ToolCall(name="add", arguments={"a": 12, "b": 7})],
            )
        if messages[-1].role == "tool":
            return ModelResponse(content=f"结果是 {messages[-1].content}")
        if question == "记住 CEDAR":
            return ModelResponse(content="已记住")
        if question == "之前的代号？":
            found = any(message.content == "记住 CEDAR" for message in messages[:-1])
            return ModelResponse(content="CEDAR" if found else "不知道")
        if question == "模拟故障":
            raise RuntimeError("离线故障")
        if question == "一直空响应":
            return ModelResponse(thinking="还没准备好")
        raise AssertionError(f"不应发送给模型的输入：{question}")


def main() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "nested" / "debug.log"
        with FileDebugLogger(path) as logger:
            session = create_session(DemoLLM(), logger, max_turns=5)
            commands = iter([
                ":help", "记住 CEDAR", "之前的代号？", "计算 12 + 7",
                "模拟故障", "一直空响应", ":clear", "之前的代号？", ":exit",
            ])
            output = StringIO()
            with redirect_stdout(output):
                run_interactive(session, lambda _: next(commands))
            text = output.getvalue()
            assert "助手：CEDAR" in text
            assert "助手：结果是 19" in text
            assert "本轮失败：RuntimeError: 离线故障" in text
            assert "本轮未完成：max_steps（6 步）" in text
            assert "助手：不知道" in text
            assert session.memory.turn_count == 1  # 清空后仅成功提交最后一轮
            assert "再见！" in text

            with redirect_stdout(StringIO()):
                assert not run_turn(session, "模拟故障")
            assert session.memory.turn_count == 1

        log = path.read_text(encoding="utf-8")
        assert "Model thinking: 选择加法工具" in log
        assert 'Tool start: add({"a": 12, "b": 7})' in log
        assert "Tool result: add -> 19" in log
        assert "Model call failed: RuntimeError: 离线故障" in log
        assert "Stop reason: max_steps" in log
        assert "Run ID:" in log
        assert ":clear" not in log
        print("交互演示：跨轮记忆、工具、失败、步骤上限、清空和退出均通过。")
        print("日志演示：独立文件含完整轨迹，命令不进入模型。")
        print("文件片段（真实事件）：")
        for line in log.splitlines():
            if line.startswith(("Run ID:", "Tool result:", "Stop reason:")):
                print(f"  {line}")

        # 同一路径重新打开时追加，而不是覆盖先前的 run。
        with FileDebugLogger(path) as logger:
            with redirect_stdout(StringIO()):
                assert run_turn(create_session(DemoLLM(), logger, 5), "记住 CEDAR")
        assert path.read_text(encoding="utf-8").count("=== Agent run started ===") == 8
        print("再次打开日志：原有 run 未丢失，新 run 已追加。")

        eof_output = StringIO()
        with redirect_stdout(eof_output):
            run_interactive(session, lambda _: (_ for _ in ()).throw(EOFError()))
        assert "再见！" in eof_output.getvalue()
        print("EOF 正常退出。")

        # 运行真正的参数入口，替换网络适配器以避免依赖 Ollama。
        cli_log = Path(directory) / "cli.log"
        trace = Path(directory) / "trace.jsonl"
        with patch("main.OllamaClient", return_value=DemoLLM()):
            with redirect_stdout(StringIO()):
                code = cli_main([
                    "--prompt", "计算 12 + 7", "--log-file", str(cli_log),
                    "--trace-jsonl", str(trace),
                ])
        assert code == 0
        assert "Final content: 结果是 19" in cli_log.read_text(encoding="utf-8")
        events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        assert any(event["event"] == "AgentFinished" for event in events)
        print("单次 CLI 参数入口：文件日志与 JSONL 同时写入。")


if __name__ == "__main__":
    main()
