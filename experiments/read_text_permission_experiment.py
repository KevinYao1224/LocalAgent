"""Phase 11A：从仓库根目录运行 .venv/bin/python experiments/read_text_permission_experiment.py。

仅使用临时目录和脚本模型；不连接 Ollama，也不会读取仓库文件。
"""

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import AgentLoop
from evaluation.cases import ScriptedLLM
from llm.base import Message, ModelResponse, ToolCall
from observability.logger import HumanReadableLogger
from runtime import ReadTextPolicy, RecoveryPolicy, ToolErrorType, ToolExecutor
from tools.filesystem import create_read_text_tool
from tools.registry import ToolRegistry


def main() -> None:
    with TemporaryDirectory() as temporary:
        base = Path(temporary)
        allowed = base / "allowed"
        allowed.mkdir()
        (allowed / "docs").mkdir()
        (allowed / "docs" / "note.txt").write_text("只读示例", encoding="utf-8")
        (allowed / "large.txt").write_text("x" * 17, encoding="utf-8")
        (allowed / "bad.txt").write_bytes(b"\xff")
        secret = base / "secret.txt"
        secret.write_text("NOT AUTHORIZED", encoding="utf-8")
        (allowed / "link.txt").symlink_to(secret)
        (allowed / "linked_dir").symlink_to(base, target_is_directory=True)
        os.mkfifo(allowed / "pipe")
        (base / "root_link").symlink_to(allowed, target_is_directory=True)
        for root, size in ((base / "root_link", 16), (allowed, 0)):
            try:
                ReadTextPolicy(root, max_bytes=size)
            except ValueError:
                pass
            else:
                raise AssertionError("Unsafe policy configuration was accepted")

        registry = ToolRegistry()
        registry.register(create_read_text_tool(ReadTextPolicy(allowed, max_bytes=16)))
        executor = ToolExecutor(registry)

        def check(path: str, expected: ToolErrorType | None) -> None:
            result = executor.execute(ToolCall("read_text_file", {"path": path}))
            assert result.error_type is expected, (path, result)
            if expected is None:
                assert result.value == "只读示例"
            else:
                assert "NOT AUTHORIZED" not in result.to_message_content()
            print(f"path={path!r} → {result.value if result.success else result.error_type.value}")

        check("docs/note.txt", None)
        for path in ("../secret.txt", "docs/../../secret.txt", str(secret),
                     "docs//note.txt", "./docs/note.txt", "link.txt",
                     "linked_dir/secret.txt", "pipe", "large.txt"):
            check(path, ToolErrorType.PERMISSION_DENIED)
        check("bad.txt", ToolErrorType.EXECUTION_ERROR)
        check("missing.txt", ToolErrorType.EXECUTION_ERROR)
        invalid = executor.execute(ToolCall("read_text_file", {}))
        assert invalid.error_type is ToolErrorType.VALIDATION_ERROR
        print("missing path → validation_error")

        executor.disable_tool("read_text_file")
        assert executor.available_tools() == []
        assert executor.execute(ToolCall("read_text_file", {"path": "docs/note.txt"})).error_type \
            is ToolErrorType.TOOL_NOT_AVAILABLE
        executor.enable_tool("read_text_file")

        # 可读事件显示拒绝与终止；权限失败不由模型纠正或自动重试。
        llm = ScriptedLLM((ModelResponse(tool_calls=[ToolCall(
            "read_text_file", {"path": "../secret.txt"},
        )]),))
        run = AgentLoop(llm, executor, max_steps=3,
                        recovery_policy=RecoveryPolicy(max_corrections=2),
                        logger=HumanReadableLogger()).run([
                            Message(role="user", content="Read outside the authorized root."),
                        ])
        assert run.stop_reason.value == "unrecoverable_tool_error"
        assert run.steps == 1 and run.tool_results[0].error_type is ToolErrorType.PERMISSION_DENIED
        print("verdict=PASS (permission denied; no retry or external read)")


if __name__ == "__main__":
    main()
