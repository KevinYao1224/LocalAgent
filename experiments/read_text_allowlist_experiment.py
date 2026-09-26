"""Phase 11C：精确文件授权的离线实验（无需 Ollama）。

从仓库根目录运行 .venv/bin/python experiments/read_text_allowlist_experiment.py。
"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import AgentLoop
from evaluation.cases import ScriptedLLM
from llm.base import Message, ModelResponse, ToolCall
from observability.logger import HumanReadableLogger
from runtime import ReadTextPolicy, RecoveryPolicy, ToolErrorType, ToolExecutor
from tools.base import PermissionDeniedError
from tools.filesystem import create_read_text_tool
from tools.registry import ToolRegistry


def main() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary) / "docs"
        root.mkdir()
        (root / "public").mkdir()
        (root / "public" / "guide.txt").write_text("Approved guide", encoding="utf-8")
        (root / "public" / "draft.txt").write_text("Not approved", encoding="utf-8")
        (root / "public" / "large.txt").write_text("x" * 129, encoding="utf-8")
        (root / "private.txt").write_text("Private", encoding="utf-8")
        (root / "public" / "alias.txt").symlink_to(root / "private.txt")

        for bad in ("../private.txt", "/tmp/file", "public//guide.txt", "public/./guide.txt",
                    "public\\guide.txt", "", 12, []):
            try:
                ReadTextPolicy(root, allowed_paths=[bad])
            except ValueError:
                pass
            else:
                raise AssertionError(f"Invalid policy path accepted: {bad!r}")

        configured = ["public/guide.txt", "public/alias.txt", "public/large.txt",
                      "public/allowed_but_missing.txt"]
        policy = ReadTextPolicy(root, max_bytes=128, allowed_paths=configured)
        configured.append("private.txt")  # 原始配置的修改不影响已创建策略
        assert policy.allowed_paths == frozenset(configured[:-1])
        registry = ToolRegistry()
        registry.register(create_read_text_tool(policy))
        executor = ToolExecutor(registry)

        def check(path: str, expected: ToolErrorType | None) -> None:
            result = executor.execute(ToolCall("read_text_file", {"path": path}))
            assert result.error_type is expected, (path, result)
            if result.success:
                assert result.value == "Approved guide"
            print(f"{path!r} → {result.value if result.success else result.error_type.value}")

        check("public/guide.txt", None)
        for path in ("public/draft.txt", "private.txt", "public/missing.txt",
                     "public/alias.txt", "public/../private.txt", "public/./guide.txt"):
            check(path, ToolErrorType.PERMISSION_DENIED)
        check("public/large.txt", ToolErrorType.PERMISSION_DENIED)
        check("public/allowed_but_missing.txt", ToolErrorType.EXECUTION_ERROR)

        # 空白名单代表不授予任何文件；即使列入白名单，symlink 与字节上限依旧生效。
        no_list = ReadTextPolicy(root, allowed_paths=[])
        assert no_list.allowed_paths == frozenset()
        try:
            no_list.read_text("public/guide.txt")
        except PermissionDeniedError:
            pass
        else:
            raise AssertionError("Empty allowlist granted access")

        # 可读 run 显示“目录内但不在白名单”也属于不可恢复的授权拒绝。
        llm = ScriptedLLM((ModelResponse(tool_calls=[
            ToolCall("read_text_file", {"path": "public/draft.txt"}),
        ]),))
        run = AgentLoop(llm, executor, max_steps=3,
                        recovery_policy=RecoveryPolicy(max_corrections=2),
                        logger=HumanReadableLogger()).run([
                            Message(role="user", content="Read the draft in the same directory."),
                        ])
        assert run.steps == 1 and run.stop_reason.value == "unrecoverable_tool_error"
        assert run.tool_results[0].error_type is ToolErrorType.PERMISSION_DENIED
        print("verdict=PASS (exact paths, snapshot, symlink denial, no correction)")


if __name__ == "__main__":
    main()
