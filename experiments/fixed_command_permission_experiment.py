"""Phase 11B：.venv/bin/python experiments/fixed_command_permission_experiment.py。

离线可读实验：仅执行本实验固定的 Python 片段，不连接 Ollama。
"""

import os
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import AgentLoop
from evaluation.cases import ScriptedLLM
from llm.base import Message, ModelResponse, ToolCall
from observability.logger import HumanReadableLogger
from runtime import FixedCommandPolicy, RecoveryPolicy, ToolErrorType, ToolExecutor
from tools.commands import create_fixed_command_tool
from tools.registry import ToolRegistry


def main() -> None:
    executable = str(Path(sys.executable).resolve())
    with TemporaryDirectory() as temporary:
        cwd = Path(temporary).resolve()

        def executor_for(code: str, *, timeout: float = 1, limit: int = 256) -> ToolExecutor:
            policy = FixedCommandPolicy(
                argv=(executable, "-c", code), cwd=cwd,
                environment={"VISIBLE": "approved"},
                timeout_seconds=timeout, max_output_bytes=limit,
            )
            registry = ToolRegistry()
            registry.register(create_fixed_command_tool(policy, "approved_action"))
            return ToolExecutor(registry)

        secret_key = "PHASE11B_EXPERIMENT_SECRET"
        old_secret = os.environ.get(secret_key)
        os.environ[secret_key] = "must-not-leak"
        try:
            executor = executor_for(
                "import os,sys; print(os.getcwd()); print(os.getenv('VISIBLE')); "
                "print(os.getenv('PHASE11B_EXPERIMENT_SECRET')); "
                "print(sys.stdin.read() == ''); print('note', file=sys.stderr)"
            )
            call = ToolCall("approved_action", {})
            result = executor.execute(call)
            assert result.success and result.value == {
                "exit_code": 0,
                "stdout": f"{cwd}\napproved\nNone\nTrue\n",
                "stderr": "note\n",
            }, result
            print(f"fixed command → exit={result.value['exit_code']}, "
                  f"stdout={result.value['stdout']!r}, stderr={result.value['stderr']!r}")
        finally:
            if old_secret is None:
                del os.environ[secret_key]
            else:
                os.environ[secret_key] = old_secret

        # 参数 schema 不允许模型覆写 argv、cwd 或运行策略。
        invalid = executor.execute(ToolCall("approved_action", {"command": "cat /etc/passwd"}))
        assert invalid.error_type is ToolErrorType.VALIDATION_ERROR
        print("model-supplied command → validation_error")
        executor.disable_tool("approved_action")
        assert executor.available_tools() == []
        assert executor.execute(call).error_type is ToolErrorType.TOOL_NOT_AVAILABLE
        print("disabled action → tool_not_available")

        for label, code, timeout, limit, expected in (
            ("timeout", "import time; time.sleep(3)", 0.1, 256, "timed out"),
            ("stdout overflow", "import sys; sys.stdout.write('x'*10000)",
             1, 32, "byte limit"),
            ("stderr overflow", "import sys; sys.stderr.write('x'*10000)",
             1, 32, "byte limit"),
            ("combined overflow", "import sys; sys.stdout.write('x'*20); "
             "sys.stderr.write('y'*20)", 1, 32, "byte limit"),
            ("nonzero exit", "import sys; sys.exit(7)", 1, 256, "status 7"),
        ):
            failed = executor_for(code, timeout=timeout, limit=limit).execute(call)
            assert failed.error_type is ToolErrorType.EXECUTION_ERROR
            assert expected in failed.error
            print(f"{label} → {failed.error_type.value}: {failed.error}")

        # 超时后同组子进程也应被结束，不能继续写出延迟标记。
        marker = cwd / "late.txt"
        child_code = f"import time,pathlib; time.sleep(0.2); pathlib.Path({str(marker)!r}).write_text('late')"
        parent_code = ("import subprocess,sys,time; "
                       f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
                       "time.sleep(3)")
        timed_out = executor_for(parent_code, timeout=0.1).execute(call)
        assert timed_out.error_type is ToolErrorType.EXECUTION_ERROR
        time.sleep(0.3)
        assert not marker.exists(), "A child escaped process-group cleanup"
        print("timeout child cleanup → no delayed write")

        try:
            FixedCommandPolicy((executable, "-c", "pass"), cwd / "missing")
        except ValueError:
            pass
        else:
            raise AssertionError("Missing cwd was accepted")

        # 一次真实 Agent run 可见工具执行、错误及不重试的终止原因。
        limited = executor_for("print('x'*10000)", limit=32)
        llm = ScriptedLLM((ModelResponse(tool_calls=[call]),))
        run = AgentLoop(llm, limited, max_steps=3,
                        recovery_policy=RecoveryPolicy(max_corrections=2),
                        logger=HumanReadableLogger()).run([
                            Message(role="user", content="Run the approved action."),
                        ])
        assert run.stop_reason.value == "unrecoverable_tool_error"
        assert run.steps == 1 and run.tool_results[0].error_type is ToolErrorType.EXECUTION_ERROR
        print("verdict=PASS (fixed argv, bounded execution, no automatic retry)")


if __name__ == "__main__":
    main()
