"""只读文件工具必须由调用方显式绑定到授权策略。"""

from runtime.permissions import ReadTextPolicy
from tools.base import Tool


def create_read_text_tool(policy: ReadTextPolicy) -> Tool:
    """返回被固定策略约束的工具；默认入口不注册它。"""

    return Tool(
        name="read_text_file",
        description="Read a UTF-8 text file relative to the authorized directory.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Relative file path."}},
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=policy.read_text,
    )
