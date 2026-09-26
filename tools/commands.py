"""固定命令工具；须由应用显式授予并注册。"""

from runtime.commands import FixedCommandPolicy
from tools.base import Tool


def create_fixed_command_tool(policy: FixedCommandPolicy, name: str) -> Tool:
    """模型只能选已注册的动作名称，不能改变 argv、cwd 或环境。"""

    if not name or not name.isidentifier():
        raise ValueError("Command tool name must be a nonempty identifier.")
    return Tool(
        name=name,
        description="Run the pre-approved fixed command with no model-supplied arguments.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=policy.run,
    )
