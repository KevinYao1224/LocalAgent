from collections import deque
from copy import deepcopy
from dataclasses import dataclass

from llm.base import Message


@dataclass(frozen=True, slots=True)
class ConversationMemorySelection:
    """一次可观察且不修改存储内容的历史选择结果。"""

    messages: list[Message]
    selected_turns: int
    dropped_turns: int
    selected_characters: int
    history_character_budget: int | None


class ConversationMemory:
    """保存最近已完成轮次的快照，不保存 raw thinking。

    一轮从一条 user 消息开始，以 assistant 最终回答结束。淘汰时始终移除完整轮次，
    包括其中的工具交互。此处限制的是轮次数，不是 token 或字节数。
    """

    def __init__(self, max_turns: int = 5) -> None:
        if isinstance(max_turns, bool) or not isinstance(max_turns, int):
            raise ValueError("max_turns must be a positive integer.")
        if max_turns < 1:
            raise ValueError("max_turns must be a positive integer.")
        self._turns: deque[list[Message]] = deque(maxlen=max_turns)

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    def append(self, messages: list[Message]) -> None:
        """校验一轮已完成的消息，并将其作为独立快照保存。"""

        snapshot = deepcopy(messages)
        self._validate_turn(snapshot)
        for message in snapshot:
            message.thinking = ""
        self._turns.append(snapshot)

    def retrieve(self) -> list[Message]:
        """按时间顺序返回历史副本，调用方可以安全修改副本。"""

        return self.select().messages

    def select(
        self,
        history_character_budget: int | None = None,
    ) -> ConversationMemorySelection:
        """在字符预算内选择最新的连续完整轮次后缀。

        预算只统计保留历史中 ``Message.content`` 的字符数，不包括 system prompt、
        当前 user 消息、工具 schema 或 provider 格式化开销。选择过程不会截断轮次，
        也不会从存储中删除轮次。
        """

        self._validate_history_character_budget(history_character_budget)

        selected_reversed: list[list[Message]] = []
        selected_characters = 0

        for turn in reversed(self._turns):
            turn_characters = sum(len(message.content) for message in turn)
            if (
                history_character_budget is not None
                and selected_characters + turn_characters
                > history_character_budget
            ):
                # 遇到放不下的轮次就停止而非跳过，以保持真正的连续后缀，
                # 并避免旧事实绕过较新的事实被选中。
                break
            selected_reversed.append(turn)
            selected_characters += turn_characters

        selected_turns = len(selected_reversed)
        selected = reversed(selected_reversed)
        messages = [message for turn in selected for message in turn]
        return ConversationMemorySelection(
            messages=deepcopy(messages),
            selected_turns=selected_turns,
            dropped_turns=len(self._turns) - selected_turns,
            selected_characters=selected_characters,
            history_character_budget=history_character_budget,
        )

    def clear(self) -> None:
        """清除此 memory 实例中保留的轮次。"""

        self._turns.clear()

    @staticmethod
    def _validate_history_character_budget(value: int | None) -> None:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(
                "history_character_budget must be a non-negative integer "
                "or None."
            )

    @staticmethod
    def _validate_turn(messages: list[Message]) -> None:
        if len(messages) < 2 or messages[0].role != "user":
            raise ValueError("A turn must start with a user message.")
        if messages[0].tool_calls or messages[0].tool_name is not None:
            raise ValueError("A user message cannot contain tool fields.")

        pending_tools: deque[str] = deque()
        for index, message in enumerate(messages[1:], start=1):
            if pending_tools:
                # 当前 Runtime 按工具调用顺序记录结果。
                if (
                    message.role != "tool"
                    or message.tool_name != pending_tools[0]
                    or message.tool_calls
                ):
                    raise ValueError("Tool results must match pending calls in order.")
                pending_tools.popleft()
                continue

            if message.role != "assistant" or message.tool_name is not None:
                raise ValueError("Expected an assistant message after the user or tools.")
            if message.tool_calls:
                pending_tools.extend(call.name for call in message.tool_calls)
            elif not message.content.strip() or index != len(messages) - 1:
                raise ValueError("A final answer must be nonempty and end the turn.")

        final = messages[-1]
        if pending_tools or final.role != "assistant" or final.tool_calls:
            raise ValueError("A completed turn must end with a final assistant answer.")
