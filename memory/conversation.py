from collections import deque
from copy import deepcopy

from llm.base import Message


class ConversationMemory:
    """Keep snapshots of the latest completed turns, without raw thinking.

    A turn starts with one user message and ends with a final assistant answer.
    Eviction always removes whole turns, including their tool exchanges.
    This limits turn count, not tokens or bytes.
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
        """Validate and store one completed turn as an independent snapshot."""

        snapshot = deepcopy(messages)
        self._validate_turn(snapshot)
        for message in snapshot:
            message.thinking = ""
        self._turns.append(snapshot)

    def retrieve(self) -> list[Message]:
        """Return chronological history that callers can mutate safely."""

        return deepcopy([
            message for turn in self._turns for message in turn
        ])

    def clear(self) -> None:
        """Forget the retained turns in this memory instance."""

        self._turns.clear()

    @staticmethod
    def _validate_turn(messages: list[Message]) -> None:
        if len(messages) < 2 or messages[0].role != "user":
            raise ValueError("A turn must start with a user message.")
        if messages[0].tool_calls or messages[0].tool_name is not None:
            raise ValueError("A user message cannot contain tool fields.")

        pending_tools: deque[str] = deque()
        for index, message in enumerate(messages[1:], start=1):
            if pending_tools:
                # The current runtime records results in tool-call order.
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
