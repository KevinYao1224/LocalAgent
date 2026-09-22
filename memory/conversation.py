from collections import deque
from copy import deepcopy
from dataclasses import dataclass

from llm.base import Message


@dataclass(frozen=True, slots=True)
class ConversationMemorySelection:
    """One observable, non-destructive selection of retained history."""

    messages: list[Message]
    selected_turns: int
    dropped_turns: int
    selected_characters: int
    history_character_budget: int | None


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

        return self.select().messages

    def select(
        self,
        history_character_budget: int | None = None,
    ) -> ConversationMemorySelection:
        """Select the newest complete-turn suffix within a character budget.

        The budget counts only ``Message.content`` in retained history. It does
        not include a system prompt, the current user message, tool schemas, or
        provider formatting. Selection never truncates a turn and never removes
        a turn from storage.
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
                # Stopping, rather than skipping this turn, preserves a true
                # suffix and prevents an older fact from bypassing a newer one.
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
        """Forget the retained turns in this memory instance."""

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
