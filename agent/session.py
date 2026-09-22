from agent.loop import AgentLoop, AgentRunResult
from llm.base import Message
from memory import ConversationMemory


class AgentSession:
    """Coordinate sequential user turns with explicit, in-process memory.

    Each run gets a fresh AgentState. Only a completed turn is committed;
    exceptions and max_steps leave memory unchanged. This does not roll back
    tools that have already executed. A session is not concurrency-safe.
    """

    def __init__(
        self,
        agent: AgentLoop,
        system_prompt: str = "",
        memory: ConversationMemory | None = None,
    ) -> None:
        self._agent = agent
        self._system_prompt = system_prompt
        self._memory = memory if memory is not None else ConversationMemory()

    @property
    def memory(self) -> ConversationMemory:
        return self._memory

    def run(self, user_content: str) -> AgentRunResult:
        """Run a new user turn using the retained completed conversation."""

        if not isinstance(user_content, str) or not user_content.strip():
            raise ValueError("user_content must be a nonempty string.")

        messages: list[Message] = []
        if self._system_prompt:
            messages.append(Message(role="system", content=self._system_prompt))
        messages.extend(self._memory.retrieve())
        history_length = len(messages)
        messages.append(Message(role="user", content=user_content))

        result = self._agent.run(messages)
        if result.completed:
            # result.messages also contains the supplied history. Commit only
            # this turn so old turns are not recursively duplicated in memory.
            self._memory.append(result.messages[history_length:])
        return result
