from agent.state import AgentStep, InputMessage, TrajectoryEntry
from llm.base import Message


class ConversationProjector:
    """根据轨迹事实构造规范的模型对话。

    投影过程有意保持无策略：不回放 thinking、不截断历史、不摘要内容，也不添加
    memory。这些选择属于 ContextBuilder 或后续的上下文策略。
    """

    def project(self, trajectory: list[TrajectoryEntry]) -> list[Message]:
        messages: list[Message] = []

        for entry in trajectory:
            messages.extend(self.project_entry(entry))

        return messages

    def project_entry(self, entry: TrajectoryEntry) -> list[Message]:
        """投影一条轨迹事实，不应用上下文选择策略。"""

        if isinstance(entry, InputMessage):
            return [entry.message]

        if not isinstance(entry, AgentStep):
            raise TypeError(
                f"Unsupported trajectory entry: {type(entry).__name__}"
            )

        messages: list[Message] = []
        response = entry.model_response
        if response.tool_calls or response.content.strip():
            messages.append(Message(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls,
            ))

        for execution in entry.tool_executions:
            messages.append(Message(
                role="tool",
                tool_name=execution.call.name,
                content=execution.result.to_message_content(),
            ))

        return messages
