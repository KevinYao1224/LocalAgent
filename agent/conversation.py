from agent.state import AgentStep, InputMessage, TrajectoryEntry
from llm.base import Message


class ConversationProjector:
    """Build the canonical model conversation from trajectory facts.

    Projection is intentionally policy-free. It does not replay thinking,
    truncate history, summarize content, or add memory. Those choices belong
    to the future ContextBuilder.
    """

    def project(self, trajectory: list[TrajectoryEntry]) -> list[Message]:
        messages: list[Message] = []

        for entry in trajectory:
            if isinstance(entry, InputMessage):
                messages.append(entry.message)
                continue

            if not isinstance(entry, AgentStep):
                raise TypeError(
                    f"Unsupported trajectory entry: {type(entry).__name__}"
                )

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
