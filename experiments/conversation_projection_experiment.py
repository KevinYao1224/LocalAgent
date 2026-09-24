"""验证规范消息只由轨迹事实投影而来。

从项目根目录运行：

    python experiments/conversation_projection_experiment.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import AgentState, ConversationProjector
from llm.base import Message, ModelResponse, ToolCall
from runtime import ToolResult


def experiment_projection_rules() -> None:
    state = AgentState.from_messages(
        messages=[
            Message(role="system", content="Use tools when needed."),
            Message(role="user", content="Calculate 12 + 7."),
        ],
        max_steps=4,
    )

    call = ToolCall(name="add", arguments={"a": 12, "b": 7})
    state.begin_step()
    state.record_model_response(ModelResponse(tool_calls=[call]))
    state.record_tool_execution(
        call=call,
        result=ToolResult.succeeded(tool_name="add", value=19),
    )

        # 空响应和仅含 thinking 的响应属于运行事实，不是对话消息。轨迹必须保留二者，
        # 但消息投影应跳过它们。
    state.begin_step()
    state.record_model_response(ModelResponse())
    state.begin_step()
    state.record_model_response(ModelResponse(
        thinking="The tool returned 19; I should answer next.",
    ))

    state.begin_step()
    state.record_model_response(ModelResponse(content="12 + 7 is 19."))

    messages = ConversationProjector().project(state.trajectory)

    assert [message.role for message in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert messages[2].tool_calls == [call]
    assert messages[3].tool_name == "add"
    assert messages[3].content == "19"
    assert messages[4].content == "12 + 7 is 19."
    assert len(state.steps) == 4

    print("\n=== canonical conversation projection ===")
    for message in messages:
        call_names = [tool_call.name for tool_call in message.tool_calls]
        print(
            f"role={message.role}, content={message.content!r}, "
            f"tool_calls={call_names}"
        )


def experiment_late_input_order() -> None:
    state = AgentState.from_messages(
        messages=[Message(role="user", content="First request")],
        max_steps=2,
    )
    state.begin_step()
    state.record_model_response(ModelResponse(content="First response"))
    state.record_input(Message(role="user", content="Follow-up request"))
    state.begin_step()
    state.record_model_response(ModelResponse(content="Follow-up response"))

    messages = ConversationProjector().project(state.trajectory)

    assert [message.content for message in messages] == [
        "First request",
        "First response",
        "Follow-up request",
        "Follow-up response",
    ]
    assert state.current_task == "Follow-up request"
    print("\nLate input preserves chronological order.")


def experiment_explicit_tool_association() -> None:
    first_call = ToolCall(name="add", arguments={"a": 1, "b": 2})
    second_call = ToolCall(name="add", arguments={"a": 10, "b": 20})
    state = AgentState.from_messages(
        messages=[Message(role="user", content="Calculate two sums.")],
        max_steps=1,
    )
    state.begin_step()
    agent_step = state.record_model_response(ModelResponse(
        tool_calls=[first_call, second_call],
    ))
    state.record_tool_execution(
        call=first_call,
        result=ToolResult.succeeded(tool_name="add", value=3),
    )
    state.record_tool_execution(
        call=second_call,
        result=ToolResult.succeeded(tool_name="add", value=30),
    )

    assert agent_step.tool_executions[0].call is first_call
    assert agent_step.tool_executions[0].result.value == 3
    assert agent_step.tool_executions[1].call is second_call
    assert agent_step.tool_executions[1].result.value == 30
    print("\nSame-named tool calls retain explicit result associations.")


if __name__ == "__main__":
    experiment_projection_rules()
    experiment_late_input_order()
    experiment_explicit_tool_association()
    print("\nAll conversation projection experiments passed.")
