"""Experiments for JSON Schema validation of untrusted tool arguments.

Run from the project root:

    python experiments/schema_validation_experiment.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.base import ToolCall
from runtime import ToolErrorType, ToolExecutor, ToolSchemaError
from tools import Tool, ToolRegistry


executed_arguments: list[dict] = []


def configure_job(
    mode: str,
    retries: int,
    tasks: list[dict],
) -> dict:
    arguments = {
        "mode": mode,
        "retries": retries,
        "tasks": tasks,
    }
    executed_arguments.append(arguments)
    return arguments


configure_job_tool = Tool(
    name="configure_job",
    description="Configure a job with validated execution settings.",
    parameters={
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["fast", "safe"],
            },
            "retries": {
                "type": "integer",
                "minimum": 0,
                "maximum": 3,
            },
            "tasks": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "minLength": 1,
                        },
                        "priority": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 5,
                        },
                    },
                    "required": ["name", "priority"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["mode", "retries", "tasks"],
        "additionalProperties": False,
    },
    handler=configure_job,
)


def execute(arguments: dict):
    registry = ToolRegistry()
    registry.register(configure_job_tool)
    executor = ToolExecutor(registry)
    return executor.execute(ToolCall(
        name="configure_job",
        arguments=arguments,
    ))


def expect_validation_error(
    label: str,
    arguments: dict,
    expected_text: str,
) -> None:
    calls_before = len(executed_arguments)
    result = execute(arguments)

    assert not result.success
    assert result.error_type is ToolErrorType.VALIDATION_ERROR
    assert result.error is not None
    assert expected_text in result.error
    assert len(executed_arguments) == calls_before

    print(f"\n=== {label} ===")
    print(result.to_message_content())


def main() -> None:
    expect_validation_error(
        "wrong type",
        {
            "mode": "fast",
            "retries": "three",
            "tasks": [{"name": "index", "priority": 3}],
        },
        "$.retries",
    )
    expect_validation_error(
        "additional property",
        {
            "mode": "fast",
            "retries": 1,
            "tasks": [{"name": "index", "priority": 3}],
            "debug": True,
        },
        "Additional properties",
    )
    expect_validation_error(
        "enum",
        {
            "mode": "turbo",
            "retries": 1,
            "tasks": [{"name": "index", "priority": 3}],
        },
        "is not one of",
    )
    expect_validation_error(
        "range",
        {
            "mode": "safe",
            "retries": 10,
            "tasks": [{"name": "index", "priority": 3}],
        },
        "greater than the maximum",
    )
    expect_validation_error(
        "nested object",
        {
            "mode": "safe",
            "retries": 1,
            "tasks": [{"name": "index", "priority": 9}],
        },
        "$.tasks[0].priority",
    )

    valid_arguments = {
        "mode": "safe",
        "retries": 2,
        "tasks": [
            {"name": "index", "priority": 4},
            {"name": "summarize", "priority": 2},
        ],
    }
    result = execute(valid_arguments)

    assert result.success
    assert result.value == valid_arguments
    assert executed_arguments[-1] == valid_arguments

    print("\n=== valid nested arguments ===")
    print(result.to_message_content())

    invalid_schema_tool = Tool(
        name="invalid_schema",
        description="A deliberately invalid tool definition.",
        parameters={"type": "not-a-json-schema-type"},
        handler=lambda: None,
    )
    registry = ToolRegistry()
    registry.register(invalid_schema_tool)
    executor = ToolExecutor(registry)

    try:
        executor.execute(ToolCall(
            name="invalid_schema",
            arguments={},
        ))
    except ToolSchemaError as exc:
        print("\n=== invalid tool schema ===")
        print(exc)
    else:
        raise AssertionError("An invalid tool schema must fail loudly.")

    print("\nAll schema validation experiments passed.")


if __name__ == "__main__":
    main()
