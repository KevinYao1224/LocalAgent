from collections.abc import Iterable
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from tools.base import Tool


class ToolArgumentsValidationError(Exception):
    """模型提供的参数与工具 schema 不匹配时抛出。"""


class ToolSchemaError(Exception):
    """工具包含无效 JSON Schema 定义时抛出。"""


class ToolArgumentsValidator:
    """使用 JSON Schema Draft 2020-12 校验不可信的工具参数。"""

    def validate(self, tool: Tool, arguments: Any) -> None:
        try:
            Draft202012Validator.check_schema(tool.parameters)
        except SchemaError as exc:
            raise ToolSchemaError(
                f"Tool '{tool.name}' has an invalid parameter schema: "
                f"{exc.message}"
            ) from exc

        validator = Draft202012Validator(tool.parameters)
        errors = sorted(
            validator.iter_errors(arguments),
            key=self._error_sort_key,
        )

        if errors:
            details = "; ".join(
                self._format_error(error)
                for error in errors
            )
            raise ToolArgumentsValidationError(
                f"Invalid arguments for tool '{tool.name}': {details}"
            )

    @staticmethod
    def _error_sort_key(error: ValidationError) -> tuple[str, ...]:
        return tuple(str(part) for part in error.absolute_path)

    def _format_error(self, error: ValidationError) -> str:
        return f"{self._format_path(error.absolute_path)}: {error.message}"

    @staticmethod
    def _format_path(path: Iterable[Any]) -> str:
        result = "$"

        for part in path:
            if isinstance(part, int):
                result += f"[{part}]"
            else:
                result += f".{part}"

        return result
