"""显式、按 namespace 隔离的文本记忆；模型和工具不能直接写入。"""

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: str
    text: str
    created_at: str
    metadata: dict[str, Any]


class LongTermMemory(Protocol):
    """由应用控制写入，并在限定范围内执行确定性读取的接口。"""

    def write(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> MemoryRecord: ...

    def search(
        self, query: str, limit: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemoryRecord]: ...


class SQLiteLongTermMemory:
    """将文本持久化到 SQLite；按不区分大小写的字面子串匹配，最新记录优先。

    每个实例绑定到调用方提供的一个 namespace。namespace 由应用选择，用于隔离数据，
    不提供用户身份认证。
    """

    def __init__(self, path: str | Path, namespace: str) -> None:
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("namespace must be a nonempty string")
        self.path = str(path)
        self.namespace = namespace
        with closing(sqlite3.connect(self.path)) as connection:
            with connection:
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        namespace TEXT NOT NULL,
                        text TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        metadata TEXT NOT NULL
                    )
                """)
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS memories_scope_order "
                    "ON memories(namespace, created_at DESC, id DESC)"
                )

    def write(self, text: str, metadata: dict[str, Any] | None = None) -> MemoryRecord:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a nonempty string")
        encoded = self._encode_metadata(metadata)
        record = MemoryRecord(
            id=uuid4().hex,
            text=text,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=json.loads(encoded),
        )
        with closing(sqlite3.connect(self.path)) as connection:
            with connection:
                connection.execute(
                    "INSERT INTO memories(id, namespace, text, created_at, metadata) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (record.id, self.namespace, text, record.created_at, encoded),
                )
        return record

    def search(
        self, query: str, limit: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemoryRecord]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        filters = json.loads(self._encode_metadata(metadata_filter))
        # 先筛选 metadata 再应用 top-k，避免旧的匹配记录仅因较新的记录
        # metadata 不同而被挤出结果。
        with closing(sqlite3.connect(self.path)) as connection:
            rows = connection.execute(
                "SELECT id, text, created_at, metadata FROM memories "
                "WHERE namespace = ? AND instr(lower(text), lower(?)) > 0 "
                "ORDER BY created_at DESC, id DESC",
                (self.namespace, query),
            )
            matches = []
            for id_, text, created_at, encoded in rows:
                metadata = json.loads(encoded)
                if any(
                    key not in metadata or metadata[key] != value
                    for key, value in filters.items()
                ):
                    continue
                matches.append(MemoryRecord(id_, text, created_at, metadata))
                if len(matches) == limit:
                    break
        return matches

    @staticmethod
    def _encode_metadata(metadata: dict[str, Any] | None) -> str:
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict) or any(
            not isinstance(key, str) for key in metadata
        ):
            raise ValueError("metadata must be a JSON object with string keys")
        try:
            return json.dumps(metadata, ensure_ascii=False, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must contain JSON-serializable values") from exc
