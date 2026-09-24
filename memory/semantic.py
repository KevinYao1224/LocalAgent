"""Small-scale semantic memory: explicit embeddings, SQLite storage, cosine top-k."""

import json
import math
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Sequence
from uuid import uuid4

from memory.long_term import MemoryRecord, SQLiteLongTermMemory


class TextEmbedder(Protocol):
    """An embedding model with a stable identity for persisted vectors."""

    @property
    def model_id(self) -> str: ...

    def embed(self, text: str) -> Sequence[float]: ...


@dataclass(frozen=True, slots=True)
class ScoredMemory:
    record: MemoryRecord
    similarity: float


def _unit_vector(values: Sequence[float]) -> list[float]:
    """Reject invalid vectors before storing or comparing them."""

    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("embedding must be a nonempty list or tuple")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in values):
        raise ValueError("embedding must contain finite numbers")
    norm = math.hypot(*values)
    if norm == 0 or not math.isfinite(norm):
        raise ValueError("embedding must have a finite nonzero norm")
    return [float(value / norm) for value in values]


class SQLiteSemanticMemory(SQLiteLongTermMemory):
    """Persist vectors alongside text; scan one namespace/model for cosine ranking.

    The embedder identity must identify the model and its version. Changing its
    weights under the same identity makes old vectors incomparable to new ones.
    """

    def __init__(self, path: str | Path, namespace: str, embedder: TextEmbedder) -> None:
        if not isinstance(embedder.model_id, str) or not embedder.model_id.strip():
            raise ValueError("embedder.model_id must be a nonempty string")
        super().__init__(path, namespace)
        self.embedder = embedder
        with closing(sqlite3.connect(self.path)) as connection:
            with connection:
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS memory_embeddings (
                        memory_id TEXT NOT NULL,
                        model_id TEXT NOT NULL,
                        vector TEXT NOT NULL,
                        PRIMARY KEY (memory_id, model_id),
                        FOREIGN KEY (memory_id) REFERENCES memories(id)
                    )
                """)

    def write(self, text: str, metadata: dict | None = None) -> MemoryRecord:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a nonempty string")
        encoded_metadata = self._encode_metadata(metadata)
        vector = _unit_vector(self.embedder.embed(text))
        record = MemoryRecord(
            uuid4().hex, text, datetime.now(timezone.utc).isoformat(),
            json.loads(encoded_metadata),
        )
        with closing(sqlite3.connect(self.path)) as connection:
            with connection:
                existing = connection.execute(
                    "SELECT e.vector FROM memory_embeddings e JOIN memories m "
                    "ON m.id = e.memory_id WHERE m.namespace = ? AND e.model_id = ? LIMIT 1",
                    (self.namespace, self.embedder.model_id),
                ).fetchone()
                if existing and len(json.loads(existing[0])) != len(vector):
                    raise ValueError("embedding dimension differs from stored model vectors")
                connection.execute(
                    "INSERT INTO memories(id, namespace, text, created_at, metadata) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (record.id, self.namespace, text, record.created_at, encoded_metadata),
                )
                connection.execute(
                    "INSERT INTO memory_embeddings(memory_id, model_id, vector) "
                    "VALUES (?, ?, ?)",
                    (record.id, self.embedder.model_id, json.dumps(vector)),
                )
        return record

    def search(
        self, query: str, limit: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[MemoryRecord]:
        return [hit.record for hit in self.search_with_scores(query, limit, metadata_filter)]

    def search_with_scores(
        self, query: str, limit: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[ScoredMemory]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        filters = json.loads(self._encode_metadata(metadata_filter))
        query_vector = _unit_vector(self.embedder.embed(query))
        hits = []
        with closing(sqlite3.connect(self.path)) as connection:
            rows = connection.execute(
                "SELECT m.id, m.text, m.created_at, m.metadata, e.vector "
                "FROM memories m JOIN memory_embeddings e ON e.memory_id = m.id "
                "WHERE m.namespace = ? AND e.model_id = ?",
                (self.namespace, self.embedder.model_id),
            )
            for id_, text, created_at, encoded_metadata, encoded_vector in rows:
                metadata = json.loads(encoded_metadata)
                if any(key not in metadata or metadata[key] != value
                       for key, value in filters.items()):
                    continue
                vector = json.loads(encoded_vector)
                if len(vector) != len(query_vector):
                    raise ValueError("embedding dimension differs from stored model vectors")
                similarity = sum(a * b for a, b in zip(query_vector, vector))
                hits.append(ScoredMemory(
                    MemoryRecord(id_, text, created_at, metadata), similarity,
                ))
        hits.sort(
            key=lambda hit: (hit.similarity, hit.record.created_at, hit.record.id),
            reverse=True,
        )
        return hits[:limit]
