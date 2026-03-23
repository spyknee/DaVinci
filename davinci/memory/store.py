"""
DaVinci Memory — Layer 2: Memory Store
=======================================
SQLite-backed persistence layer for fractal memory nodes.

Every memory node is stored with its fractal attributes (c value, iteration
count, classification) and automatically reclassified as access patterns change
over time via :meth:`~MemoryStore.decay_cycle`.

No external dependencies — pure Python + stdlib only (``sqlite3``).
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import Any

from davinci.core.fractal_engine import (
    MemoryNode,
    classify,
    compute_c,
    iterate,
)

__all__ = ["MemoryStore"]

_CLASSIFICATION_ORDER = {"core": 0, "boundary": 1, "decay": 2, "forget": 3}
_MAX_RECENCY_AGE_SECONDS = 30 * 24 * 60 * 60.0
_MAX_FREQUENCY = 1000.0


def _default_zoom_levels(content: str) -> dict[int, str]:
    first_sentence_end = content.find('. ')
    if 0 < first_sentence_end <= 100:
        zoom1 = content[:first_sentence_end + 1]
    else:
        zoom1 = content[:100] + ("…" if len(content) > 100 else "")
    zoom2 = content[:500] + ("…" if len(content) > 500 else "")
    return {1: zoom1, 2: zoom2, 3: content}


class MemoryStore:
    """SQLite-backed store for fractal :class:`~davinci.core.fractal_engine.MemoryNode` objects."""

    def __init__(self, db_path: str = "davinci_memory.db", max_iter: int = 1000) -> None:
        self._db_path = db_path
        self._max_iter = max_iter
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_schema()
        self._init_meta()

    def _create_schema(self) -> None:
        """Create the ``memories`` table, indexes, FTS5 virtual table."""
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id              TEXT PRIMARY KEY,
                    content         TEXT NOT NULL,
                    frequency       INTEGER NOT NULL DEFAULT 0,
                    recency         REAL NOT NULL,
                    created_at      REAL NOT NULL,
                    classification  TEXT NOT NULL,
                    c_real          REAL NOT NULL,
                    c_imag          REAL NOT NULL,
                    iteration_count INTEGER NOT NULL,
                    zoom_level_1    TEXT NOT NULL,
                    zoom_level_2    TEXT NOT NULL,
                    zoom_level_3    TEXT NOT NULL,
                    meta            TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            for idx in ("classification", "frequency", "recency"):
                self._conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{idx} ON memories ({idx})")

            try:
                self._conn.execute(
                    """CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(id UNINDEXED, content)"""
                )
                for trigger in ("ai", "ad"):
                    self._conn.execute(
                        f"""CREATE TRIGGER IF NOT EXISTS memories_{trigger}
                            AFTER {trigger[0].upper()}{trigger[1:]} ON memories BEGIN
                            INSERT INTO memories_fts(id, content)
                            VALUES (new.id, new.content);
                            END"""
                    )
            except sqlite3.OperationalError:
                pass

    def _init_meta(self) -> None:
        """Ensure meta table exists with turn counter."""
        with self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value INTEGER DEFAULT 0)"
            )
            self._conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('turns', 0)")

    def _get_turn_count(self) -> int:
        """Return current global turn count."""
        row = self._conn.execute("SELECT value FROM meta WHERE key='turns'").fetchone()
        return row[0] if row else 0

    def _inc_turn_count(self) -> None:
        """Increment global turn count by one."""
        with self._conn:
            self._conn.execute("UPDATE meta SET value = value + 1 WHERE key='turns'")

    def store(
        self,
        content: str,
        zoom_levels: dict[int, str] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> str:
        """Persist a new memory and return its UUID."""
        now = time.time()
        freq_range, recency_range = (0.0, _MAX_FREQUENCY), (0.0, _MAX_RECENCY_AGE_SECONDS)
        zl = zoom_levels or _default_zoom_levels(content)

        node = MemoryNode(
            content=content,
            frequency=_MAX_FREQUENCY * 0.5,
            recency=_MAX_RECENCY_AGE_SECONDS,
            freq_range=freq_range,
            recency_range=recency_range,
            max_iter=self._max_iter,
            zoom_levels=zl,
        )
        node.classification = "core"

        memory_id = str(uuid.uuid4())
        meta_json = json.dumps(meta or {})

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO memories
                    (id, content, frequency, recency, created_at,
                     classification, c_real, c_imag, iteration_count,
                     zoom_level_1, zoom_level_2, zoom_level_3, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    node.content,
                    int(node.frequency),
                    now,
                    now,
                    node.classification,
                    node.c_value.real,
                    node.c_value.imag,
                    node.iteration_count,
                    zl.get(1, content[:100]),
                    zl.get(2, content[:500]),
                    zl.get(3, content),
                    meta_json,
                ),
            )

        return memory_id

    def retrieve(self, memory_id: str) -> MemoryNode | None:
        """Fetch a memory by ID, record an access, and return the node."""
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None

        freq_range, recency_range = (0.0, _MAX_FREQUENCY), (0.0, _MAX_RECENCY_AGE_SECONDS)
        node = self._row_to_node(row, freq_range, recency_range)
        node.frequency += 1
        node.recency = _MAX_RECENCY_AGE_SECONDS

        with self._conn:
            self._conn.execute(
                """
                UPDATE memories SET frequency = ?, recency = ?, classification = ? WHERE id = ?
                """,
                (int(node.frequency), time.time(), node.classification, memory_id),
            )
        return node

    def search(self, query: str, limit: int = 10) -> list[MemoryNode]:
        """Return memories matching *query*, ranked by relevance."""
        query_lower = query.lower()
        words = query_lower.split()[:15]

        rows = None
        try:
            fts_ids = [
                r[0] for r in self._conn.execute(
                    "SELECT id FROM memories_fts WHERE memories_fts MATCH ?",
                    (" OR ".join(f"{w}*" for w in words),)
                ).fetchall()
            ]
            if fts_ids:
                rows = self._conn.execute(
                    f"SELECT * FROM memories WHERE id IN ({','.join('?'*len(fts_ids))})",
                    fts_ids
                ).fetchall()
        except sqlite3.OperationalError:
            pass

        if rows is None:
            where = " OR ".join(["content LIKE ? COLLATE NOCASE"] * len(words))
            rows = self._conn.execute(
                f"SELECT * FROM memories WHERE {where}",
                [f"%{w}%" for w in words]
            ).fetchall()

        freq_range, recency_range = (0.0, _MAX_FREQUENCY), (0.0, _MAX_RECENCY_AGE_SECONDS)
        scored: list[tuple[int, MemoryNode]] = []

        for row in rows:
            content_lower = row["content"].lower()
            score = (
                3 if query_lower in content_lower else
                2 if all(w in content_lower for w in words) else
                1 if any(w in content_lower for w in words) else 0
            )
            if not score:
                continue

            node = self._row_to_node(row, freq_range, recency_range)
            scored.append((score, node))

        scored.sort(key=lambda x: (-x[0], _CLASSIFICATION_ORDER.get(x[1].classification, 99), -x[1].frequency))
        return [n for _, n in scored[:limit]]

    def get_by_classification(self, classification: str) -> list[MemoryNode]:
        """Return all memories with the given classification."""
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE classification = ?", (classification,)
        ).fetchall()
        freq_range, recency_range = (0.0, _MAX_FREQUENCY), (0.0, _MAX_RECENCY_AGE_SECONDS)
        nodes = []
        for row in rows:
            node = self._row_to_node(row, freq_range, recency_range)
            nodes.append(node)
        return nodes

    def decay_cycle(self) -> dict[str, list[str]]:
        """Reclassify every memory using current global frequency/recency ranges."""
        now = time.time()
        range_row = self._conn.execute(
            "SELECT MIN(frequency), MAX(frequency), MIN(recency) FROM memories WHERE classification != 'forget'"
        ).fetchone()

        if not range_row or range_row[0] is None:
            return {}

        freq_min, freq_max, oldest_recency = range_row
        freq_range = (freq_min or 0.0, freq_max or _MAX_FREQUENCY)
        recency_range = (0.0, _MAX_RECENCY_AGE_SECONDS)

        changed: dict[str, list[str]] = {}

        for row in self._conn.execute("SELECT * FROM memories").fetchall():
            old_class = row["classification"]
            age = now - row["recency"] if row["recency"] > 0 else _MAX_RECENCY_AGE_SECONDS
            freshness = max(0.0, _MAX_RECENCY_AGE_SECONDS - min(age, _MAX_RECENCY_AGE_SECONDS))
            c = compute_c(row["frequency"], freshness, freq_range, recency_range)
            new_class = classify(c, self._max_iter)

            if new_class != old_class:
                priority_diff = _CLASSIFICATION_ORDER.get(new_class, 99) - _CLASSIFICATION_ORDER.get(old_class, 99)
                should_update = priority_diff > 0 or abs(priority_diff) >= 2
                if should_update:
                    iteration_count, _ = iterate(c, self._max_iter)
                    with self._conn:
                        self._conn.execute(
                            """
                            UPDATE memories SET classification = ?, c_real = ?, c_imag = ?, iteration_count = ? WHERE id = ?
                            """,
                            (new_class, c.real, c.imag, iteration_count, row["id"]),
                        )
                    changed.setdefault(new_class, []).append(row["id"])

        return {k: v for k, v in changed.items() if v}

    def consolidate(self, strategy: str = "frequency") -> int:
        """Merge similar memories based on strategy."""
        nodes = self.get_by_classification("core")
        merged_count = 0
        i = 0
        while i < len(nodes):
            node1 = nodes[i]
            j = i + 1
            while j < len(nodes):
                node2 = nodes[j]
                text1, text2 = node1.content.strip().lower(), node2.content.strip().lower()
                if text1 == text2 or (len(set(text1.split()) & set(text2.split())) >= min(len(text1.split()), len(text2.split())) * 0.8):
                    with self._conn:
                        self._conn.execute("DELETE FROM memories WHERE id = ?", (node2.id,))
                    nodes.pop(j)
                    merged_count += 1
                else:
                    j += 1
            i += 1
        return merged_count

    def prune(self, classification: str = "forget") -> int:
        """Delete all memories with the given classification."""
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE classification = ?", (classification,)
        )
        count = cursor.fetchone()[0]
        with self._conn:
            self._conn.execute("DELETE FROM memories WHERE classification = ?", (classification,))
        return count

    def stats(self) -> dict[str, Any]:
        """Return aggregate statistics."""
        total_row = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        total = total_row[0] if total_row else 0
        by_classification: dict[str, int] = {}
        for cls in ("core", "boundary", "decay", "forget"):
            row = self._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE classification = ?", (cls,)
            ).fetchone()
            by_classification[cls] = row[0] if row else 0
        agg = self._conn.execute("SELECT AVG(frequency), MIN(created_at), MAX(created_at) FROM memories").fetchone()
        return {
            "total": total,
            "by_classification": by_classification,
            "avg_frequency": agg[0] if agg[0] is not None else 0.0,
            "oldest_timestamp": agg[1],
            "newest_timestamp": agg[2],
        }

    def get_all(self) -> list[MemoryNode]:
        """Return all memories sorted by classification priority then frequency."""
        rows = self._conn.execute("SELECT * FROM memories").fetchall()
        freq_range, recency_range = (0.0, _MAX_FREQUENCY), (0.0, _MAX_RECENCY_AGE_SECONDS)
        nodes = []
        for row in rows:
            node = self._row_to_node(row, freq_range, recency_range)
            nodes.append(node)
        nodes.sort(key=lambda n: (_CLASSIFICATION_ORDER.get(n.classification, 99), -n.frequency))
        return nodes

    def _row_to_node(self, row: sqlite3.Row, freq_range: tuple, recency_range: tuple) -> MemoryNode:
        """Convert a SQLite row to a :class:`MemoryNode`."""
        now = time.time()
        freshness = max(0.0, _MAX_RECENCY_AGE_SECONDS - (now - row["recency"]))
        zoom_levels = {
            1: row["zoom_level_1"],
            2: row["zoom_level_2"],
            3: row["zoom_level_3"],
        }
        node = MemoryNode(
            content=row["content"],
            frequency=row["frequency"],
            recency=freshness,
            freq_range=freq_range,
            recency_range=recency_range,
            max_iter=self._max_iter,
            zoom_levels=zoom_levels,
        )
        node.created_at = row["created_at"]
        node.id = row["id"]
        node.classification = row["classification"]
        node.c_value = complex(row["c_real"], row["c_imag"])
        node.iteration_count = row["iteration_count"]
        return node

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
