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

# Classification priority order (used for search result ordering)
_CLASSIFICATION_ORDER = {"core": 0, "boundary": 1, "decay": 2, "forget": 3}


class MemoryStore:
    """SQLite-backed store for fractal :class:`~davinci.core.fractal_engine.MemoryNode` objects.

    Parameters
    ----------
    db_path:  Path to the SQLite database file (``":memory:"`` for in-process).
    max_iter: Mandelbrot iteration limit passed through to the fractal engine.

    Examples
    --------
    >>> with MemoryStore(":memory:") as store:
    ...     memory_id = store.store("Hello, DaVinci!")
    ...     node = store.retrieve(memory_id)
    ...     print(node.classification)
    """

    def __init__(self, db_path: str = "davinci_memory.db", max_iter: int = 1000) -> None:
        self._db_path = db_path
        self._max_iter = max_iter
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_schema(self) -> None:
        """Create the ``memories`` table and indexes if they don't exist."""
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
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_classification ON memories (classification)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_frequency ON memories (frequency)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_recency ON memories (recency)"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(
        self,
        content: str,
        zoom_levels: dict[int, str] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> str:
        """Persist a new memory and return its UUID.

        Parameters
        ----------
        content:     The text content of the memory.
        zoom_levels: Optional mapping ``{1: ..., 2: ..., 3: ...}`` of detail
                     levels.  Defaults to *content* at all three levels.
        meta:        Optional JSON-serialisable metadata dict.

        Returns
        -------
        str
            UUID of the newly created memory.
        """
        freq_range, recency_range = self._get_ranges()
        now = time.time()

        zl = zoom_levels or {1: content, 2: content, 3: content}

        node = MemoryNode(
            content=content,
            frequency=0,
            recency=now,
            freq_range=freq_range,
            recency_range=recency_range,
            max_iter=self._max_iter,
            zoom_levels=zl,
        )

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
                    node.frequency,
                    node.recency,
                    node.created_at,
                    node.classification,
                    node.c_value.real,
                    node.c_value.imag,
                    node.iteration_count,
                    node.zoom_levels.get(1, content),
                    node.zoom_levels.get(2, content),
                    node.zoom_levels.get(3, content),
                    meta_json,
                ),
            )

        return memory_id

    def retrieve(self, memory_id: str) -> MemoryNode | None:
        """Fetch a memory by ID, record an access, and return the node.

        The node's ``frequency`` is incremented, ``recency`` is refreshed to
        the current time, and the classification is recomputed against the
        current global ranges before writing back to the database.

        Parameters
        ----------
        memory_id: UUID string of the target memory.

        Returns
        -------
        MemoryNode | None
            The updated node, or ``None`` if no memory with that ID exists.
        """
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()

        if row is None:
            return None

        node = self._row_to_node(row)

        # Record access
        node.frequency += 1
        node.recency = time.time()

        # Recompute with updated global ranges
        freq_range, recency_range = self._get_ranges()
        node._freq_range = freq_range
        node._recency_range = recency_range
        node._recompute()

        with self._conn:
            self._conn.execute(
                """
                UPDATE memories SET
                    frequency = ?,
                    recency = ?,
                    classification = ?,
                    c_real = ?,
                    c_imag = ?,
                    iteration_count = ?
                WHERE id = ?
                """,
                (
                    node.frequency,
                    node.recency,
                    node.classification,
                    node.c_value.real,
                    node.c_value.imag,
                    node.iteration_count,
                    memory_id,
                ),
            )

        return node

    def search(self, query: str, limit: int = 10) -> list[MemoryNode]:
        """Return memories whose content contains *query* (case-insensitive).

        Results are ordered by classification priority
        (core → boundary → decay → forget) then by frequency descending.

        Parameters
        ----------
        query: Substring to search for.
        limit: Maximum number of results to return.

        Returns
        -------
        list[MemoryNode]
        """
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE content LIKE ? COLLATE NOCASE",
            (f"%{query}%",),
        ).fetchall()

        nodes = [self._row_to_node(row) for row in rows]
        nodes.sort(
            key=lambda n: (_CLASSIFICATION_ORDER.get(n.classification, 99), -n.frequency)
        )
        return nodes[:limit]

    def get_by_classification(self, classification: str) -> list[MemoryNode]:
        """Return all memories with the given classification.

        Parameters
        ----------
        classification: One of ``"core"``, ``"boundary"``, ``"decay"``, ``"forget"``.

        Returns
        -------
        list[MemoryNode]
        """
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE classification = ?", (classification,)
        ).fetchall()
        return [self._row_to_node(row) for row in rows]

    def decay_cycle(self) -> dict[str, int]:
        """Reclassify every memory using current global frequency/recency ranges.

        Nodes that have changed classification are updated in the database.

        Returns
        -------
        dict[str, int]
            Mapping of new classification → count of nodes that *moved* to it.
        """
        rows = self._conn.execute("SELECT * FROM memories").fetchall()
        freq_range, recency_range = self._get_ranges()

        changed: dict[str, int] = {}

        for row in rows:
            old_classification = row["classification"]
            c = compute_c(
                row["frequency"],
                row["recency"],
                freq_range,
                recency_range,
            )
            new_classification = classify(c, self._max_iter)

            if new_classification != old_classification:
                iteration_count, _ = iterate(c, self._max_iter)

                with self._conn:
                    self._conn.execute(
                        """
                        UPDATE memories SET
                            classification = ?,
                            c_real = ?,
                            c_imag = ?,
                            iteration_count = ?
                        WHERE id = ?
                        """,
                        (
                            new_classification,
                            c.real,
                            c.imag,
                            iteration_count,
                            row["id"],
                        ),
                    )

                changed[new_classification] = changed.get(new_classification, 0) + 1

        return changed

    def prune(self, classification: str = "forget") -> int:
        """Delete all memories with the given classification.

        Parameters
        ----------
        classification: Classification bucket to prune (default ``"forget"``).

        Returns
        -------
        int
            Number of rows deleted.
        """
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE classification = ?", (classification,)
        )
        count = cursor.fetchone()[0]

        with self._conn:
            self._conn.execute(
                "DELETE FROM memories WHERE classification = ?", (classification,)
            )

        return count

    def migrate(self) -> dict[str, list[str]]:
        """Identify nodes whose classification has drifted and report them.

        Runs a single reclassification pass (like :meth:`decay_cycle`) and
        returns a mapping of *new* classification → list of memory IDs that
        moved there.

        Returns
        -------
        dict[str, list[str]]
            ``{"core": [...], "boundary": [...], ...}`` — only non-empty
            classifications are included.
        """
        rows = self._conn.execute("SELECT * FROM memories").fetchall()
        freq_range, recency_range = self._get_ranges()

        result: dict[str, list[str]] = {}

        for row in rows:
            old_classification = row["classification"]
            c = compute_c(
                row["frequency"],
                row["recency"],
                freq_range,
                recency_range,
            )
            new_classification = classify(c, self._max_iter)

            if new_classification != old_classification:
                iteration_count, _ = iterate(c, self._max_iter)
                with self._conn:
                    self._conn.execute(
                        """
                        UPDATE memories SET
                            classification = ?,
                            c_real = ?,
                            c_imag = ?,
                            iteration_count = ?
                        WHERE id = ?
                        """,
                        (
                            new_classification,
                            c.real,
                            c.imag,
                            iteration_count,
                            row["id"],
                        ),
                    )

                result.setdefault(new_classification, []).append(row["id"])

        return result

    def stats(self) -> dict[str, Any]:
        """Return aggregate statistics about the memory store.

        Returns
        -------
        dict
            Keys: ``total``, ``by_classification`` (counts per bucket),
            ``avg_frequency``, ``oldest_timestamp``, ``newest_timestamp``.
        """
        total_row = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        total = total_row[0]

        by_classification: dict[str, int] = {}
        for cls in ("core", "boundary", "decay", "forget"):
            row = self._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE classification = ?", (cls,)
            ).fetchone()
            by_classification[cls] = row[0]

        agg = self._conn.execute(
            "SELECT AVG(frequency), MIN(created_at), MAX(created_at) FROM memories"
        ).fetchone()

        return {
            "total": total,
            "by_classification": by_classification,
            "avg_frequency": agg[0] if agg[0] is not None else 0.0,
            "oldest_timestamp": agg[1],
            "newest_timestamp": agg[2],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_ranges(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Query DB for min/max frequency and recency.

        Returns
        -------
        (freq_range, recency_range)
            Both as ``(min, max)`` tuples.  Falls back to ``(0, 1)`` when the
            table is empty to avoid degenerate normalisation.
        """
        row = self._conn.execute(
            "SELECT MIN(frequency), MAX(frequency), MIN(recency), MAX(recency) FROM memories"
        ).fetchone()

        if row is None or row[0] is None:
            return (0.0, 1.0), (0.0, 1.0)

        freq_min = float(row[0])
        freq_max = float(row[1])
        rec_min = float(row[2])
        rec_max = float(row[3])

        return (freq_min, freq_max), (rec_min, rec_max)

    def _row_to_node(self, row: sqlite3.Row) -> MemoryNode:
        """Convert a SQLite row to a :class:`MemoryNode`.

        Parameters
        ----------
        row: A ``sqlite3.Row`` object from the ``memories`` table.

        Returns
        -------
        MemoryNode
        """
        freq_range, recency_range = self._get_ranges()

        zoom_levels = {
            1: row["zoom_level_1"],
            2: row["zoom_level_2"],
            3: row["zoom_level_3"],
        }

        node = MemoryNode(
            content=row["content"],
            frequency=row["frequency"],
            recency=row["recency"],
            freq_range=freq_range,
            recency_range=recency_range,
            max_iter=self._max_iter,
            zoom_levels=zoom_levels,
        )
        node.created_at = row["created_at"]
        node.id = row["id"]
        return node

    def _node_to_row(self, node: MemoryNode, memory_id: str) -> tuple:
        """Convert a :class:`MemoryNode` to a tuple for SQLite insert/update.

        Parameters
        ----------
        node:      The node to serialise.
        memory_id: UUID for this row.

        Returns
        -------
        tuple
            Values in the column order of the ``memories`` table INSERT.
        """
        return (
            memory_id,
            node.content,
            node.frequency,
            node.recency,
            node.created_at,
            node.classification,
            node.c_value.real,
            node.c_value.imag,
            node.iteration_count,
            node.zoom_levels.get(1, node.content),
            node.zoom_levels.get(2, node.content),
            node.zoom_levels.get(3, node.content),
            "{}",
        )

    def get_all(self) -> list[MemoryNode]:
        """Return all memories sorted by classification priority then frequency.

        Returns
        -------
        list[MemoryNode]
            All stored memories, ordered core → boundary → decay → forget,
            then by frequency descending within each bucket.
        """
        rows = self._conn.execute("SELECT * FROM memories").fetchall()
        nodes = [self._row_to_node(row) for row in rows]
        nodes.sort(
            key=lambda n: (_CLASSIFICATION_ORDER.get(n.classification, 99), -n.frequency)
        )
        return nodes

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
