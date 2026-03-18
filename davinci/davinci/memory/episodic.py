"""
DaVinci Memory — Episodic Store
=================================
SQLite-backed conversation history with importance decay and FTS5 search.

Shares the same database file as :class:`~davinci.memory.store.MemoryStore`
but uses a separate ``episodic`` table.  Importance scores decay over time
and low-importance entries can be pruned to keep the store lean.

No external dependencies — pure Python + stdlib only (``sqlite3``).
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import Any

__all__ = ["EpisodicStore"]


class EpisodicStore:
    """SQLite-backed store for episodic (conversation) memories.

    Parameters
    ----------
    db_path: Path to the SQLite database file.  Use ``":memory:"`` for tests.
             Shares the same file as :class:`~davinci.memory.store.MemoryStore`
             when both are pointed at the same path.

    Examples
    --------
    >>> with EpisodicStore(":memory:") as es:
    ...     eid = es.save("What is Python?", "A high-level language.")
    ...     results = es.retrieve("Python")
    ...     print(len(results) >= 1)
    True
    """

    def __init__(self, db_path: str = "davinci_memory.db") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_schema(self) -> None:
        """Create the ``episodic`` table and FTS5 virtual table if absent."""
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodic (
                    id            TEXT PRIMARY KEY,
                    question      TEXT NOT NULL,
                    answer        TEXT NOT NULL,
                    timestamp     REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    importance    REAL NOT NULL DEFAULT 0.5,
                    meta          TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            # FTS5 virtual table for full-text search on question + answer
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS episodic_fts
                USING fts5(question, answer, id UNINDEXED)
                """
            )
            # Triggers to keep FTS5 in sync with the episodic table
            self._conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS episodic_ai
                AFTER INSERT ON episodic BEGIN
                    INSERT INTO episodic_fts(id, question, answer)
                    VALUES (new.id, new.question, new.answer);
                END
                """
            )
            self._conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS episodic_ad
                AFTER DELETE ON episodic BEGIN
                    DELETE FROM episodic_fts WHERE id = old.id;
                END
                """
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        question: str,
        answer: str,
        importance: float = 0.5,
        meta: dict | None = None,
    ) -> str:
        """Save a Q&A episode and return its UUID.

        Parameters
        ----------
        question:   The user's question.
        answer:     The assistant's answer.
        importance: Initial importance score (0–1, default 0.5).
        meta:       Optional JSON-serialisable metadata dict.

        Returns
        -------
        str
            UUID of the newly saved episode.
        """
        now = time.time()
        eid = str(uuid.uuid4())
        meta_json = json.dumps(meta or {})

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO episodic
                    (id, question, answer, timestamp, last_accessed, importance, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (eid, question, answer, now, now, importance, meta_json),
            )

        return eid

    def retrieve(self, query: str, limit: int = 5) -> list[dict]:
        """Search episodic entries using FTS5 and return the best matches.

        Access boosts the importance of each returned entry.

        Parameters
        ----------
        query: Full-text search query string.
        limit: Maximum number of results to return.

        Returns
        -------
        list[dict]
            Each dict has keys: ``id``, ``question``, ``answer``,
            ``timestamp``, ``last_accessed``, ``importance``, ``meta``.
        """
        try:
            rows = self._conn.execute(
                """
                SELECT e.* FROM episodic e
                JOIN episodic_fts f ON f.id = e.id
                WHERE episodic_fts MATCH ?
                ORDER BY e.importance DESC, e.timestamp DESC
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # FTS5 query syntax error — fall back to LIKE search
            rows = self._conn.execute(
                """
                SELECT * FROM episodic
                WHERE question LIKE ? OR answer LIKE ?
                ORDER BY importance DESC, timestamp DESC
                LIMIT ?
                """,
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()

        results = []
        now = time.time()
        for row in rows:
            # Boost importance on access
            new_importance = min(1.0, row["importance"] + 0.1)
            with self._conn:
                self._conn.execute(
                    "UPDATE episodic SET last_accessed = ?, importance = ? WHERE id = ?",
                    (now, new_importance, row["id"]),
                )
            results.append(self._row_to_dict(row))

        return results

    def decay(self, rate_per_day: float = 0.05) -> int:
        """Reduce importance scores based on time since last access.

        Parameters
        ----------
        rate_per_day: Fraction of importance lost per day of inactivity.

        Returns
        -------
        int
            Number of entries that had their importance reduced.
        """
        now = time.time()
        rows = self._conn.execute(
            "SELECT id, last_accessed, importance FROM episodic"
        ).fetchall()

        count = 0
        for row in rows:
            days_idle = (now - row["last_accessed"]) / 86400.0
            decay_amount = rate_per_day * days_idle
            if decay_amount <= 0:
                continue
            new_importance = max(0.0, row["importance"] - decay_amount)
            if new_importance != row["importance"]:
                with self._conn:
                    self._conn.execute(
                        "UPDATE episodic SET importance = ? WHERE id = ?",
                        (new_importance, row["id"]),
                    )
                count += 1

        return count

    def prune(self, threshold: float = 0.2) -> int:
        """Delete entries whose importance is below *threshold*.

        Parameters
        ----------
        threshold: Minimum importance to keep (default 0.2).

        Returns
        -------
        int
            Number of entries deleted.
        """
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM episodic WHERE importance < ?", (threshold,)
        )
        count = cursor.fetchone()[0]

        with self._conn:
            self._conn.execute(
                "DELETE FROM episodic WHERE importance < ?", (threshold,)
            )

        return count

    def count(self) -> int:
        """Return the total number of episodic entries."""
        return self._conn.execute("SELECT COUNT(*) FROM episodic").fetchone()[0]

    def status(self) -> dict[str, Any]:
        """Return aggregate statistics about the episodic store.

        Returns
        -------
        dict
            Keys: ``count``, ``avg_importance``, ``oldest_timestamp``.
        """
        row = self._conn.execute(
            "SELECT COUNT(*), AVG(importance), MIN(timestamp) FROM episodic"
        ).fetchone()
        return {
            "count": row[0],
            "avg_importance": row[1] if row[1] is not None else 0.0,
            "oldest_timestamp": row[2],
        }

    def clear(self) -> int:
        """Delete all episodic entries.

        Returns
        -------
        int
            Number of entries deleted.
        """
        count = self.count()
        with self._conn:
            self._conn.execute("DELETE FROM episodic")
        return count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "timestamp": row["timestamp"],
            "last_accessed": row["last_accessed"],
            "importance": row["importance"],
            "meta": json.loads(row["meta"]) if row["meta"] else {},
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> "EpisodicStore":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
