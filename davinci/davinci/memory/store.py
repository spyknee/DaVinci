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

# Classification priority order: core (best) → forget (worst).
# Used for search/sort ordering and for decay-cycle hysteresis decisions.
_CLASSIFICATION_ORDER = {"core": 0, "boundary": 1, "decay": 2, "forget": 3}

# Maximum age (in seconds) used as the fixed recency normalisation range.
# A memory last accessed more than this long ago maps to the ``forget`` end of
# the real axis; one accessed just now maps to the ``core`` end.  Using a fixed
# range means adding new memories never shifts existing classifications, and the
# decay cycle degrades memories naturally over time without any patching.
_MAX_RECENCY_AGE_SECONDS = 30 * 24 * 60 * 60.0  # 30 days in seconds
_MAX_FREQUENCY = 1000.0  # Fixed frequency ceiling — population-independent axis
# _MAX_FREQUENCY documents the intended upper bound for the frequency axis and
# is available for use whenever a fixed ceiling is needed.


def _default_zoom_levels(content: str) -> dict[int, str]:
    """Generate default zoom levels from content when none are supplied.

    Parameters
    ----------
    content: The full text content of the memory.

    Returns
    -------
    dict[int, str]
        ``{1: zoom1, 2: zoom2, 3: content}`` where zoom1 is the first
        sentence (up to 100 chars) and zoom2 is the first 500 chars.
    """
    # Zoom 1: first sentence or first 100 chars, whichever is shorter
    first_sentence_end = content.find('. ')
    if 0 < first_sentence_end <= 100:
        zoom1 = content[:first_sentence_end + 1]
    else:
        zoom1 = content[:100] + ("…" if len(content) > 100 else "")

    # Zoom 2: first 500 chars
    zoom2 = content[:500] + ("…" if len(content) > 500 else "")

    # Zoom 3: full content
    return {1: zoom1, 2: zoom2, 3: content}


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
        """Create the ``memories`` table, FTS5 virtual table, indexes, and triggers."""
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
            # FTS5 virtual table for full-text search.  Wrapped in try/except so
            # that builds of SQLite without FTS5 support degrade gracefully.
            try:
                self._conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                    USING fts5(id UNINDEXED, content)
                    """
                )
                self._conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS memories_ai
                    AFTER INSERT ON memories BEGIN
                        INSERT INTO memories_fts(id, content)
                        VALUES (new.id, new.content);
                    END
                    """
                )
                self._conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS memories_ad
                    AFTER DELETE ON memories BEGIN
                        DELETE FROM memories_fts WHERE id = old.id;
                    END
                    """
                )
            except sqlite3.OperationalError:
                pass  # FTS5 not available on this SQLite build

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
        now = time.time()
        freq_range, recency_range = self._get_ranges(now=now)

        zl = zoom_levels or _default_zoom_levels(content)

        # Brand-new memory = maximum freshness so it maps to 'core'.
        node = MemoryNode(
            content=content,
            frequency=0,
            recency=_MAX_RECENCY_AGE_SECONDS,
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
                    now,              # raw Unix timestamp for age tracking
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

        freq_range, recency_range = self._get_ranges()
        node = self._row_to_node(row, freq_range, recency_range)

        # Record access
        node.frequency += 1
        # Just accessed = maximum freshness.  We track the raw timestamp
        # separately for the DB update so age can be computed later.
        node.recency = _MAX_RECENCY_AGE_SECONDS
        raw_now = time.time()

        # Recompute classification with current global ranges
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
                    raw_now,          # raw Unix timestamp for age tracking
                    node.classification,
                    node.c_value.real,
                    node.c_value.imag,
                    node.iteration_count,
                    memory_id,
                ),
            )

        return node

    def search(self, query: str, limit: int = 10) -> list[MemoryNode]:
        """Return memories whose content matches *query*, ranked by relevance.

        Scoring:
        - Exact phrase match (case-insensitive) → score 3
        - All query words present → score 2
        - Any query word present → score 1

        Within the same score, results are ordered by classification priority
        (core → boundary → decay → forget) then by frequency descending.

        Parameters
        ----------
        query: Search string.
        limit: Maximum number of results to return.

        Returns
        -------
        list[MemoryNode]
        """
        query_lower = query.lower()
        words = query_lower.split()

        if not words:
            return []

        # Cap at 15 words to prevent excessive clause construction.
        words = words[:15]

        # Pre-filter via FTS5 when available; fall back to LIKE otherwise.
        # FTS5 prefix-OR query (e.g. "quick* OR brown*") replicates the
        # LIKE '%word%' OR semantics so that the Python scoring pass below
        # receives the same candidate set as before.
        rows = None
        try:
            fts_terms = " OR ".join(f"{w}*" for w in words)
            fts_ids = [
                r[0]
                for r in self._conn.execute(
                    "SELECT id FROM memories_fts WHERE memories_fts MATCH ?",
                    (fts_terms,),
                ).fetchall()
            ]
            if fts_ids:
                placeholders = ",".join("?" * len(fts_ids))
                rows = self._conn.execute(
                    f"SELECT * FROM memories WHERE id IN ({placeholders})",
                    fts_ids,
                ).fetchall()
            else:
                rows = []
        except sqlite3.OperationalError:
            rows = None

        if rows is None:
            # FTS5 unavailable or query error — fall back to LIKE pre-filter.
            where_clauses = " OR ".join(["content LIKE ? COLLATE NOCASE"] * len(words))
            params = [f"%{w}%" for w in words]
            rows = self._conn.execute(
                f"SELECT * FROM memories WHERE {where_clauses}",
                params,
            ).fetchall()

        freq_range, recency_range = self._get_ranges()

        scored: list[tuple[int, MemoryNode]] = []
        for row in rows:
            content_lower = row["content"].lower()
            if query_lower in content_lower:
                score = 3
            elif all(w in content_lower for w in words):
                score = 2
            elif any(w in content_lower for w in words):
                score = 1
            else:
                continue
            node = self._row_to_node(row, freq_range, recency_range)
            node._recompute()
            scored.append((score, node))

        scored.sort(key=lambda item: (-item[0], _CLASSIFICATION_ORDER.get(item[1].classification, 99), -item[1].frequency))
        return [node for _, node in scored[:limit]]

    def search_fts(self, query: str, limit: int = 10) -> list[MemoryNode]:
        """Full-text search using SQLite FTS5, falling back to LIKE on error.

        Unlike :meth:`search`, this method does not apply score-based ranking;
        results are ordered by classification priority (core → forget) then by
        frequency descending.

        Parameters
        ----------
        query: FTS5 query string (single word, phrase, or FTS5 expression).
        limit: Maximum number of results to return.

        Returns
        -------
        list[MemoryNode]
        """
        if not query.strip():
            return []

        rows = None
        try:
            fts_ids = [
                r[0]
                for r in self._conn.execute(
                    "SELECT id FROM memories_fts WHERE memories_fts MATCH ?",
                    (query,),
                ).fetchall()
            ]
            if not fts_ids:
                return []
            placeholders = ",".join("?" * len(fts_ids))
            rows = self._conn.execute(
                f"SELECT * FROM memories WHERE id IN ({placeholders})",
                fts_ids,
            ).fetchall()
        except sqlite3.OperationalError:
            rows = None

        if rows is None:
            # FTS5 unavailable or invalid query — fall back to LIKE.
            words = query.lower().split()[:15]
            if not words:
                return []
            where_clauses = " OR ".join(["content LIKE ? COLLATE NOCASE"] * len(words))
            params = [f"%{w}%" for w in words]
            rows = self._conn.execute(
                f"SELECT * FROM memories WHERE {where_clauses}",
                params,
            ).fetchall()

        freq_range, recency_range = self._get_ranges()
        nodes = []
        for row in rows:
            node = self._row_to_node(row, freq_range, recency_range)
            node._recompute()
            nodes.append(node)

        nodes.sort(key=lambda n: (_CLASSIFICATION_ORDER.get(n.classification, 99), -n.frequency))
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
        freq_range, recency_range = self._get_ranges()
        nodes = []
        for row in rows:
            node = self._row_to_node(row, freq_range, recency_range)
            node._recompute()
            nodes.append(node)
        return nodes

    def decay_cycle(self) -> dict[str, list[str]]:
        """Reclassify every memory using current global frequency/recency ranges.

        Nodes that have changed classification are updated in the database.
        Ranges are computed excluding nodes already classified as ``forget``
        so that stale outliers do not skew the active distribution.

        Hysteresis is applied: a node only updates if it is decaying (moving
        toward ``forget``) OR improving by at least two full tiers.  This
        prevents classification flickering at tier boundaries.

        Returns
        -------
        dict[str, list[str]]
            Mapping of new classification → list of memory IDs that *moved* to it.
            Only classifications that received at least one node are included.
        """
        rows = self._conn.execute("SELECT * FROM memories").fetchall()

        # Compute frequency range excluding current forget nodes to avoid skew
        range_row = self._conn.execute(
            """SELECT MIN(frequency), MAX(frequency)
               FROM memories WHERE classification != 'forget'"""
        ).fetchone()

        if range_row is None or range_row[0] is None:
            # No non-forget nodes — nothing meaningful to reclassify.
            return {}

        freq_range: tuple[float, float] = (0.0, _MAX_FREQUENCY)
        recency_range: tuple[float, float] = (0.0, _MAX_RECENCY_AGE_SECONDS)

        changed: dict[str, list[str]] = {}
        now = time.time()

        for row in rows:
            old_classification = row["classification"]
            freshness = max(0.0, _MAX_RECENCY_AGE_SECONDS - (now - row["recency"]))
            c = compute_c(
                row["frequency"],
                freshness,
                freq_range,
                recency_range,
            )
            new_classification = classify(c, self._max_iter)

            if new_classification != old_classification:
                old_priority = _CLASSIFICATION_ORDER.get(old_classification, 99)
                new_priority = _CLASSIFICATION_ORDER.get(new_classification, 99)

                # Allow decay (getting worse) freely; require 2+ tier improvement
                if new_priority > old_priority:
                    should_update = True   # always decay
                elif old_priority - new_priority >= 2:
                    should_update = True   # significant improvement
                else:
                    should_update = False  # suppress minor flicker (e.g. boundary ↔ core)

                if should_update:
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

                    changed.setdefault(new_classification, []).append(row["id"])

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

    def _backfill_fts(self) -> None:
        """Insert any ``memories`` rows that are missing from the FTS5 index.

        Call this once after opening a database that was created before FTS5
        support was added, or after manually removing rows from the FTS5
        shadow table.  Normal writes through :meth:`store` and deletes through
        :meth:`prune` are kept in sync automatically via SQL triggers, so this
        method is only needed for one-time migration.

        Silently does nothing if FTS5 is not available on this SQLite build.
        """
        try:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO memories_fts(id, content)
                    SELECT id, content FROM memories
                    WHERE id NOT IN (SELECT id FROM memories_fts)
                    """
                )
        except sqlite3.OperationalError:
            pass  # FTS5 not available

    def _get_ranges(self, now: float | None = None) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return fixed frequency and recency ranges for normalisation.

        Both axes are fixed (not derived from the current DB population) so
        that adding or removing memories never shifts the classification of
        existing ones.

        Returns
        -------
        (freq_range, recency_range)
            ``freq_range`` is always ``(0.0, 0.0)`` — a degenerate range whose
            midpoint maps every frequency to the imaginary-axis center so that
            classification is driven purely by recency, not by
            population-relative frequency comparisons.
            ``recency_range`` is always ``(0.0, _MAX_RECENCY_AGE_SECONDS)``.
        """
        return (0.0, _MAX_FREQUENCY), (0.0, _MAX_RECENCY_AGE_SECONDS)

    def _row_to_node(self, row: sqlite3.Row, freq_range: tuple, recency_range: tuple) -> MemoryNode:
        """Convert a SQLite row to a :class:`MemoryNode`.

        Parameters
        ----------
        row:           A ``sqlite3.Row`` object from the ``memories`` table.
        freq_range:    ``(min, max)`` frequency range from :meth:`_get_ranges`.
        recency_range: ``(min, max)`` recency range from :meth:`_get_ranges`
                       (always ``(0, _MAX_RECENCY_AGE_SECONDS)``).

        Returns
        -------
        MemoryNode
            The node's ``recency`` attribute is set to the *freshness score*
            (``max(0, MAX_AGE - age)``) so that :meth:`~MemoryNode._recompute`
            maps it correctly onto the complex plane.  The raw Unix timestamp
            remains stored in the database ``recency`` column.
        """
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
        # Restore the persisted fractal values from the DB row rather than using
        # the recomputed ones from __init__.  Callers that need the *current*
        # classification (e.g. retrieve()) must call node._recompute() explicitly.
        node.classification = row["classification"]
        node.c_value = complex(row["c_real"], row["c_imag"])
        node.iteration_count = row["iteration_count"]
        return node

    def get_all(self) -> list[MemoryNode]:
        """Return all memories sorted by classification priority then frequency.

        Returns
        -------
        list[MemoryNode]
            All stored memories, ordered core → boundary → decay → forget,
            then by frequency descending within each bucket.
        """
        rows = self._conn.execute("SELECT * FROM memories").fetchall()
        freq_range, recency_range = self._get_ranges()
        nodes = []
        for row in rows:
            node = self._row_to_node(row, freq_range, recency_range)
            node._recompute()
            nodes.append(node)
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
