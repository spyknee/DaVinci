"""
DaVinci Memory — Layer 2: Consolidation Engine
===============================================
Handles merging of related memories and promotion of frequently-accessed ones.

No external dependencies — pure Python + stdlib only.
"""

from __future__ import annotations

from davinci.memory.store import MemoryStore

__all__ = ["ConsolidationEngine"]

# Default threshold: memories accessed more than this many times get promoted
_DEFAULT_FREQUENCY_THRESHOLD = 5


class ConsolidationEngine:
    """Consolidates memories by promoting high-frequency nodes and merging
    similar content.

    Parameters
    ----------
    store: The :class:`~davinci.memory.store.MemoryStore` to operate on.

    Examples
    --------
    >>> with MemoryStore(":memory:") as store:
    ...     engine = ConsolidationEngine(store)
    ...     merged = engine.merge_similar(similarity_threshold=0.8)
    """

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def consolidate(self, strategy: str = "frequency") -> int:
        """Consolidate memories according to the chosen strategy.

        Strategies
        ----------
        ``"frequency"``
            Memories whose ``frequency`` exceeds the threshold have their
            zoom levels refined: ``zoom_level_1`` is set to a summary
            (first 50 chars), ``zoom_level_2`` to the first 100 chars, and
            ``zoom_level_3`` retains the full content.  This simulates
            memory *sharpening* — the more a node is accessed the richer
            its detail representation becomes.

        Parameters
        ----------
        strategy: Consolidation strategy (currently only ``"frequency"``).

        Returns
        -------
        int
            Count of memories that were promoted / updated.

        Raises
        ------
        ValueError
            If an unknown strategy name is provided.
        """
        if strategy == "frequency":
            return self._consolidate_by_frequency()
        raise ValueError(f"Unknown consolidation strategy: {strategy!r}")

    def merge_similar(self, similarity_threshold: float = 0.8) -> int:
        """Merge memories whose content is highly similar.

        Similarity is measured as the Jaccard overlap of word sets.  When
        two memories exceed *similarity_threshold*, the less-accessed one is
        deleted and its ``frequency`` is added to the survivor's count.

        Parameters
        ----------
        similarity_threshold: Minimum word-overlap ratio to trigger a merge
                              (0 – 1, default 0.8).

        Returns
        -------
        int
            Number of merges performed (i.e. number of records deleted).
        """
        conn = self._store._conn
        rows = conn.execute("SELECT id, content, frequency FROM memories").fetchall()

        # Index rows by id for O(1) lookup; use a set to track deleted ids
        deleted: set[str] = set()
        merge_count = 0

        for i in range(len(rows)):
            id_i = rows[i]["id"]
            if id_i in deleted:
                continue

            content_i = rows[i]["content"]
            freq_i = rows[i]["frequency"]
            words_i = set(content_i.lower().split())

            for j in range(i + 1, len(rows)):
                id_j = rows[j]["id"]
                if id_j in deleted:
                    continue

                content_j = rows[j]["content"]
                freq_j = rows[j]["frequency"]
                words_j = set(content_j.lower().split())

                similarity = _jaccard(words_i, words_j)
                if similarity >= similarity_threshold:
                    # Merge j into i: add frequencies, delete j
                    new_freq = freq_i + freq_j
                    freq_i = new_freq  # update local for subsequent comparisons

                    with conn:
                        conn.execute(
                            "UPDATE memories SET frequency = ? WHERE id = ?",
                            (new_freq, id_i),
                        )
                        conn.execute(
                            "DELETE FROM memories WHERE id = ?", (id_j,)
                        )

                    deleted.add(id_j)
                    merge_count += 1

        return merge_count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _consolidate_by_frequency(self, threshold: int = _DEFAULT_FREQUENCY_THRESHOLD) -> int:
        """Promote memories whose frequency exceeds *threshold*.

        Zoom levels are refined:
        - ``zoom_level_1`` → first 50 characters (summary view)
        - ``zoom_level_2`` → first 100 characters (detail view)
        - ``zoom_level_3`` → full content (complete view)

        Returns
        -------
        int
            Count of promoted memories.
        """
        conn = self._store._conn
        rows = conn.execute(
            "SELECT id, content FROM memories WHERE frequency > ?", (threshold,)
        ).fetchall()

        count = 0
        for row in rows:
            content = row["content"]
            z1 = content[:50]
            z2 = content[:100]
            z3 = content

            with conn:
                conn.execute(
                    """
                    UPDATE memories SET
                        zoom_level_1 = ?,
                        zoom_level_2 = ?,
                        zoom_level_3 = ?
                    WHERE id = ?
                    """,
                    (z1, z2, z3, row["id"]),
                )

            count += 1

        return count


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two word sets.

    Returns 1.0 when both sets are empty (identical empty content).
    """
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)
