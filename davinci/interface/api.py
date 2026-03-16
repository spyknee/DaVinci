"""
DaVinci Interface — Layer 3: Unified Python API
================================================
The single entry-point class that wraps Layer 1 (fractal engine) and
Layer 2 (memory store + consolidation) into a clean, ergonomic API.

Usage::

    from davinci import DaVinci

    with DaVinci(":memory:") as dv:
        mid = dv.remember("The Mandelbrot set is infinite.")
        node = dv.recall(mid)
        print(node.classification)

No external dependencies — pure Python + stdlib only.
"""

from __future__ import annotations

from typing import Any

from davinci.core.fractal_engine import MemoryNode
from davinci.interface.base import BaseInterface
from davinci.memory.consolidation import ConsolidationEngine
from davinci.memory.store import MemoryStore

__all__ = ["DaVinci"]


class DaVinci(BaseInterface):
    """Unified Python API for the DaVinci fractal memory system.

    Wraps :class:`~davinci.memory.store.MemoryStore` and
    :class:`~davinci.memory.consolidation.ConsolidationEngine` so callers
    never need to touch the lower layers directly.

    Parameters
    ----------
    db_path:  Path to the SQLite database file.  Use ``":memory:"`` for an
              ephemeral in-process store (useful in tests).
    max_iter: Mandelbrot iteration limit passed to the fractal engine.

    Examples
    --------
    >>> dv = DaVinci(":memory:")
    >>> mid = dv.remember("Hello, DaVinci!")
    >>> node = dv.recall(mid)
    >>> print(node.classification)
    forget
    >>> dv.close()
    """

    def __init__(self, db_path: str = "davinci_memory.db", max_iter: int = 1000) -> None:
        self._store = MemoryStore(db_path=db_path, max_iter=max_iter)
        self._engine = ConsolidationEngine(self._store)

    # ------------------------------------------------------------------
    # Memory operations
    # ------------------------------------------------------------------

    def remember(
        self,
        content: str,
        zoom_levels: dict | None = None,
        meta: dict | None = None,
    ) -> str:
        """Store a new memory and return its UUID.

        Parameters
        ----------
        content:     The text content to remember.
        zoom_levels: Optional detail levels — keys may be ints or strings
                     (``{1: "summary", 2: "detail", 3: "full"}``).
        meta:        Optional JSON-serialisable metadata dict.

        Returns
        -------
        str
            UUID of the newly stored memory.
        """
        zl: dict[int, str] | None = None
        if zoom_levels is not None:
            zl = {int(k): v for k, v in zoom_levels.items()}
        return self._store.store(content, zoom_levels=zl, meta=meta)

    def recall(self, memory_id: str) -> MemoryNode | None:
        """Retrieve a memory by ID and increment its access count.

        Parameters
        ----------
        memory_id: UUID string of the target memory.

        Returns
        -------
        MemoryNode | None
            The updated node, or ``None`` if the ID does not exist.
        """
        return self._store.retrieve(memory_id)

    def search(self, query: str, limit: int = 10) -> list[MemoryNode]:
        """Search memories by content substring (case-insensitive).

        Parameters
        ----------
        query: Substring to search for.
        limit: Maximum number of results (default 10).

        Returns
        -------
        list[MemoryNode]
            Matching nodes ordered by classification priority then frequency.
        """
        return self._store.search(query, limit=limit)

    def forget(self, classification: str = "forget") -> int:
        """Delete all memories with the given classification.

        Parameters
        ----------
        classification: Bucket to prune (default ``"forget"``).

        Returns
        -------
        int
            Number of memories deleted.
        """
        return self._store.prune(classification)

    def decay(self) -> dict[str, list[str]]:
        """Reclassify all memories using current global access ranges.

        Returns
        -------
        dict[str, list[str]]
            Mapping ``{new_classification: list_of_memory_ids_moved}``.
        """
        return self._store.decay_cycle()

    def consolidate(self, strategy: str = "frequency") -> int:
        """Run the consolidation engine.

        Parameters
        ----------
        strategy: Consolidation strategy — currently only ``"frequency"``.

        Returns
        -------
        int
            Count of memories promoted / updated.
        """
        return self._engine.consolidate(strategy)

    def merge_similar(self, threshold: float = 0.8) -> int:
        """Merge memories whose Jaccard word-overlap exceeds *threshold*.

        Parameters
        ----------
        threshold: Similarity threshold (0–1, default 0.8).

        Returns
        -------
        int
            Number of merges performed.
        """
        return self._engine.merge_similar(threshold)

    def stats(self) -> dict[str, Any]:
        """Return aggregate statistics about the memory store.

        Returns
        -------
        dict
            Keys: ``total``, ``by_classification``, ``avg_frequency``,
            ``oldest_timestamp``, ``newest_timestamp``.
        """
        return self._store.stats()

    def memories(self, classification: str | None = None) -> list[MemoryNode]:
        """Return all memories, optionally filtered by classification.

        Parameters
        ----------
        classification: If given, return only memories in that bucket
                        (``"core"``, ``"boundary"``, ``"decay"``,
                        ``"forget"``).  If ``None``, return all memories.

        Returns
        -------
        list[MemoryNode]
        """
        if classification is not None:
            return self._store.get_by_classification(classification)
        return self._store.get_all()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database connection."""
        self._store.close()

    def __enter__(self) -> "DaVinci":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
