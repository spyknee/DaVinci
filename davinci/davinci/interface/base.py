"""
DaVinci Interface — Layer 3: Base Interface
===========================================
Abstract base class that defines the DaVinci interface contract.

This is the **extension point** for future interfaces.  Every user-facing
adapter — voice, REST, LLM agent, GUI — must subclass :class:`BaseInterface`
and implement its abstract methods.  The :class:`~davinci.interface.api.DaVinci`
class is the reference implementation shipped with Layer 3.

Layer 4 territory: drop-in voice/REST/LLM interfaces by subclassing this class
and registering them with the application without touching core logic.

No external dependencies — pure Python + stdlib only.
"""

from __future__ import annotations

import abc
from typing import Any

from davinci.core.fractal_engine import MemoryNode

__all__ = ["BaseInterface"]


class BaseInterface(abc.ABC):
    """Abstract contract for all DaVinci user-facing interfaces.

    Subclass this to build voice assistants, REST APIs, LLM tool adapters,
    or any other interface on top of the DaVinci fractal memory engine.

    All methods mirror the public API of
    :class:`~davinci.interface.api.DaVinci` so that interfaces are
    interchangeable at the call-site.
    """

    # ------------------------------------------------------------------
    # Memory operations
    # ------------------------------------------------------------------

    @abc.abstractmethod
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
        zoom_levels: Optional ``{1: ..., 2: ..., 3: ...}`` detail levels.
        meta:        Optional JSON-serialisable metadata dict.

        Returns
        -------
        str
            UUID of the newly stored memory.
        """

    @abc.abstractmethod
    def recall(self, memory_id: str) -> MemoryNode | None:
        """Retrieve a memory by its UUID and increment its access count.

        Parameters
        ----------
        memory_id: UUID string of the target memory.

        Returns
        -------
        MemoryNode | None
            The updated node, or ``None`` if no memory with that ID exists.
        """

    @abc.abstractmethod
    def search(self, query: str, limit: int = 10) -> list[MemoryNode]:
        """Search memories by content substring.

        Parameters
        ----------
        query: Substring to search for (case-insensitive).
        limit: Maximum number of results.

        Returns
        -------
        list[MemoryNode]
        """

    @abc.abstractmethod
    def forget(self, classification: str = "forget") -> int:
        """Prune memories by classification.

        Parameters
        ----------
        classification: Bucket to delete (default ``"forget"``).

        Returns
        -------
        int
            Number of memories deleted.
        """

    @abc.abstractmethod
    def decay(self) -> dict[str, list[str]]:
        """Run a decay cycle, reclassifying all memories.

        Returns
        -------
        dict[str, list[str]]
            Mapping of new classification → list of memory IDs that moved to it.
        """

    @abc.abstractmethod
    def consolidate(self, strategy: str = "frequency") -> int:
        """Run the consolidation engine.

        Parameters
        ----------
        strategy: Consolidation strategy (default ``"frequency"``).

        Returns
        -------
        int
            Number of memories updated.
        """

    @abc.abstractmethod
    def merge_similar(self, threshold: float = 0.8) -> int:
        """Merge memories whose content is highly similar.

        Parameters
        ----------
        threshold: Minimum Jaccard word-overlap to trigger a merge (0–1).

        Returns
        -------
        int
            Number of merges performed.
        """

    @abc.abstractmethod
    def stats(self) -> dict[str, Any]:
        """Return aggregate statistics about the memory store.

        Returns
        -------
        dict
            Keys: ``total``, ``by_classification``, ``avg_frequency``,
            ``oldest_timestamp``, ``newest_timestamp``.
        """

    @abc.abstractmethod
    def memories(self, classification: str | None = None) -> list[MemoryNode]:
        """Return all memories, optionally filtered by classification.

        Parameters
        ----------
        classification: If given, return only memories in that bucket.

        Returns
        -------
        list[MemoryNode]
        """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def close(self) -> None:
        """Release any resources held by this interface."""
