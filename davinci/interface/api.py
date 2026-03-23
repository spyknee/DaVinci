"""DaVinci API layer — high-level operations on memory."""
from __future__ import annotations

import time
from typing import Optional, List, Dict, Any

from davinci.memory.store import MemoryStore
from davinci.memory.consolidation import ConsolidationEngine


class DaVinci:
    """Fractal-aware persistent memory manager."""

    def __init__(self, db_path: str = "davinci_memory.db"):
        self._store = MemoryStore(db_path)
        self._engine = ConsolidationEngine(self._store)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._store.close()

    # Core operations
    def remember(
        self,
        content: str,
        zoom_levels: Optional[Dict[int, str]] = None,
        meta: Optional[Dict[str, Any]] = None,
        speaker: Optional[str] = None,
        source: Optional[str] = "manual",
        tags: Optional[List[str]] = None,
    ) -> str:
        """Store a new memory."""
        # Convert provenance to meta dict (if provided)
        if meta is None:
            meta = {}
        if speaker:
            meta["speaker"] = speaker
        if source and source != "manual":
            meta["source"] = source
        if tags:
            meta["tags"] = tags

        return self._store.store(content=content, zoom_levels=zoom_levels, meta=meta)

    def recall(self, node_id: str) -> Optional[Any]:
        """Retrieve a memory by ID."""
        return self._store.retrieve(node_id)

    def search(self, query: str, limit: int = 10) -> List[Any]:
        """Search memories by content."""
        return self._store.search(query, limit)

    def forget(self, classification: str = "forget") -> int:
        """Delete all memories of a given classification."""
        return self._store.prune(classification)

    def decay(self, max_iter: Optional[int] = None) -> Dict[str, List[str]]:
        """
        Run fractal decay cycle — uses Julia-inspired retention modeling.
        
        Returns dict like: {"decay": ["id1"], "forget": ["id2"]}
        """
        return self._store.decay_cycle()

    def consolidate(self, strategy: str = "frequency") -> int:
        """Promote memories based on strategy."""
        if strategy == "frequency":
            count = 0
            try:
                count = self._engine.consolidate("frequency")
            except Exception:
                pass  # placeholder for future refinement
            return count
        raise ValueError(f"Unknown consolidation strategy: {strategy}")

    def merge_similar(self, threshold: float = 0.8) -> int:
        """Merge duplicate/similar memories."""
        return self._engine.merge_similar(similarity_threshold=threshold)

    def stats(self) -> Dict[str, Any]:
        """Return memory statistics."""
        return self._store.stats()

    def memories(
        self, classification: Optional[str] = None
    ) -> List[Any]:
        """List memories optionally filtered by classification."""
        if classification:
            return self._store.get_by_classification(classification)
        else:
            return self._store.get_all()
