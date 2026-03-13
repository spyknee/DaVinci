"""
DaVinci Interface — Layer 3: Unified Python API
================================================
The single entry-point class that wraps Layer 1 (fractal engine) and
Layer 2 (memory store + consolidation) into a clean, ergonomic API.

Layer 5A extends the API with LLM integration, episodic memory,
auto-zoom, auto-learn, FTS5 search, and model management.

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
from davinci.memory.episodic import EpisodicStore
from davinci.memory.store import MemoryStore

__all__ = ["DaVinci"]


class DaVinci(BaseInterface):
    """Unified Python API for the DaVinci fractal memory system.

    Wraps :class:`~davinci.memory.store.MemoryStore`,
    :class:`~davinci.memory.consolidation.ConsolidationEngine`,
    :class:`~davinci.memory.episodic.EpisodicStore`, and the LLM layer
    so callers never need to touch the lower layers directly.

    Parameters
    ----------
    db_path:      Path to the SQLite database file.  Use ``":memory:"`` for an
                  ephemeral in-process store (useful in tests).
    max_iter:     Mandelbrot iteration limit passed to the fractal engine.
    profile_path: Optional path to a JSON profile file for LLM configuration.
    llm_enabled:  Set to ``False`` to disable LLM integration entirely.

    Examples
    --------
    >>> dv = DaVinci(":memory:")
    >>> mid = dv.remember("Hello, DaVinci!")
    >>> node = dv.recall(mid)
    >>> print(node.classification)
    forget
    >>> dv.close()
    """

    def __init__(
        self,
        db_path: str = "davinci_memory.db",
        max_iter: int = 1000,
        profile_path: str | None = None,
        llm_enabled: bool = True,
    ) -> None:
        self._store = MemoryStore(db_path=db_path, max_iter=max_iter)
        self._engine = ConsolidationEngine(self._store)
        self._episodic = EpisodicStore(db_path=db_path)

        # LLM layer (optional)
        self._model_manager = None
        self._auto_zoom = None
        self._auto_learn = None
        self._conversation: list[dict] = []  # working memory for LLM

        if llm_enabled:
            self._setup_llm(profile_path)

    def _setup_llm(self, profile_path: str | None) -> None:
        """Initialise ModelManager, AutoZoom and AutoLearn."""
        try:
            from davinci.llm.manager import ModelManager
            from davinci.llm.auto_zoom import AutoZoom
            from davinci.llm.auto_learn import AutoLearn
            from davinci.llm.profile import Profile

            profile = Profile(profile_path)
            llm_cfg = profile.llm_config()
            self._model_manager = ModelManager(profile=llm_cfg if llm_cfg else None)
            backend = self._model_manager.active()
            self._auto_zoom = AutoZoom(backend)
            self._auto_learn = AutoLearn(
                self._store, llm_backend=backend, auto_zoom=self._auto_zoom
            )
        except Exception:  # noqa: BLE001
            # LLM layer is optional — degrade gracefully
            self._model_manager = None
            self._auto_zoom = None
            self._auto_learn = None

    # ------------------------------------------------------------------
    # Memory operations (existing API — unchanged)
    # ------------------------------------------------------------------

    def remember(
        self,
        content: str,
        zoom_levels: dict | None = None,
        meta: dict | None = None,
    ) -> str:
        """Store a new memory and return its UUID.

        If AutoZoom is available and no *zoom_levels* are provided, zoom
        levels are generated automatically via the LLM.

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
        elif self._auto_zoom is not None:
            levels = self._auto_zoom.generate_zoom_levels(content)
            zl = {int(k): v for k, v in levels.items()}
        return self._store.store(content, zoom_levels=zl, meta=meta)

    def recall(self, memory_id: str) -> MemoryNode | None:
        """Retrieve a memory by ID and increment its access count."""
        return self._store.retrieve(memory_id)

    def search(self, query: str, limit: int = 10) -> list[MemoryNode]:
        """Search memories by content substring (case-insensitive)."""
        return self._store.search(query, limit=limit)

    def search_fts(self, query: str, limit: int = 10) -> list[MemoryNode]:
        """Search memories using FTS5 full-text search.

        Falls back to ``LIKE`` search if FTS5 is unavailable.

        Parameters
        ----------
        query: Full-text search query string.
        limit: Maximum number of results (default 10).

        Returns
        -------
        list[MemoryNode]
        """
        return self._store.search_fts(query, limit=limit)

    def forget(self, classification: str = "forget") -> int:
        """Delete all memories with the given classification."""
        return self._store.prune(classification)

    def decay(self) -> dict[str, int]:
        """Reclassify all memories using current global access ranges."""
        return self._store.decay_cycle()

    def consolidate(self, strategy: str = "frequency") -> int:
        """Run the consolidation engine."""
        return self._engine.consolidate(strategy)

    def merge_similar(self, threshold: float = 0.8) -> int:
        """Merge memories whose Jaccard word-overlap exceeds *threshold*."""
        return self._engine.merge_similar(threshold)

    def stats(self) -> dict[str, Any]:
        """Return aggregate statistics about the memory store."""
        return self._store.stats()

    def migrate(self) -> dict[str, list[str]]:
        """Identify and update memories whose classification has drifted."""
        return self._store.migrate()

    def memories(self, classification: str | None = None) -> list[MemoryNode]:
        """Return all memories, optionally filtered by classification."""
        if classification is not None:
            return self._store.get_by_classification(classification)
        return self._store.get_all()

    # ------------------------------------------------------------------
    # LLM operations (Layer 5A)
    # ------------------------------------------------------------------

    def ask(self, question: str, system_prompt: str | None = None) -> str:
        """Full LLM-powered answer pipeline.

        Steps:
        1. FTS5 search for memory context.
        2. Retrieve episodic context.
        3. Build system prompt with memory context.
        4. Send to LLM with conversation history.
        5. Save to episodic memory.
        6. Run auto-learn extraction.

        Parameters
        ----------
        question:      The user's question.
        system_prompt: Optional custom system prompt (overrides default).

        Returns
        -------
        str
            The LLM's response, or an error message if unavailable.
        """
        if self._model_manager is None:
            return "[LLM unavailable: llm_enabled=False or LLM layer failed to initialise]"

        backend = self._model_manager.active()

        # 1. Search memories for context
        memory_hits = self._store.search_fts(question, limit=5)
        memory_context = "\n".join(
            f"- {n.zoom_levels.get(2, n.content[:100])}" for n in memory_hits
        )

        # 2. Retrieve episodic context
        episodic_hits = self._episodic.retrieve(question, limit=2)
        episodic_context = "\n".join(
            f"Q: {e['question']}\nA: {e['answer']}" for e in episodic_hits
        )

        # 3. Build system prompt
        if system_prompt is None:
            parts = ["You are DaVinci, a knowledgeable AI assistant."]
            if memory_context:
                parts.append(f"\nRelevant memories:\n{memory_context}")
            if episodic_context:
                parts.append(f"\nRecent conversation context:\n{episodic_context}")
            system_prompt = "\n".join(parts)

        # 4. Build messages list with conversation history
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        messages.extend(self._conversation[-20:])  # last 20 turns
        messages.append({"role": "user", "content": question})

        answer = backend.chat(messages)

        # Update conversation history
        self._conversation.append({"role": "user", "content": question})
        self._conversation.append({"role": "assistant", "content": answer})

        # 5. Save to episodic memory
        self._episodic.save(question, answer)

        # 6. Auto-learn (non-blocking — queue for review by default)
        if self._auto_learn is not None:
            self._auto_learn.learn(question, answer, auto_store=False)

        return answer

    def model_switch(self, name: str) -> bool:
        """Switch to a different LLM model.

        Parameters
        ----------
        name: Model key name (e.g. ``"qwen35"``).

        Returns
        -------
        bool
            ``True`` on success, ``False`` if unavailable or unknown.
        """
        if self._model_manager is None:
            return False
        result = self._model_manager.switch(name)
        if result:
            self._refresh_llm_references()
        return result

    def model_toggle(self) -> str:
        """Cycle to the next configured LLM model.

        Returns
        -------
        str
            Key name of the newly active model, or ``""`` if unavailable.
        """
        if self._model_manager is None:
            return ""
        new_name = self._model_manager.toggle()
        self._refresh_llm_references()
        return new_name

    def model_status(self) -> dict:
        """Return status information for the active LLM model.

        Returns
        -------
        dict
            Keys: ``active``, ``model``, ``base_url``, ``available``.
            Returns empty dict if LLM is disabled.
        """
        if self._model_manager is None:
            return {}
        return self._model_manager.status()

    def _refresh_llm_references(self) -> None:
        """Update AutoZoom and AutoLearn to use the newly active backend."""
        if self._model_manager is None:
            return
        from davinci.llm.auto_zoom import AutoZoom
        from davinci.llm.auto_learn import AutoLearn

        backend = self._model_manager.active()
        self._auto_zoom = AutoZoom(backend)
        self._auto_learn = AutoLearn(
            self._store, llm_backend=backend, auto_zoom=self._auto_zoom
        )

    # ------------------------------------------------------------------
    # Episodic memory operations (Layer 5A)
    # ------------------------------------------------------------------

    def episodic_status(self) -> dict:
        """Return aggregate statistics about the episodic memory store."""
        return self._episodic.status()

    def episodic_decay(self, rate: float = 0.05) -> int:
        """Run importance decay on episodic memories.

        Parameters
        ----------
        rate: Fraction of importance lost per day of inactivity.

        Returns
        -------
        int
            Number of entries updated.
        """
        return self._episodic.decay(rate_per_day=rate)

    def episodic_prune(self, threshold: float = 0.2) -> int:
        """Prune episodic memories whose importance is below *threshold*.

        Parameters
        ----------
        threshold: Minimum importance to keep.

        Returns
        -------
        int
            Number of entries deleted.
        """
        return self._episodic.prune(threshold=threshold)

    # ------------------------------------------------------------------
    # Auto-learn operations (Layer 5A)
    # ------------------------------------------------------------------

    def review_pending(self) -> list[dict]:
        """Return the list of pending auto-learn facts awaiting approval."""
        if self._auto_learn is None:
            return []
        return self._auto_learn.pending

    def approve_fact(self, index: int) -> bool:
        """Approve and store a pending auto-learn fact.

        Parameters
        ----------
        index: Zero-based index into the pending list.

        Returns
        -------
        bool
        """
        if self._auto_learn is None:
            return False
        return self._auto_learn.approve(index)

    def reject_fact(self, index: int) -> bool:
        """Reject and discard a pending auto-learn fact.

        Parameters
        ----------
        index: Zero-based index into the pending list.

        Returns
        -------
        bool
        """
        if self._auto_learn is None:
            return False
        return self._auto_learn.reject(index)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database connections."""
        self._store.close()
        self._episodic.close()

    def __enter__(self) -> "DaVinci":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
