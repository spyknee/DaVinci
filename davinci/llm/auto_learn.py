"""
DaVinci LLM — Auto-Learn Pipeline
===================================
Extract facts from LLM conversations and store them as memories.

Inspired by Gilligan's Section 15 (auto-learn from conversations) but
implemented with DaVinci's clean architecture.

No external dependencies — pure Python + stdlib only.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from davinci.llm.backend import LLMBackend

if TYPE_CHECKING:
    from davinci.llm.auto_zoom import AutoZoom
    from davinci.memory.store import MemoryStore

__all__ = ["AutoLearn"]

_EXTRACT_PROMPT = """\
Extract the key facts from the following question and answer exchange.
Return a JSON array of concise factual statements (strings).
Each fact should be a standalone, self-contained sentence.
Return ONLY valid JSON — no markdown fences, no commentary.

Question: {question}
Answer: {answer}"""


class AutoLearn:
    """Extracts facts from Q&A exchanges and stores them as memories.

    Parameters
    ----------
    store:      :class:`~davinci.memory.store.MemoryStore` instance where
                approved facts are persisted.
    llm_backend: Optional LLM backend for AI-powered fact extraction.
                 When ``None``, a simple heuristic is used instead.
    auto_zoom:  Optional :class:`~davinci.llm.auto_zoom.AutoZoom` instance
                used to generate zoom levels when storing facts.

    Examples
    --------
    >>> learn = AutoLearn(store)
    >>> facts = learn.learn("What is Python?", "Python is a high-level language.")
    >>> len(facts) > 0
    True
    """

    def __init__(
        self,
        store: "MemoryStore",
        llm_backend: "LLMBackend | None" = None,
        auto_zoom: "AutoZoom | None" = None,
    ) -> None:
        self._store = store
        self._llm = llm_backend
        self._auto_zoom = auto_zoom
        self._pending: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Fact extraction
    # ------------------------------------------------------------------

    def extract_facts(self, question: str, answer: str) -> list[str]:
        """Extract key facts from a question/answer pair.

        Uses the LLM when available; falls back to a simple heuristic
        (sentences longer than 60 characters ending with a period).

        Parameters
        ----------
        question: The user's question.
        answer:   The LLM's answer.

        Returns
        -------
        list[str]
            Extracted fact strings.
        """
        if self._llm is not None:
            return self._extract_with_llm(question, answer)
        return self._extract_heuristic(answer)

    def _extract_with_llm(self, question: str, answer: str) -> list[str]:
        prompt = _EXTRACT_PROMPT.format(question=question, answer=answer)
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = self._llm.chat(messages, max_tokens=400, temperature=0.3)
        except Exception:  # noqa: BLE001
            return self._extract_heuristic(answer)

        if raw.startswith("[LLM unavailable"):
            return self._extract_heuristic(answer)

        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(
                line for line in clean.splitlines() if not line.startswith("```")
            ).strip()

        try:
            data = json.loads(clean)
            if isinstance(data, list):
                return [str(f) for f in data if f]
        except (json.JSONDecodeError, TypeError):
            pass

        return self._extract_heuristic(answer)

    @staticmethod
    def _extract_heuristic(text: str) -> list[str]:
        """Simple heuristic: sentences > 60 chars ending with a period."""
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        facts = []
        for s in sentences:
            full = s + "."
            if len(full) > 60:
                facts.append(full)
        return facts

    # ------------------------------------------------------------------
    # Learn pipeline
    # ------------------------------------------------------------------

    def learn(
        self,
        question: str,
        answer: str,
        auto_store: bool = False,
    ) -> list[dict]:
        """Extract facts and optionally store them immediately.

        Parameters
        ----------
        question:   The user's question.
        answer:     The LLM's answer.
        auto_store: If ``True``, facts are stored in the memory store
                    immediately.  If ``False``, they are queued as pending
                    candidates for manual review.

        Returns
        -------
        list[dict]
            List of ``{"fact": str, "question": str, "timestamp": float}``
            dicts for each extracted fact.
        """
        facts = self.extract_facts(question, answer)
        now = time.time()
        result: list[dict] = []

        for fact_text in facts:
            entry: dict = {"fact": fact_text, "question": question, "timestamp": now}
            result.append(entry)
            if auto_store:
                self._store_fact(fact_text)
            else:
                self._pending.append(entry)

        return result

    def _store_fact(self, fact: str) -> str:
        """Persist a fact in the memory store, using auto-zoom if available."""
        zoom_levels: dict[int, str] | None = None
        if self._auto_zoom is not None:
            levels = self._auto_zoom.generate_zoom_levels(fact)
            zoom_levels = {int(k): v for k, v in levels.items()}
        return self._store.store(fact, zoom_levels=zoom_levels)

    # ------------------------------------------------------------------
    # Pending queue management
    # ------------------------------------------------------------------

    @property
    def pending(self) -> list[dict]:
        """List of unconfirmed (pending) facts awaiting approval."""
        return list(self._pending)

    def approve(self, index: int) -> bool:
        """Approve and store the pending fact at *index*.

        Parameters
        ----------
        index: Zero-based index into the pending list.

        Returns
        -------
        bool
            ``True`` on success, ``False`` if the index is out of range.
        """
        if index < 0 or index >= len(self._pending):
            return False
        entry = self._pending.pop(index)
        self._store_fact(entry["fact"])
        return True

    def reject(self, index: int) -> bool:
        """Remove the pending fact at *index* without storing it.

        Parameters
        ----------
        index: Zero-based index into the pending list.

        Returns
        -------
        bool
            ``True`` on success, ``False`` if the index is out of range.
        """
        if index < 0 or index >= len(self._pending):
            return False
        self._pending.pop(index)
        return True

    def approve_all(self) -> int:
        """Approve and store all pending facts.

        Returns
        -------
        int
            Number of facts stored.
        """
        count = len(self._pending)
        for entry in self._pending:
            self._store_fact(entry["fact"])
        self._pending.clear()
        return count

    def purge(self) -> int:
        """Discard all pending facts without storing them.

        Returns
        -------
        int
            Number of facts discarded.
        """
        count = len(self._pending)
        self._pending.clear()
        return count
