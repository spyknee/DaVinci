"""
DaVinci LLM — LM Studio client
================================
Wraps the ``lmstudio-python`` SDK to provide a streaming, bi-directional
pipeline between DaVinci's fractal memory store and a locally running
LM Studio instance.

Usage::

    from davinci.llm import LMStudioClient
    from davinci.memory.store import MemoryStore

    store = MemoryStore(":memory:")
    with LMStudioClient(store=store) as client:
        for chunk in client.ingest("The Mandelbrot set is fractal."):
            print(chunk, end="", flush=True)
"""

from __future__ import annotations

import warnings
from typing import Generator

from lmstudio import Client

from davinci.llm.config import (
    LARGE_MODEL_HINTS,
    LMS_HOST,
    LMS_PORT,
    SMALL_MODEL_HINTS,
)
from davinci.memory.store import MemoryStore

__all__ = ["LMStudioClient"]


class LMStudioClient:
    """Streaming LLM client backed by LM Studio.

    Parameters
    ----------
    store:
        Injected :class:`~davinci.memory.store.MemoryStore` instance.
        ``LMStudioClient`` never creates its own store.
    host:
        LM Studio host (default: ``LMS_HOST`` from config).
    port:
        LM Studio port (default: ``LMS_PORT`` from config).
    """

    def __init__(
        self,
        store: MemoryStore,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        self._store = store
        self._host = host or LMS_HOST
        self._port = port or LMS_PORT
        self._client = Client(f"{self._host}:{self._port}")
        self._negotiate_model()

    # ------------------------------------------------------------------
    # Model negotiation
    # ------------------------------------------------------------------

    def _negotiate_model(self) -> None:
        """Detect the active model loaded in LM Studio.

        Sets :attr:`model_id`, :attr:`model_name`, and :attr:`model_size`.

        Raises
        ------
        RuntimeError
            If no model is currently loaded.
        """
        loaded = list(self._client.llm.list_loaded())
        if not loaded:
            raise RuntimeError(
                "No model loaded in LM Studio. Load a model and retry."
            )

        if len(loaded) > 1:
            names = [m.displayName for m in loaded]
            first = names[0]
            warnings.warn(
                f"Multiple models loaded: {names}. Using {first}."
            )

        model = loaded[0]
        self.model_id: str = model.identifier
        self.model_name: str = model.displayName
        self._detect_size()

    def _detect_size(self) -> None:
        """Classify ``self.model_name`` as ``"large"``, ``"small"``, or ``"unknown"``."""
        name_lower = self.model_name.lower()
        if any(hint in name_lower for hint in LARGE_MODEL_HINTS):
            self.model_size = "large"
        elif any(hint in name_lower for hint in SMALL_MODEL_HINTS):
            self.model_size = "small"
        else:
            self.model_size = "unknown"

    def warn_if_wrong_size(self, expected: str) -> None:
        """Emit a warning when the loaded model size does not match *expected*.

        Parameters
        ----------
        expected:
            ``"large"`` or ``"small"``.
        """
        if self.model_size != expected and self.model_size != "unknown":
            warnings.warn(
                f"Task expects a {expected} model but {self.model_name!r} is {self.model_size}."
            )

    def refresh_model(self) -> None:
        """Re-detect the active model (call after swapping models in LM Studio)."""
        self._negotiate_model()

    # ------------------------------------------------------------------
    # Pipeline — ingest (text → memory)
    # ------------------------------------------------------------------

    def ingest(self, text: str) -> Generator[str, None, None]:
        """Summarise *text* via LM Studio and store the result as a memory.

        Streams each token chunk to the caller. After the stream ends, the
        full accumulated response is stored in the memory store and a final
        sentinel ``"\\n[memory:<uuid>]"`` is yielded so callers know the ID.

        Parameters
        ----------
        text:
            Raw text to summarise and remember.

        Yields
        ------
        str
            Token chunks from the LLM, followed by ``"\\n[memory:<uuid>]"``.
        """
        self.warn_if_wrong_size("small")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a memory assistant. Summarise the following into a "
                    "concise memory entry. Be factual and brief."
                ),
            },
            {"role": "user", "content": text},
        ]

        accumulated: list[str] = []
        for chunk in self._client.llm.respond_stream(
            self.model_id, messages=messages
        ):
            token = chunk.content
            accumulated.append(token)
            yield token

        full_response = "".join(accumulated)
        memory_id = self._store.store(full_response)
        yield f"\n[memory:{memory_id}]"

    # ------------------------------------------------------------------
    # Pipeline — reason (query → LLM grounded on memories)
    # ------------------------------------------------------------------

    def reason(self, query: str, limit: int = 5) -> Generator[str, None, None]:
        """Answer *query* using memories retrieved from the store.

        Streams each token chunk to the caller.

        Parameters
        ----------
        query:
            The question or prompt to reason about.
        limit:
            Maximum number of memories to include as context (default 5).

        Yields
        ------
        str
            Token chunks from the LLM.
        """
        self.warn_if_wrong_size("large")

        nodes = self._store.search(query, limit=limit)
        if not nodes:
            yield "(No relevant memories found.)"
            return

        context_block = "\n".join(
            node.zoom_levels.get(3, node.content) for node in nodes
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a reasoning assistant with access to a memory store. "
                    "Use the provided memories to answer accurately. "
                    "If the memories do not contain relevant information, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Memories:\n{context_block}\n\nQuery: {query}",
            },
        ]

        for chunk in self._client.llm.respond_stream(
            self.model_id, messages=messages
        ):
            yield chunk.content

    # ------------------------------------------------------------------
    # Pipeline — chat (full persistent conversation turn)
    # ------------------------------------------------------------------

    def chat(
        self,
        user_message: str,
        context_limit: int = 5,
    ) -> Generator[str, None, None]:
        """Full persistent conversation pipeline.

        Steps:
        1. Retrieve relevant memories for the user message.
        2. Build context block from full memory content (zoom_level_3).
        3. Send to LLM with a continuity-focused system prompt.
        4. Stream response tokens to caller.
        5. After stream ends, summarise and store the exchange as a memory
           via the same LLM summarisation used by :meth:`ingest`.

        Parameters
        ----------
        user_message:
            The user's input for this turn.
        context_limit:
            Maximum number of memories to inject as context (default 5).

        Yields
        ------
        str
            Token chunks from the LLM, followed by ``"\\n[memory:<uuid>]"``
            sentinel once the exchange is stored.
        """
        # Step 1 & 2: retrieve memories and build context
        nodes = self._store.search(user_message, limit=context_limit)
        if nodes:
            context_block = "\n".join(
                node.zoom_levels.get(3, node.content) for node in nodes
            )
            memory_context = f"Memories from previous sessions:\n{context_block}\n\n"
        else:
            memory_context = ""

        # Step 3 & 4: stream LLM response
        messages = [
            {
                "role": "system",
                "content": (
                    "You are DaVinci, a persistent AI assistant. "
                    "You have access to memories from previous sessions. "
                    "Use them to maintain continuity across sessions. "
                    "Be concise and direct."
                ),
            },
            {
                "role": "user",
                "content": f"{memory_context}{user_message}",
            },
        ]

        accumulated: list[str] = []
        for chunk in self._client.llm.respond_stream(
            self.model_id, messages=messages
        ):
            token = chunk.content
            accumulated.append(token)
            yield token

        # Step 5: summarise the full exchange and store as a memory
        assistant_response = "".join(accumulated)
        exchange = f"User: {user_message}\nAssistant: {assistant_response}"

        summarise_messages = [
            {
                "role": "system",
                "content": (
                    "You are a memory assistant. Summarise the following conversation "
                    "exchange into a concise memory entry. Be factual and brief."
                ),
            },
            {"role": "user", "content": exchange},
        ]

        summary_parts: list[str] = []
        for chunk in self._client.llm.respond_stream(
            self.model_id, messages=summarise_messages
        ):
            summary_parts.append(chunk.content)

        summary = "".join(summary_parts)
        memory_id = self._store.store(summary)
        yield f"\n[memory:{memory_id}]"

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "LMStudioClient":
        return self

    def __exit__(self, *_: object) -> None:
        pass
