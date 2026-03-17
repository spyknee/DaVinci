"""
DaVinci LLM — LM Studio client
"""

from __future__ import annotations

import warnings
from typing import Generator

from lmstudio import Client, Chat

from davinci.llm.config import (
    LARGE_MODEL_HINTS,
    LMS_HOST,
    LMS_PORT,
    SMALL_MODEL_HINTS,
)
from davinci.memory.store import MemoryStore

__all__ = ["LMStudioClient"]


class LMStudioClient:
    """Streaming LLM client backed by LM Studio."""

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

    def _negotiate_model(self) -> None:
        loaded = list(self._client.llm.list_loaded())
        if not loaded:
            raise RuntimeError(
                "No model loaded in LM Studio. Load a model and retry."
            )

        if len(loaded) > 1:
            names = [getattr(m, "displayName", None) or m.identifier for m in loaded]
            warnings.warn(f"Multiple models loaded: {names}. Using {names[0]}.")

        model = loaded[0]
        self.model_id: str = model.identifier
        self.model_name: str = getattr(model, "displayName", None) or model.identifier
        self._detect_size()

    def _detect_size(self) -> None:
        name_lower = self.model_name.lower()
        if any(hint in name_lower for hint in LARGE_MODEL_HINTS):
            self.model_size = "large"
        elif any(hint in name_lower for hint in SMALL_MODEL_HINTS):
            self.model_size = "small"
        else:
            self.model_size = "unknown"

    def warn_if_wrong_size(self, expected: str) -> None:
        if self.model_size != expected and self.model_size != "unknown":
            warnings.warn(
                f"Task expects a {expected} model but {self.model_name!r} is {self.model_size}."
            )

    def refresh_model(self) -> None:
        self._negotiate_model()

    def _build_chat(self, system: str, user: str) -> Chat:
        chat = Chat(system)
        chat.add_user_message(user)
        return chat

    def _stream(self, chat: Chat) -> Generator[str, None, None]:
        for chunk in self._client.llm.model(self.model_id).respond_stream(chat):
            yield chunk.content

    def _stream_collect(self, chat: Chat) -> tuple[Generator[str, None, None], list[str]]:
        """Stream and collect into a list simultaneously."""
        accumulated: list[str] = []

        def _gen() -> Generator[str, None, None]:
            for chunk in self._client.llm.model(self.model_id).respond_stream(chat):
                accumulated.append(chunk.content)
                yield chunk.content

        return _gen(), accumulated

    def ingest(self, text: str) -> Generator[str, None, None]:
        self.warn_if_wrong_size("small")
        chat = self._build_chat(
            "You are a memory assistant. Summarise the following into a concise memory entry. Be factual and brief.",
            text,
        )
        accumulated: list[str] = []
        for chunk in self._client.llm.model(self.model_id).respond_stream(chat):
            accumulated.append(chunk.content)
            yield chunk.content

        full_response = "".join(accumulated)
        memory_id = self._store.store(full_response)
        yield f"\n[memory:{memory_id}]"

    def reason(self, query: str, limit: int = 5) -> Generator[str, None, None]:
        self.warn_if_wrong_size("large")
        nodes = self._store.search(query, limit=limit)
        if not nodes:
            yield "(No relevant memories found.)"
            return

        context_block = "\n".join(
            node.zoom_levels.get(3, node.content) for node in nodes
        )
        chat = self._build_chat(
            "You are a reasoning assistant with access to a memory store. "
            "Use the provided memories to answer accurately. "
            "If the memories do not contain relevant information, say so.",
            f"Memories:\n{context_block}\n\nQuery: {query}",
        )
        yield from self._stream(chat)

    def chat(
        self,
        user_message: str,
        context_limit: int = 5,
    ) -> Generator[str, None, None]:
        nodes = self._store.search(user_message, limit=context_limit)
        if nodes:
            context_block = "\n".join(
                node.zoom_levels.get(3, node.content) for node in nodes
            )
            memory_context = f"Memories from previous sessions:\n{context_block}\n\n"
        else:
            memory_context = ""

        chat = self._build_chat(
            "You are DaVinci, a persistent AI assistant. "
            "You have access to memories from previous sessions. "
            "Use them to maintain continuity across sessions. "
            "Be concise and direct.",
            f"{memory_context}{user_message}",
        )

        accumulated: list[str] = []
        for chunk in self._client.llm.model(self.model_id).respond_stream(chat):
            accumulated.append(chunk.content)
            yield chunk.content

        assistant_response = "".join(accumulated)
        exchange = f"User: {user_message}\nAssistant: {assistant_response}"

        summary_chat = self._build_chat(
            "You are a memory assistant. Summarise the following conversation exchange into a concise memory entry. Be factual and brief.",
            exchange,
        )
        summary_parts: list[str] = []
        for chunk in self._client.llm.model(self.model_id).respond_stream(summary_chat):
            summary_parts.append(chunk.content)

        summary = "".join(summary_parts)
        memory_id = self._store.store(summary)
        yield f"\n[memory:{memory_id}]"

    def __enter__(self) -> "LMStudioClient":
        return self

    def __exit__(self, *_: object) -> None:
        pass