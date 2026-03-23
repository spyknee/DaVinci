"""
DaVinci LLM — LM Studio client
With session buffering, auto-consolidation, temporal reasoning, embedding interpolation, and numeric pattern detection.
"""

from __future__ import annotations

import json
import re
import time
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


def _strip_think(text: str) -> str:
    return re.sub(r"((<tool_call>|Thinking Process:).*?)", "", text, flags=re.DOTALL).strip()


class LMStudioClient:
    def __init__(
        self,
        store: MemoryStore,
        host: str | None = None,
        port: int | None = None,
        user_id: str = "default",
        max_session_turns: int = 5,
    ) -> None:
        self._store = store
        self._host = host or LMS_HOST
        self._port = port or LMS_PORT
        self.user_id = user_id
        self.max_session_turns = max_session_turns
        self._session_history: list[tuple[str, str]] = []
        self._client = Client(f"{self._host}:{self._port}")
        self._negotiate_model()

    def _negotiate_model(self) -> None:
        loaded = list(self._client.llm.list_loaded())
        if not loaded:
            raise RuntimeError("No model loaded in LM Studio. Load a model and retry.")
        if len(loaded) > 1:
            names = [getattr(m, "displayName", None) or m.identifier for m in loaded]
            warnings.warn(f"Multiple models loaded: {names}. Using {names[0]}.")
        model = loaded[0]
        self.model_id = model.identifier
        self.model_name = getattr(model, "displayName", None) or model.identifier
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
            warnings.warn(f"Task expects a {expected} model but {self.model_name!r} is {self.model_size}.")

    def refresh_model(self) -> None:
        self._negotiate_model()

    def _build_chat(self, system: str, user: str) -> Chat:
        chat = Chat(system)
        chat.add_user_message(user)
        return chat

    def _stream(self, chat: Chat) -> Generator[str, None, None]:
        for chunk in self._client.llm.model(self.model_id).respond_stream(chat):
            yield chunk.content

    def ingest(self, text: str) -> Generator[str, None, None]:
        self.warn_if_wrong_size("small")
        chat = self._build_chat(
            "You are a memory assistant. Summarise the following into a concise memory entry.",
            text,
        )
        accumulated: list[str] = []
        for chunk in self._client.llm.model(self.model_id).respond_stream(chat):
            accumulated.append(chunk.content)
        full_response = _strip_think("".join(accumulated))
        yield full_response
        memory_id = self._store.store(full_response, meta={"user_id": self.user_id})
        yield f"\n[memory:{memory_id}]"

    def reason(self, query: str, limit: int = 5) -> Generator[str, None, None]:
        nodes = []

        # Temporal queries → use ALL core memories, sorted by time
        if _is_temporal_query(query):
            try:
                nodes = self._store.get_by_classification("core")
                nodes.sort(key=lambda n: getattr(n, "created_at", 0))
            except Exception:
                pass

        # Non-temporal queries → search with safe cap
        else:
            try:
                nodes = self._store.search(query, limit=limit)
                nodes = nodes[:3] if len(nodes) > 3 else nodes
            except Exception:
                pass

        if not nodes:
            yield "(No relevant memories found.)"
            return

        # Weighted interpolation: blend contexts by relevance (frequency as proxy)
        total_freq = sum(n.frequency for n in nodes) or 1
        weighted_parts = []
        for n in nodes:
            weight = n.frequency / total_freq
            context = n.zoom_levels.get(3, n.content)[:500]  # truncate to avoid overflow
            weighted_parts.append(f"[{weight:.2%}] {context}")
        context_block = "\n".join(weighted_parts)

        chat = self._build_chat(
            "You are a reasoning assistant with access to memory logs. "
            "Answer accurately using the provided context, considering relative weights.",
            f"Memories:\n{context_block}\n\nQuery: {query}",
        )
        yield from self._stream(chat)

    def chat(self, user_message: str, context_limit: int = 5) -> Generator[str, None, None]:
        recent_context = ""
        if self._session_history:
            recent_turns = self._session_history[-self.max_session_turns:]
            recent_context = "\n\nRecent conversation:\n" + "\n".join(
                f"User: {u}\nAssistant: {a}" for u, a in recent_turns
            )

        try:
            nodes = self._store.search(user_message, limit=context_limit)
        except Exception:
            nodes = []

        long_term_context = ""
        if nodes:
            long_term_context = "\n\nMemories from previous sessions:\n" + "\n".join(
                node.zoom_levels.get(3, node.content) for node in nodes
            )

        combined_context = f"{recent_context}{long_term_context}".strip()
        prompt = f"{combined_context}\n\nUser: {user_message}" if combined_context else user_message

        chat = self._build_chat(
            "You are DaVinci. Use recent and long-term memories to maintain continuity. Be concise.",
            prompt,
        )

        accumulated: list[str] = []
        for chunk in self._client.llm.model(self.model_id).respond_stream(chat):
            accumulated.append(chunk.content)

        assistant_response = _strip_think("".join(accumulated))
        yield assistant_response

        # Store main fact (exact user question + answer)
        self._store.store(
            f"User asked: “{user_message}”, Assistant replied: “{assistant_response}”.",
            meta={"user_id": self.user_id}
        )

                # Numeric pattern detection for arithmetic
        match = re.match(r'^(\d+)\s*([×*])\s*(\d+)$', user_message.strip())
        if match:
            a, op, b = match.groups()
            val = eval(f"{a}{op}{b}")
            self._store.store(
                f"Pattern: {a} {op} {b} = {val}",
                meta={"user_id": self.user_id, "type": "pattern"}
            )
            
        # Summary (for concise recall)
        summary_chat = self._build_chat(
            "Summarise the exchange concisely and factually. "
            "DO NOT reference other memories or use labels like 'first', 'last'.",
            f"User asked: “{user_message}”, Assistant replied: “{assistant_response}”."
        )
        
        summary_parts = []
        for chunk in self._client.llm.model(self.model_id).respond_stream(summary_chat):
            summary_parts.append(chunk.content)
        
        full_summary = "".join(summary_parts)
        final_summary = (
            _strip_think(full_summary) 
            if "asked" in full_summary.lower() and "replied" in full_summary.lower()
            else f"User asked: “{user_message}”, Assistant replied: “{assistant_response}”."
        )
        self._store.store(final_summary, meta={"user_id": self.user_id})
        yield "\n[memory saved]"

        self._session_history.append((user_message, assistant_response))

        # Auto-consolidate every 10 turns globally
        if (self._store._get_turn_count() + 1) % 10 == 0:
            count = self._store.consolidate(strategy="frequency")
            yield f"\n[consolidated {count} memories]"
        self._store._inc_turn_count()

    def __enter__(self) -> "LMStudioClient":
        return self

    def __exit__(self, *_: object) -> None:
        pass


def _is_temporal_query(query: str) -> bool:
    q = query.lower()
    keywords = ["first", "earliest", "previous", "before", "initial"]
    return any(kw in q for kw in keywords)
