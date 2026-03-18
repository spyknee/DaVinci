"""
DaVinci Voice — Layer 4: Voice Interface
=========================================
Voice-driven interface to the DaVinci fractal memory system.

:class:`VoiceInterface` implements :class:`~davinci.interface.base.BaseInterface`
and wraps a :class:`~davinci.interface.api.DaVinci` instance together with
pluggable STT and TTS backends so the full memory API is accessible via
spoken commands.

Usage::

    from davinci.voice.interface import VoiceInterface

    with VoiceInterface(db_path=":memory:") as vi:
        vi.speak("Hello!")
        text = vi.listen()        # returns next pre-canned or keyboard input
        vi.conversation_loop()    # interactive conversation

No external dependencies — pure Python + stdlib only.
"""

from __future__ import annotations

from typing import Any

from davinci.core.fractal_engine import MemoryNode
from davinci.interface.api import DaVinci
from davinci.interface.base import BaseInterface
from davinci.voice.stt import STTBackend, STTRegistry
from davinci.voice.tts import TTSBackend, TTSRegistry

__all__ = ["VoiceInterface"]


class VoiceInterface(BaseInterface):
    """Voice-driven interface to DaVinci's fractal memory system.

    Implements :class:`~davinci.interface.base.BaseInterface` so it is a
    drop-in replacement for :class:`~davinci.interface.api.DaVinci` at the
    call-site.  All memory operations are delegated to an internal
    :class:`~davinci.interface.api.DaVinci` instance.

    Parameters
    ----------
    db_path:     Path to the SQLite database file.  Use ``":memory:"`` for
                 ephemeral storage (useful in tests).
    stt_backend: Name of the STT backend to use (default ``"stub"``).
                 Must be registered with :class:`~davinci.voice.stt.STTRegistry`.
    tts_backend: Name of the TTS backend to use (default ``"stub"``).
                 Must be registered with :class:`~davinci.voice.tts.TTSRegistry`.
    max_iter:    Mandelbrot iteration limit passed to the fractal engine.
    **config:    Extra keyword arguments forwarded to both STT and TTS
                 backend constructors.

    Examples
    --------
    >>> from davinci.voice.interface import VoiceInterface
    >>> from davinci.voice.stt import STTRegistry, StubSTT
    >>> STTRegistry.register("stub", StubSTT)
    >>> with VoiceInterface(db_path=":memory:") as vi:
    ...     mid = vi.remember("Fractals are self-similar")
    ...     vi.speak("Stored.")
    Stored.
    """

    def __init__(
        self,
        db_path: str = "davinci_memory.db",
        stt_backend: str = "stub",
        tts_backend: str = "stub",
        max_iter: int = 1000,
        **config,
    ) -> None:
        self._davinci = DaVinci(db_path=db_path, max_iter=max_iter)
        self._stt: STTBackend = STTRegistry.get(stt_backend, **config)
        self._tts: TTSBackend = TTSRegistry.get(tts_backend, **config)

    # ------------------------------------------------------------------
    # Voice-specific methods
    # ------------------------------------------------------------------

    def listen(self) -> str:
        """Capture and return the next spoken utterance via the STT backend."""
        return self._stt.listen()

    def speak(self, text: str) -> None:
        """Output *text* via the TTS backend."""
        self._tts.speak(text)

    def parse_intent(self, text: str) -> tuple[str, str]:
        """Parse a natural-language utterance into a (command, argument) pair.

        Simple keyword-prefix matching is used.  The mapping is:

        ============================================ =====================
        Utterance pattern                            Result
        ============================================ =====================
        ``"remember <content>"``                     ``("remember", content)``
        ``"recall <id>"``                            ``("search", id)``
        ``"search [for] <query>"``                   ``("search", query)``
        ``"forget"``                                 ``("forget", "")``
        ``"stats"``                                  ``("stats", "")``
        ``"decay"``                                  ``("decay", "")``
        ``"quit"`` / ``"exit"`` / ``"stop"``         ``("quit", "")``
        anything else                                ``("remember", text)``
        ============================================ =====================

        Parameters
        ----------
        text: Raw text as returned by :meth:`listen`.

        Returns
        -------
        tuple[str, str]
            ``(command, argument)`` where *argument* is ``""`` if unused.
        """
        lowered = text.strip().lower()

        if lowered.startswith("remember "):
            return ("remember", text.strip()[len("remember "):])

        if lowered.startswith("recall "):
            return ("search", text.strip()[len("recall "):])

        if lowered.startswith("search for "):
            return ("search", text.strip()[len("search for "):])

        if lowered.startswith("search "):
            return ("search", text.strip()[len("search "):])

        if lowered in ("forget", "forget all"):
            return ("forget", "")

        if lowered == "stats":
            return ("stats", "")

        if lowered == "decay":
            return ("decay", "")

        if lowered in ("quit", "exit", "stop"):
            return ("quit", "")

        # Default: store whatever was said as a new memory.
        return ("remember", text)

    def conversation_loop(self) -> None:
        """Run an interactive voice conversation loop.

        Each iteration:

        1. Speaks a prompt asking what to remember or recall.
        2. Listens for a spoken utterance.
        3. Parses the intent and dispatches to the appropriate memory
           operation.
        4. Speaks a confirmation or result.
        5. Repeats until the user says "quit", "exit", or "stop".
        """
        while True:
            self.speak("What would you like to remember or recall?")
            text = self.listen()
            command, argument = self.parse_intent(text)

            if command == "quit":
                self.speak("Goodbye!")
                break

            elif command == "remember":
                mid = self.remember(argument)
                self.speak(f"Remembered with ID {mid}.")

            elif command == "search":
                results = self.search(argument)
                if results:
                    summary = "; ".join(n.content for n in results[:3])
                    self.speak(f"Found {len(results)} result(s): {summary}")
                else:
                    self.speak("No memories found.")

            elif command == "forget":
                count = self.forget()
                self.speak(f"Forgot {count} memor{'y' if count == 1 else 'ies'}.")

            elif command == "stats":
                s = self.stats()
                self.speak(
                    f"You have {s.get('total', 0)} memor"
                    f"{'y' if s.get('total', 0) == 1 else 'ies'}."
                )

            elif command == "decay":
                result = self.decay()
                total_moved = sum(result.values())
                self.speak(f"Decay complete. {total_moved} memor"
                           f"{'y' if total_moved == 1 else 'ies'} reclassified.")

    # ------------------------------------------------------------------
    # BaseInterface delegation — memory operations
    # ------------------------------------------------------------------

    def remember(
        self,
        content: str,
        zoom_levels: dict | None = None,
        meta: dict | None = None,
    ) -> str:
        """Store a new memory and return its UUID."""
        return self._davinci.remember(content, zoom_levels=zoom_levels, meta=meta)

    def recall(self, memory_id: str) -> MemoryNode | None:
        """Retrieve a memory by UUID."""
        return self._davinci.recall(memory_id)

    def search(self, query: str, limit: int = 10) -> list[MemoryNode]:
        """Search memories by content substring."""
        return self._davinci.search(query, limit=limit)

    def forget(self, classification: str = "forget") -> int:
        """Prune memories by classification."""
        return self._davinci.forget(classification)

    def decay(self) -> dict[str, int]:
        """Run a decay cycle."""
        return self._davinci.decay()

    def consolidate(self, strategy: str = "frequency") -> int:
        """Run the consolidation engine."""
        return self._davinci.consolidate(strategy)

    def merge_similar(self, threshold: float = 0.8) -> int:
        """Merge highly similar memories."""
        return self._davinci.merge_similar(threshold)

    def stats(self) -> dict[str, Any]:
        """Return aggregate statistics about the memory store."""
        return self._davinci.stats()

    def memories(self, classification: str | None = None) -> list[MemoryNode]:
        """Return all memories, optionally filtered by classification."""
        return self._davinci.memories(classification)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database connection."""
        self._davinci.close()

    def __enter__(self) -> "VoiceInterface":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
