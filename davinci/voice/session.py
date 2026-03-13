"""
DaVinci Voice — Layer 4: Voice Session
=======================================
:class:`VoiceSession` wraps a :class:`~davinci.voice.interface.VoiceInterface`
and adds conversation-history tracking so callers can inspect what was said
and what DaVinci responded.

Usage::

    from davinci.voice.interface import VoiceInterface
    from davinci.voice.session import VoiceSession

    with VoiceInterface(db_path=":memory:") as vi:
        session = VoiceSession(vi)
        session.start()                   # runs conversation_loop
        for entry in session.history():
            print(entry["role"], entry["text"])
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from davinci.voice.interface import VoiceInterface

__all__ = ["VoiceSession"]


class VoiceSession:
    """Manages a voice conversation session with history tracking.

    Wraps a :class:`~davinci.voice.interface.VoiceInterface` and intercepts
    STT/TTS calls to record every exchange in a timestamped history list.

    Parameters
    ----------
    voice_interface: A configured :class:`~davinci.voice.interface.VoiceInterface`
                     instance.

    Examples
    --------
    >>> from davinci.voice.interface import VoiceInterface
    >>> from davinci.voice.session import VoiceSession
    >>> vi = VoiceInterface(db_path=":memory:")
    >>> session = VoiceSession(vi)
    >>> session.history()
    []
    >>> vi.close()
    """

    def __init__(self, voice_interface: "VoiceInterface") -> None:
        self._vi = voice_interface
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Session control
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the conversation loop with history tracking.

        Patches the voice interface's :meth:`~VoiceInterface.listen` and
        :meth:`~VoiceInterface.speak` calls so every utterance and response
        is appended to :attr:`_history` before the conversation loop runs.
        """
        # Wrap listen/speak to record history entries.
        original_listen = self._vi.listen
        original_speak = self._vi.speak

        def tracked_listen() -> str:
            text = original_listen()
            self._history.append({
                "role": "user",
                "text": text,
                "timestamp": time.time(),
            })
            return text

        def tracked_speak(text: str) -> None:
            original_speak(text)
            self._history.append({
                "role": "davinci",
                "text": text,
                "timestamp": time.time(),
            })

        self._vi.listen = tracked_listen  # type: ignore[method-assign]
        self._vi.speak = tracked_speak    # type: ignore[method-assign]

        try:
            self._vi.conversation_loop()
        finally:
            # Restore original methods so the interface can be reused.
            self._vi.listen = original_listen  # type: ignore[method-assign]
            self._vi.speak = original_speak    # type: ignore[method-assign]

    # ------------------------------------------------------------------
    # History access
    # ------------------------------------------------------------------

    def history(self) -> list[dict]:
        """Return the full conversation history.

        Returns
        -------
        list[dict]
            Each entry is ``{"role": "user"|"davinci", "text": str,
            "timestamp": float}``.
        """
        return list(self._history)

    def last_response(self) -> str | None:
        """Return the most recent DaVinci response text, or ``None``.

        Returns
        -------
        str | None
        """
        for entry in reversed(self._history):
            if entry["role"] == "davinci":
                return entry["text"]
        return None

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self._history.clear()
