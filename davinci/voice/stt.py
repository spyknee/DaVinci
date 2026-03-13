"""
DaVinci Voice — Layer 4: Speech-to-Text Backend System
=======================================================
Pluggable STT (Speech-to-Text) backend architecture.

The registry is the extension point.  Adding a real STT provider is as
simple as::

    from davinci.voice.stt import STTRegistry

    class WhisperSTT(STTBackend):
        def listen(self) -> str:
            # call OpenAI Whisper here
            ...
        def is_available(self) -> bool:
            return True
        def name(self) -> str:
            return "whisper"

    STTRegistry.register("whisper", WhisperSTT)
    stt = STTRegistry.get("whisper", api_key="...")

No external dependencies — pure Python + stdlib only.  The :class:`StubSTT`
backend uses pre-canned responses or ``input()`` for keyboard fallback, so
the full voice pipeline can be exercised in tests and CI without any
microphone hardware.
"""

from __future__ import annotations

import abc
import itertools
from typing import Iterator

__all__ = ["STTBackend", "StubSTT", "STTRegistry"]


class STTBackend(abc.ABC):
    """Abstract base class for Speech-to-Text backends.

    Every concrete STT implementation must subclass this and implement the
    three abstract methods.  Constructor keyword arguments are forwarded from
    :meth:`STTRegistry.get` so backends can receive provider-specific config
    (API keys, model names, sample-rates, etc.) without changing the registry
    interface.
    """

    def __init__(self, **config) -> None:  # noqa: ANN003
        self._config = config

    @abc.abstractmethod
    def listen(self) -> str:
        """Capture audio and return the transcribed text.

        Returns
        -------
        str
            The transcribed utterance.
        """

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if this backend is ready to use.

        A backend should return ``False`` if required libraries are missing,
        hardware is unavailable, or credentials are not configured.
        """

    @abc.abstractmethod
    def name(self) -> str:
        """Return a short identifier for this backend (e.g. ``"whisper"``)."""


class StubSTT(STTBackend):
    """Stub STT backend for testing and development.

    Parameters
    ----------
    responses:
        Optional list of pre-canned responses.  :meth:`listen` cycles
        through them in order (and wraps around).  If ``None`` or an empty
        list, :meth:`listen` falls back to ``input()`` so a human can type
        at the keyboard.

    Examples
    --------
    >>> stt = StubSTT(responses=["hello world", "goodbye"])
    >>> stt.listen()
    'hello world'
    >>> stt.listen()
    'goodbye'
    >>> stt.listen()  # cycles back
    'hello world'
    """

    def __init__(self, responses: list[str] | None = None, **config) -> None:
        super().__init__(**config)
        self._responses: list[str] = responses if responses is not None else []
        self._cycle: Iterator[str] | None = (
            itertools.cycle(self._responses) if self._responses else None
        )

    def listen(self) -> str:
        """Return the next pre-canned response, or prompt via ``input()``."""
        if self._cycle is not None:
            return next(self._cycle)
        return input("You: ")

    def is_available(self) -> bool:
        """Always available — no hardware required."""
        return True

    def name(self) -> str:
        """Return backend identifier ``"stub"``."""
        return "stub"


class STTRegistry:
    """Class-level registry of available STT backends.

    Use this as the central extension point to add new STT providers::

        STTRegistry.register("google", GoogleSTT)
        stt = STTRegistry.get("google", credentials_file="creds.json")

    :class:`StubSTT` is pre-registered under the name ``"stub"``.
    """

    _backends: dict[str, type[STTBackend]] = {}

    @classmethod
    def register(cls, name: str, backend_class: type[STTBackend]) -> None:
        """Register a backend class under *name*.

        Parameters
        ----------
        name:          Short identifier (e.g. ``"whisper"``).
        backend_class: Concrete :class:`STTBackend` subclass (not an instance).
        """
        cls._backends[name] = backend_class

    @classmethod
    def get(cls, name: str, **config) -> STTBackend:
        """Instantiate and return a registered backend.

        Parameters
        ----------
        name:   Backend identifier as passed to :meth:`register`.
        config: Forwarded as keyword arguments to the backend constructor.

        Raises
        ------
        KeyError
            If *name* has not been registered.
        """
        if name not in cls._backends:
            raise KeyError(f"STT backend '{name}' is not registered. "
                           f"Available: {cls.available()}")
        return cls._backends[name](**config)

    @classmethod
    def available(cls) -> list[str]:
        """Return a list of all registered backend names."""
        return list(cls._backends.keys())


# Pre-register the stub backend so it is always available out of the box.
STTRegistry.register("stub", StubSTT)
