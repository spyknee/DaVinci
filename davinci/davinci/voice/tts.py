"""
DaVinci Voice — Layer 4: Text-to-Speech Backend System
=======================================================
Pluggable TTS (Text-to-Speech) backend architecture.

The registry is the extension point.  Adding a real TTS provider is as
simple as::

    from davinci.voice.tts import TTSRegistry

    class Pyttsx3TTS(TTSBackend):
        def speak(self, text: str) -> None:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        def is_available(self) -> bool:
            try:
                import pyttsx3  # noqa: F401
                return True
            except ImportError:
                return False
        def name(self) -> str:
            return "pyttsx3"

    TTSRegistry.register("pyttsx3", Pyttsx3TTS)
    tts = TTSRegistry.get("pyttsx3")

No external dependencies — pure Python + stdlib only.  The :class:`StubTTS`
backend writes to stdout (and optionally to an in-memory list) so the full
voice pipeline can be exercised in tests and CI without speakers.
"""

from __future__ import annotations

import abc

__all__ = ["TTSBackend", "StubTTS", "TTSRegistry"]


class TTSBackend(abc.ABC):
    """Abstract base class for Text-to-Speech backends.

    Every concrete TTS implementation must subclass this and implement the
    three abstract methods.  Constructor keyword arguments are forwarded from
    :meth:`TTSRegistry.get` so backends can receive provider-specific config
    (voice IDs, speech rates, API keys, etc.) without changing the registry
    interface.
    """

    def __init__(self, **config) -> None:  # noqa: ANN003
        self._config = config

    @abc.abstractmethod
    def speak(self, text: str) -> None:
        """Output speech from *text*.

        Parameters
        ----------
        text: The text to synthesise and speak aloud.
        """

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if this backend is ready to use.

        A backend should return ``False`` if required libraries are missing,
        hardware is unavailable, or credentials are not configured.
        """

    @abc.abstractmethod
    def name(self) -> str:
        """Return a short identifier for this backend (e.g. ``"pyttsx3"``)."""


class StubTTS(TTSBackend):
    """Stub TTS backend for testing and development.

    Parameters
    ----------
    output:
        Optional list to collect spoken text for assertions in tests.
        Each call to :meth:`speak` appends the text to this list.

    Examples
    --------
    >>> spoken = []
    >>> tts = StubTTS(output=spoken)
    >>> tts.speak("Hello, world!")
    Hello, world!
    >>> spoken
    ['Hello, world!']
    """

    def __init__(self, output: list | None = None, **config) -> None:
        super().__init__(**config)
        self._output: list | None = output

    def speak(self, text: str) -> None:
        """Print *text* to stdout and optionally append it to the output list."""
        print(text)
        if self._output is not None:
            self._output.append(text)

    def is_available(self) -> bool:
        """Always available — no hardware required."""
        return True

    def name(self) -> str:
        """Return backend identifier ``"stub"``."""
        return "stub"


class TTSRegistry:
    """Class-level registry of available TTS backends.

    Use this as the central extension point to add new TTS providers::

        TTSRegistry.register("pyttsx3", Pyttsx3TTS)
        tts = TTSRegistry.get("pyttsx3")

    :class:`StubTTS` is pre-registered under the name ``"stub"``.
    """

    _backends: dict[str, type[TTSBackend]] = {}

    @classmethod
    def register(cls, name: str, backend_class: type[TTSBackend]) -> None:
        """Register a backend class under *name*.

        Parameters
        ----------
        name:          Short identifier (e.g. ``"pyttsx3"``).
        backend_class: Concrete :class:`TTSBackend` subclass (not an instance).
        """
        cls._backends[name] = backend_class

    @classmethod
    def get(cls, name: str, **config) -> TTSBackend:
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
            raise KeyError(f"TTS backend '{name}' is not registered. "
                           f"Available: {cls.available()}")
        return cls._backends[name](**config)

    @classmethod
    def available(cls) -> list[str]:
        """Return a list of all registered backend names."""
        return list(cls._backends.keys())


# Pre-register the stub backend so it is always available out of the box.
TTSRegistry.register("stub", StubTTS)
