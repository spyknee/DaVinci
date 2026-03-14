"""
DaVinci LLM — Backend System
=============================
Abstract base class and concrete implementations for LLM backends.

Connects to LM Studio (or other OpenAI-compatible endpoints) using
stdlib ``http.client`` and ``json`` only — **no openai package, no requests**.

Backends
--------
- :class:`LMStudioBackend` — connects to a local LM Studio instance
- :class:`GitHubModelsBackend` — placeholder for GitHub Models inference

Registry
--------
:class:`LLMRegistry` — same registry pattern as the STT/TTS registries.
"""

from __future__ import annotations

import http.client
import json
import os
import urllib.parse
from abc import ABC, abstractmethod

__all__ = [
    "LLMBackend",
    "LMStudioBackend",
    "GitHubModelsBackend",
    "LLMRegistry",
]


class LLMBackend(ABC):
    """Abstract base class for all LLM backends.

    Parameters
    ----------
    **config: Arbitrary keyword arguments forwarded by subclasses.
    """

    def __init__(self, **config) -> None:  # noqa: ANN003
        self._config = config

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 900,
        temperature: float = 0.7,
    ) -> str:
        """Send *messages* to the LLM and return the response text.

        Parameters
        ----------
        messages:    OpenAI-style message list (``[{"role": ..., "content": ...}, ...]``).
        max_tokens:  Maximum tokens for the completion.
        temperature: Sampling temperature.

        Returns
        -------
        str
            The assistant's reply, or an error message prefixed with
            ``"[LLM unavailable: ..."`` if the backend cannot be reached.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if this backend is currently reachable."""

    @abstractmethod
    def name(self) -> str:
        """Return the short identifier for this backend (e.g. ``"lmstudio"``)."""

    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string (e.g. ``"qwen/qwen3-coder-next"``)."""


# ---------------------------------------------------------------------------
# LM Studio backend
# ---------------------------------------------------------------------------


class LMStudioBackend(LLMBackend):
    """LLM backend that talks to a local LM Studio instance.

    Uses stdlib ``http.client`` — **no openai package, no requests**.

    Parameters
    ----------
    base_url:  Base URL of the LM Studio server (default ``"http://192.168.0.176:1234"``).
    model:     Model identifier to request (default ``"qwen/qwen3-coder-next"``).
    api_key:   API key sent as Bearer token (default ``"lm-studio"``).
    """

    def __init__(
        self,
        base_url: str = "http://192.168.0.176:1234",
        model: str = "qwen/qwen3-coder-next",
        api_key: str = "lm-studio",
        **config,
    ) -> None:
        super().__init__(**config)
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parsed_url(self) -> tuple[str, str, str]:
        """Return ``(scheme, host_with_port, path_prefix)`` parsed from base_url."""
        parsed = urllib.parse.urlparse(self._base_url)
        scheme = parsed.scheme or "http"
        netloc = parsed.netloc or parsed.path  # handle bare "host:port"
        return scheme, netloc, parsed.path.rstrip("/")

    def _make_connection(self) -> http.client.HTTPConnection | http.client.HTTPSConnection:
        scheme, netloc, _ = self._parsed_url()
        if scheme == "https":
            return http.client.HTTPSConnection(netloc, timeout=120)
        return http.client.HTTPConnection(netloc, timeout=120)

    # ------------------------------------------------------------------
    # LLMBackend interface
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 900,
        temperature: float = 0.7,
    ) -> str:
        """POST to ``/v1/chat/completions`` and return the assistant reply."""
        payload = json.dumps(
            {
                "model": self._model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            }
        ).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        _, _, path_prefix = self._parsed_url()
        endpoint = f"{path_prefix}/v1/chat/completions"

        try:
            conn = self._make_connection()
            conn.request("POST", endpoint, body=payload, headers=headers)
            response = conn.getresponse()
            raw = response.read().decode("utf-8")
            conn.close()
            data = json.loads(raw)
            return data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            return f"[LLM unavailable: {exc}]"

    def is_available(self) -> bool:
        """Return ``True`` if the ``/v1/models`` endpoint responds with HTTP 200."""
        _, _, path_prefix = self._parsed_url()
        endpoint = f"{path_prefix}/v1/models"
        try:
            conn = self._make_connection()
            conn.request("GET", endpoint, headers={"Authorization": f"Bearer {self._api_key}"})
            response = conn.getresponse()
            conn.close()
            return response.status == 200
        except Exception:  # noqa: BLE001
            return False

    def name(self) -> str:
        return "lmstudio"

    def model_name(self) -> str:
        return self._model


# ---------------------------------------------------------------------------
# GitHub Models backend (placeholder)
# ---------------------------------------------------------------------------


class GitHubModelsBackend(LLMBackend):
    """Placeholder LLM backend for GitHub Models inference.

    Reads the API key from the environment variable specified by *api_key_env*.
    Uses stdlib ``http.client`` — **no openai package, no requests**.

    Parameters
    ----------
    base_url:    Inference endpoint base URL.
    model:       Model identifier (``"PLACEHOLDER"`` until a real model is chosen).
    api_key_env: Name of the environment variable that holds the GitHub token.
    """

    def __init__(
        self,
        base_url: str = "https://models.inference.ai.azure.com",
        model: str = "PLACEHOLDER",
        api_key_env: str = "GITHUB_TOKEN",
        **config,
    ) -> None:
        super().__init__(**config)
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key_env = api_key_env

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _api_key(self) -> str:
        return os.environ.get(self._api_key_env, "")

    def _parsed_url(self) -> tuple[str, str, str]:
        parsed = urllib.parse.urlparse(self._base_url)
        scheme = parsed.scheme or "https"
        netloc = parsed.netloc or parsed.path
        return scheme, netloc, parsed.path.rstrip("/")

    def _make_connection(self) -> http.client.HTTPConnection | http.client.HTTPSConnection:
        scheme, netloc, _ = self._parsed_url()
        if scheme == "https":
            return http.client.HTTPSConnection(netloc, timeout=30)
        return http.client.HTTPConnection(netloc, timeout=30)

    # ------------------------------------------------------------------
    # LLMBackend interface
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 900,
        temperature: float = 0.7,
    ) -> str:
        """POST to ``/chat/completions`` on the GitHub Models endpoint."""
        api_key = self._api_key()
        if not api_key:
            return f"[LLM unavailable: {self._api_key_env} environment variable not set]"

        payload = json.dumps(
            {
                "model": self._model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            }
        ).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        _, _, path_prefix = self._parsed_url()
        endpoint = f"{path_prefix}/chat/completions"

        try:
            conn = self._make_connection()
            conn.request("POST", endpoint, body=payload, headers=headers)
            response = conn.getresponse()
            raw = response.read().decode("utf-8")
            conn.close()
            data = json.loads(raw)
            return data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            return f"[LLM unavailable: {exc}]"

    def is_available(self) -> bool:
        """Return ``True`` if the ``GITHUB_TOKEN`` environment variable is set."""
        return bool(self._api_key())

    def name(self) -> str:
        return "github"

    def model_name(self) -> str:
        return self._model


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class LLMRegistry:
    """Registry of LLM backend classes.

    Follows the same pattern as the STT/TTS registries in Layer 4.

    Pre-registered backends
    -----------------------
    ``"lmstudio"`` → :class:`LMStudioBackend`
    ``"github"``   → :class:`GitHubModelsBackend`
    """

    def __init__(self) -> None:
        self._registry: dict[str, type[LLMBackend]] = {}
        # Pre-register built-in backends
        self.register("lmstudio", LMStudioBackend)
        self.register("github", GitHubModelsBackend)

    def register(self, name: str, backend_class: type[LLMBackend]) -> None:
        """Register a backend class under *name*.

        Parameters
        ----------
        name:          Short string key (e.g. ``"lmstudio"``).
        backend_class: A subclass of :class:`LLMBackend`.
        """
        self._registry[name] = backend_class

    def get(self, name: str, **config) -> LLMBackend:
        """Instantiate and return the backend registered under *name*.

        Parameters
        ----------
        name:    Backend key (must have been registered first).
        **config: Passed directly to the backend constructor.

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        cls = self._registry[name]
        return cls(**config)

    def available(self) -> list[str]:
        """Return the list of registered backend names."""
        return list(self._registry.keys())
