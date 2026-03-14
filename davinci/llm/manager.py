"""
DaVinci LLM — Model Manager
=============================
Handles model switching, toggling, and profile-based auto-selection.

Inspired by Gilligan's ``/model``, ``/model_use``, and ``/model_toggle``
commands but implemented with DaVinci's clean architecture.

No external dependencies — pure Python + stdlib only.
"""

from __future__ import annotations

from typing import Any

from davinci.llm.backend import LLMBackend, LLMRegistry

__all__ = ["ModelManager"]

_REGISTRY = LLMRegistry()

# Default profile mirrors Gilligan's profile.json exactly
_DEFAULT_PROFILE: dict = {
    "active_model": "qwen",
    "models": {
        "qwen": {
            "base_url": "http://192.168.0.176:1234",
            "model": "qwen/qwen3-coder-next",
            "api_key": "lm-studio",
            "provider": "lmstudio",
        },
        "qwen35": {
            "base_url": "http://192.168.0.176:1234",
            "model": "qwen/qwen3.5-9b",
            "api_key": "lm-studio",
            "provider": "lmstudio",
        },
        "model3": {
            "base_url": "https://models.inference.ai.azure.com",
            "model": "PLACEHOLDER",
            "api_key_env": "GITHUB_TOKEN",
            "provider": "github",
        },
    },
}


class ModelManager:
    """Manages multiple LLM backend configurations with runtime switching.

    Parameters
    ----------
    profile: Optional dict containing ``active_model`` and ``models`` keys
             (same shape as Gilligan's ``llm`` profile section).  When
             ``None`` the built-in defaults are used.

    Examples
    --------
    >>> mm = ModelManager()
    >>> mm.active_name()
    'qwen'
    >>> mm.toggle() in mm.available_models()
    True
    """

    def __init__(self, profile: dict | None = None) -> None:
        cfg = profile if profile is not None else _DEFAULT_PROFILE
        self._active_name: str = cfg.get("active_model", "qwen")
        self._models: dict[str, dict] = dict(cfg.get("models", {}))
        # Lazy backend cache: name → LLMBackend instance
        self._backends: dict[str, LLMBackend] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_backend(self, name: str) -> LLMBackend:
        """Instantiate a backend for the model config registered under *name*."""
        cfg = self._models[name]
        provider = cfg.get("provider", "lmstudio")
        # Build kwargs without the "provider" key
        kwargs: dict[str, Any] = {k: v for k, v in cfg.items() if k != "provider"}
        return _REGISTRY.get(provider, **kwargs)

    def _backend_for(self, name: str) -> LLMBackend:
        """Return (and lazily create) the backend instance for *name*."""
        if name not in self._backends:
            self._backends[name] = self._build_backend(name)
        return self._backends[name]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def active(self) -> LLMBackend:
        """Return the currently active :class:`~davinci.llm.backend.LLMBackend` instance."""
        return self._backend_for(self._active_name)

    def active_name(self) -> str:
        """Return the key name of the active model (e.g. ``"qwen"``)."""
        return self._active_name

    def active_model_name(self) -> str:
        """Return the model identifier of the active model (e.g. ``"qwen/qwen3-coder-next"``)."""
        return self.active().model_name()

    def switch(self, name: str) -> bool:
        """Switch the active model to *name*.

        Parameters
        ----------
        name: Key name of the target model (must exist in the configured models).

        Returns
        -------
        bool
            ``True`` on success, ``False`` if *name* is not a known model.
        """
        if name not in self._models:
            return False
        self._active_name = name
        return True

    def toggle(self) -> str:
        """Cycle to the next model in the configured order.

        Returns
        -------
        str
            The key name of the newly active model.
        """
        keys = list(self._models.keys())
        if not keys:
            return self._active_name
        try:
            idx = keys.index(self._active_name)
        except ValueError:
            idx = -1
        next_idx = (idx + 1) % len(keys)
        self._active_name = keys[next_idx]
        return self._active_name

    def available_models(self) -> list[str]:
        """Return the list of configured model key names."""
        return list(self._models.keys())

    def status(self) -> dict:
        """Return a status dict for the currently active model.

        Returns
        -------
        dict
            Keys: ``active``, ``model``, ``base_url``, ``available``.
        """
        cfg = self._models.get(self._active_name, {})
        backend = self._backend_for(self._active_name)
        return {
            "active": self._active_name,
            "model": backend.model_name(),
            "base_url": cfg.get("base_url", ""),
            "available": backend.is_available(),
        }
