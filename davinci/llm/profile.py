"""
DaVinci LLM — Profile System
==============================
Simple JSON-backed profile management.

Inspired by Gilligan's ``profile.json`` design but simplified for DaVinci's
clean architecture.  Supports dot-path access and persistent save/load.

No external dependencies — pure Python + stdlib only.
"""

from __future__ import annotations

import json
import os
from typing import Any

__all__ = ["Profile"]

# Default profile structure — mirrors Gilligan's profile defaults exactly
_DEFAULTS: dict = {
    "llm": {
        "active_model": "qwen",
        "models": {
            "qwen": {
                "base_url": "http://127.0.0.1:1234",
                "model": "qwen/qwen3-coder-next",
                "api_key": "lm-studio",
                "provider": "lmstudio",
            },
            "qwen35": {
                "base_url": "http://127.0.0.1:1234",
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
    },
    "memory": {
        "decay_rate_per_day": 0.05,
        "prune_threshold": 0.2,
        "episodic_top_k": 2,
        "episodic_importance_boost": 0.1,
        "episodic_default_importance": 0.5,
        "auto_learn_enabled": True,
        "auto_learn_require_user_approval": True,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict.

    Always creates new dicts — never shares references with *base* or *override*.
    """
    # Start by deep-copying base
    result: dict = {}
    for k, v in base.items():
        if isinstance(v, dict):
            result[k] = _deep_merge(v, {})
        else:
            result[k] = v
    # Apply override on top
    for k, v in override.items():
        if isinstance(v, dict):
            if isinstance(result.get(k), dict):
                result[k] = _deep_merge(result[k], v)
            else:
                # Deep-copy the override dict rather than sharing the reference
                result[k] = _deep_merge(v, {})
        else:
            result[k] = v
    return result


class Profile:
    """JSON-backed profile with dot-path access.

    Parameters
    ----------
    path: Path to the JSON profile file.  If ``None`` (or the file does not
          exist), the built-in defaults are used.

    Examples
    --------
    >>> p = Profile()
    >>> p.get("llm.active_model")
    'qwen'
    >>> p.set("llm.active_model", "qwen35")
    >>> p.get("llm.active_model")
    'qwen35'
    """

    def __init__(self, path: str | None = None) -> None:
        self._path = path
        self._data: dict = _deep_merge({}, _DEFAULTS)
        if path is not None:
            try:
                loaded = self.load()
                self._data = _deep_merge(_DEFAULTS, loaded)
            except (FileNotFoundError, ValueError, json.JSONDecodeError):
                pass  # Use defaults

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

    def load(self) -> dict:
        """Read and return the profile dict from disk.

        Raises
        ------
        FileNotFoundError
            If the profile file does not exist.
        ValueError
            If the file exists but contains invalid JSON.
        """
        if self._path is None:
            return {}
        with open(self._path, encoding="utf-8") as fh:
            content = fh.read().strip()
        if not content:
            return {}
        return json.loads(content)

    def save(self) -> None:
        """Write the current profile to disk.

        Does nothing if no *path* was provided at construction time.
        """
        if self._path is None:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2)

    # ------------------------------------------------------------------
    # Dot-path access
    # ------------------------------------------------------------------

    def get(self, key_path: str, default: Any = None) -> Any:
        """Read a nested value using dot-path notation.

        Parameters
        ----------
        key_path: Dot-separated key path, e.g. ``"llm.active_model"``.
        default:  Value returned if the path does not exist.

        Returns
        -------
        Any
        """
        parts = key_path.split(".")
        node: Any = self._data
        for part in parts:
            if not isinstance(node, dict):
                return default
            node = node.get(part)
            if node is None:
                return default
        return node

    def set(self, key_path: str, value: Any) -> None:
        """Set a nested value using dot-path notation.

        Intermediate dicts are created automatically.

        Parameters
        ----------
        key_path: Dot-separated key path, e.g. ``"llm.active_model"``.
        value:    The value to store.
        """
        parts = key_path.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    # ------------------------------------------------------------------
    # Shortcuts
    # ------------------------------------------------------------------

    def llm_config(self) -> dict:
        """Return the entire ``llm`` section of the profile."""
        return self._data.get("llm", {})

    # ------------------------------------------------------------------
    # Dict-like access
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def as_dict(self) -> dict:
        """Return a shallow copy of the underlying data dict."""
        return dict(self._data)
