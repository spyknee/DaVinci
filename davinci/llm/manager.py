"""
DaVinci LLM — Model Manager
DaVinci does NOT select models — it reports whatever LM Studio has active.
"""

from __future__ import annotations

from davinci.llm.backend import LMStudioBackend
from davinci.llm.config import LMS_HOST, LMS_PORT

__all__ = ["ModelManager"]


class ModelManager:
    """Reports the live active model from LM Studio. No switching."""

    def __init__(self) -> None:
        self._backend = LMStudioBackend(
            base_url=f"http://{LMS_HOST}:{LMS_PORT}",
        )

    def active(self) -> LMStudioBackend:
        return self._backend

    def active_model_name(self) -> str:
        return self._backend.model_name()

    def is_available(self) -> bool:
        return self._backend.is_available()

    def base_url(self) -> str:
        return self._backend._base_url

    def status(self) -> dict:
        return {
            "model": self.active_model_name(),
            "base_url": self.base_url(),
            "available": self.is_available(),
        }