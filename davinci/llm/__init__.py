"""
DaVinci LLM — Layer 5A: LLM Integration
========================================
Re-exports for the ``davinci.llm`` package.

No external dependencies — pure Python + stdlib only.
"""

from davinci.llm.backend import (
    GitHubModelsBackend,
    LLMBackend,
    LLMRegistry,
    LMStudioBackend,
)
from davinci.llm.manager import ModelManager
from davinci.llm.profile import Profile
from davinci.llm.auto_zoom import AutoZoom
from davinci.llm.auto_learn import AutoLearn

__all__ = [
    "LLMBackend",
    "LMStudioBackend",
    "GitHubModelsBackend",
    "LLMRegistry",
    "ModelManager",
    "Profile",
    "AutoZoom",
    "AutoLearn",
]
