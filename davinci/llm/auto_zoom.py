"""
DaVinci LLM — Auto-Zoom
========================
Generates zoom levels automatically when storing a memory, using the LLM.

Inspired by Gilligan's L0/L1/L2/L3 zoom concept but implemented with
DaVinci's clean architecture and no external dependencies.

No external dependencies — pure Python + stdlib only.
"""

from __future__ import annotations

import json

from davinci.llm.backend import LLMBackend

__all__ = ["AutoZoom"]

_ZOOM_PROMPT = """\
Given the following text, produce a JSON object with exactly these keys:
- "zoom_level_1": a 1-3 word label or tag summarising the topic
- "zoom_level_2": a single sentence summary (max 25 words)
- "zoom_level_3": the full text, verbatim

Return ONLY valid JSON — no markdown fences, no extra commentary.

Text:
{content}"""


class AutoZoom:
    """Generates zoom levels for memories using an LLM backend.

    Parameters
    ----------
    llm_backend: An :class:`~davinci.llm.backend.LLMBackend` instance.

    Examples
    --------
    >>> zoom = AutoZoom(my_backend)
    >>> levels = zoom.generate_zoom_levels("The Mandelbrot set is a fractal.")
    >>> levels[1]  # 1-3 word label
    'Mandelbrot fractal'
    """

    def __init__(self, llm_backend: LLMBackend) -> None:
        self._llm = llm_backend

    def generate_zoom_levels(self, content: str) -> dict:
        """Ask the LLM to generate zoom levels for *content*.

        Falls back gracefully when the LLM is unavailable.

        Parameters
        ----------
        content: The text to summarise.

        Returns
        -------
        dict
            ``{1: label, 2: sentence_summary, 3: full_content}``
        """
        fallback = {
            1: content[:20].strip(),
            2: content[:100].strip(),
            3: content,
        }

        prompt = _ZOOM_PROMPT.format(content=content)
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = self._llm.chat(messages, max_tokens=300, temperature=0.3)
        except Exception:  # noqa: BLE001
            return fallback

        if raw.startswith("[LLM unavailable"):
            return fallback

        # Strip potential markdown code fences
        clean = raw.strip()
        if clean.startswith("```"):
            lines = clean.splitlines()
            # Remove first and last fence lines
            clean = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()

        try:
            data = json.loads(clean)
            return {
                1: str(data.get("zoom_level_1", fallback[1])),
                2: str(data.get("zoom_level_2", fallback[2])),
                3: str(data.get("zoom_level_3", content)),
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            return fallback

    def summarize(self, text: str, max_words: int = 50) -> str:
        """Return a concise summary of *text* using the LLM.

        Falls back to the first ``max_words`` words if the LLM is unavailable.

        Parameters
        ----------
        text:      The text to summarise.
        max_words: Approximate word limit for the summary.

        Returns
        -------
        str
        """
        words = text.split()
        fallback = " ".join(words[:max_words])

        messages = [
            {
                "role": "user",
                "content": (
                    f"Summarise the following text in at most {max_words} words. "
                    "Return only the summary, nothing else.\n\n" + text
                ),
            }
        ]

        try:
            result = self._llm.chat(messages, max_tokens=200, temperature=0.3)
        except Exception:  # noqa: BLE001
            return fallback

        if result.startswith("[LLM unavailable"):
            return fallback

        return result.strip()
