"""
Tests for davinci.llm.auto_zoom — AutoZoom
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock

from davinci.llm.auto_zoom import AutoZoom
from davinci.llm.backend import LLMBackend


def _make_backend(response: str) -> LLMBackend:
    """Create a mock LLMBackend that returns *response* from chat()."""
    backend = MagicMock(spec=LLMBackend)
    backend.chat.return_value = response
    return backend


class TestAutoZoomFallback(unittest.TestCase):
    """Fallback behaviour when the LLM is unavailable."""

    def test_fallback_returns_dict_with_three_keys(self):
        backend = _make_backend("[LLM unavailable: connection refused]")
        zoom = AutoZoom(backend)
        result = zoom.generate_zoom_levels("Hello world content here.")
        self.assertIn(1, result)
        self.assertIn(2, result)
        self.assertIn(3, result)

    def test_fallback_zoom3_is_full_content(self):
        content = "This is the full content of the memory."
        backend = _make_backend("[LLM unavailable: error]")
        zoom = AutoZoom(backend)
        result = zoom.generate_zoom_levels(content)
        self.assertEqual(result[3], content)

    def test_fallback_zoom1_truncated(self):
        content = "A" * 100
        backend = _make_backend("[LLM unavailable: error]")
        zoom = AutoZoom(backend)
        result = zoom.generate_zoom_levels(content)
        self.assertLessEqual(len(result[1]), 20)

    def test_fallback_zoom2_truncated(self):
        content = "B" * 200
        backend = _make_backend("[LLM unavailable: error]")
        zoom = AutoZoom(backend)
        result = zoom.generate_zoom_levels(content)
        self.assertLessEqual(len(result[2]), 100)

    def test_fallback_on_exception(self):
        backend = MagicMock(spec=LLMBackend)
        backend.chat.side_effect = OSError("network error")
        zoom = AutoZoom(backend)
        result = zoom.generate_zoom_levels("some content")
        self.assertIn(1, result)
        self.assertIn(3, result)


class TestAutoZoomWithMockLLM(unittest.TestCase):
    """Tests with a mock LLM that returns valid JSON."""

    def _make_zoom_response(self, label, summary, full):
        return json.dumps({
            "zoom_level_1": label,
            "zoom_level_2": summary,
            "zoom_level_3": full,
        })

    def test_generates_zoom_levels_from_llm(self):
        content = "Python is a high-level programming language."
        response = self._make_zoom_response(
            "Python programming",
            "Python is a popular high-level language.",
            content,
        )
        backend = _make_backend(response)
        zoom = AutoZoom(backend)
        result = zoom.generate_zoom_levels(content)
        self.assertEqual(result[1], "Python programming")
        self.assertEqual(result[2], "Python is a popular high-level language.")
        self.assertEqual(result[3], content)

    def test_handles_markdown_fenced_json(self):
        content = "Fractal geometry is the study of self-similar patterns."
        inner = json.dumps({
            "zoom_level_1": "Fractal geometry",
            "zoom_level_2": "Study of self-similar patterns.",
            "zoom_level_3": content,
        })
        response = f"```json\n{inner}\n```"
        backend = _make_backend(response)
        zoom = AutoZoom(backend)
        result = zoom.generate_zoom_levels(content)
        self.assertEqual(result[1], "Fractal geometry")

    def test_malformed_json_falls_back(self):
        backend = _make_backend("this is not json")
        zoom = AutoZoom(backend)
        result = zoom.generate_zoom_levels("Some content here.")
        self.assertIn(1, result)
        self.assertIn(3, result)


class TestAutoZoomSummarize(unittest.TestCase):
    def test_summarize_with_mock_llm(self):
        backend = _make_backend("A short summary.")
        zoom = AutoZoom(backend)
        result = zoom.summarize("A very long text that needs summarising.")
        self.assertEqual(result, "A short summary.")

    def test_summarize_fallback_on_unavailable(self):
        backend = _make_backend("[LLM unavailable: error]")
        zoom = AutoZoom(backend)
        text = "word " * 100
        result = zoom.summarize(text, max_words=10)
        words = result.split()
        self.assertLessEqual(len(words), 10)

    def test_summarize_fallback_on_exception(self):
        backend = MagicMock(spec=LLMBackend)
        backend.chat.side_effect = ConnectionError("failed")
        zoom = AutoZoom(backend)
        result = zoom.summarize("Some text here.", max_words=5)
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
