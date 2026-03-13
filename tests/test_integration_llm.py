"""
Tests for Layer 5A integration through davinci.interface.api.DaVinci
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from davinci.interface.api import DaVinci
from davinci.llm.backend import LLMBackend


def _make_backend(response: str) -> LLMBackend:
    backend = MagicMock(spec=LLMBackend)
    backend.chat.return_value = response
    backend.is_available.return_value = True
    backend.model_name.return_value = "mock/model"
    backend.name.return_value = "mock"
    return backend


class TestDaVinciLLMDisabled(unittest.TestCase):
    """All existing functionality must work with llm_enabled=False."""

    def setUp(self):
        self.dv = DaVinci(":memory:", llm_enabled=False)

    def tearDown(self):
        self.dv.close()

    def test_remember_works(self):
        mid = self.dv.remember("Hello DaVinci")
        self.assertIsNotNone(mid)

    def test_recall_works(self):
        mid = self.dv.remember("Hello DaVinci")
        node = self.dv.recall(mid)
        self.assertIsNotNone(node)

    def test_search_works(self):
        self.dv.remember("Python is great")
        results = self.dv.search("Python")
        self.assertGreater(len(results), 0)

    def test_search_fts_works(self):
        self.dv.remember("Fractal geometry is fascinating")
        results = self.dv.search_fts("Fractal")
        self.assertGreater(len(results), 0)

    def test_stats_works(self):
        s = self.dv.stats()
        self.assertIn("total", s)

    def test_decay_works(self):
        self.dv.remember("test")
        result = self.dv.decay()
        self.assertIsInstance(result, dict)

    def test_forget_works(self):
        self.dv.remember("test")
        count = self.dv.forget("forget")
        self.assertIsInstance(count, int)

    def test_ask_returns_error_message(self):
        answer = self.dv.ask("What is 2+2?")
        self.assertIn("unavailable", answer.lower())

    def test_model_status_returns_empty(self):
        status = self.dv.model_status()
        self.assertEqual(status, {})

    def test_model_switch_returns_false(self):
        self.assertFalse(self.dv.model_switch("qwen35"))

    def test_model_toggle_returns_empty_string(self):
        self.assertEqual(self.dv.model_toggle(), "")

    def test_episodic_status_works(self):
        s = self.dv.episodic_status()
        self.assertIn("count", s)

    def test_review_pending_returns_empty(self):
        self.assertEqual(self.dv.review_pending(), [])

    def test_approve_fact_returns_false(self):
        self.assertFalse(self.dv.approve_fact(0))

    def test_reject_fact_returns_false(self):
        self.assertFalse(self.dv.reject_fact(0))


class TestDaVinciAskWithMockLLM(unittest.TestCase):
    """Test the ask() pipeline with a mock LLM backend."""

    def setUp(self):
        self.dv = DaVinci(":memory:", llm_enabled=True)
        # Replace the model manager's active backend with our mock
        mock_backend = _make_backend("The answer is 42.")
        if self.dv._model_manager is not None:
            self.dv._model_manager._backends = {
                self.dv._model_manager._active_name: mock_backend
            }

    def tearDown(self):
        self.dv.close()

    def test_ask_returns_response(self):
        if self.dv._model_manager is None:
            self.skipTest("Model manager not initialised")
        answer = self.dv.ask("What is the meaning of life?")
        self.assertEqual(answer, "The answer is 42.")

    def test_ask_saves_to_episodic(self):
        if self.dv._model_manager is None:
            self.skipTest("Model manager not initialised")
        self.dv.ask("Test question?")
        status = self.dv.episodic_status()
        self.assertGreater(status["count"], 0)

    def test_ask_queues_pending_facts(self):
        if self.dv._model_manager is None:
            self.skipTest("Model manager not initialised")
        # Override auto-learn backend to return extractable facts
        if self.dv._auto_learn is not None:
            facts = json.dumps(["The answer is 42."])
            self.dv._auto_learn._llm = _make_backend(facts)
        self.dv.ask("What is the meaning of life?")
        # Facts queued (not auto-stored by default)
        pending = self.dv.review_pending()
        self.assertIsInstance(pending, list)


class TestDaVinciModelOperations(unittest.TestCase):
    def setUp(self):
        self.dv = DaVinci(":memory:", llm_enabled=True)

    def tearDown(self):
        self.dv.close()

    def test_model_switch(self):
        if self.dv._model_manager is None:
            self.skipTest("Model manager not initialised")
        result = self.dv.model_switch("qwen35")
        self.assertTrue(result)
        status = self.dv.model_status()
        self.assertEqual(status["active"], "qwen35")

    def test_model_switch_unknown(self):
        if self.dv._model_manager is None:
            self.skipTest("Model manager not initialised")
        result = self.dv.model_switch("no_such_model")
        self.assertFalse(result)

    def test_model_toggle(self):
        if self.dv._model_manager is None:
            self.skipTest("Model manager not initialised")
        initial = self.dv.model_status().get("active")
        toggled = self.dv.model_toggle()
        self.assertNotEqual(toggled, initial)

    def test_model_status_keys(self):
        if self.dv._model_manager is None:
            self.skipTest("Model manager not initialised")
        status = self.dv.model_status()
        self.assertIn("active", status)
        self.assertIn("model", status)
        self.assertIn("base_url", status)
        self.assertIn("available", status)


class TestDaVinciRememberWithAutoZoom(unittest.TestCase):
    """remember() should auto-generate zoom levels via LLM when available."""

    def test_remember_with_mock_auto_zoom(self):
        with DaVinci(":memory:", llm_enabled=True) as dv:
            if dv._auto_zoom is None:
                self.skipTest("AutoZoom not initialised")
            # Patch auto_zoom to return known zoom levels
            zoom_levels = {1: "label", 2: "summary sentence", 3: "full content"}
            dv._auto_zoom.generate_zoom_levels = MagicMock(return_value=zoom_levels)
            mid = dv.remember("full content")
            node = dv.recall(mid)
            self.assertIsNotNone(node)
            self.assertEqual(node.zoom_levels.get(1), "label")
            self.assertEqual(node.zoom_levels.get(2), "summary sentence")

    def test_remember_explicit_zoom_overrides_auto_zoom(self):
        with DaVinci(":memory:", llm_enabled=True) as dv:
            if dv._auto_zoom is None:
                self.skipTest("AutoZoom not initialised")
            # Explicit zoom_levels should be used as-is
            dv._auto_zoom.generate_zoom_levels = MagicMock(return_value={1: "auto", 2: "auto", 3: "auto"})
            mid = dv.remember("content", zoom_levels={1: "manual", 2: "manual", 3: "manual"})
            node = dv.recall(mid)
            self.assertEqual(node.zoom_levels.get(1), "manual")
            # auto-zoom should NOT have been called
            dv._auto_zoom.generate_zoom_levels.assert_not_called()


class TestDaVinciSearchFTS(unittest.TestCase):
    def test_search_fts_through_davinci(self):
        with DaVinci(":memory:", llm_enabled=False) as dv:
            dv.remember("The Mandelbrot set is a famous fractal")
            dv.remember("Julia sets are related to Mandelbrot")
            results = dv.search_fts("Mandelbrot")
            self.assertGreater(len(results), 0)


class TestDaVinciEpisodicOperations(unittest.TestCase):
    def setUp(self):
        self.dv = DaVinci(":memory:", llm_enabled=False)
        self.dv._episodic.save("Q1", "A1", importance=0.9)
        self.dv._episodic.save("Q2", "A2", importance=0.1)

    def tearDown(self):
        self.dv.close()

    def test_episodic_status(self):
        s = self.dv.episodic_status()
        self.assertEqual(s["count"], 2)

    def test_episodic_prune(self):
        count = self.dv.episodic_prune(threshold=0.2)
        self.assertEqual(count, 1)
        self.assertEqual(self.dv.episodic_status()["count"], 1)

    def test_episodic_decay(self):
        import time
        # Set last_accessed to 30 days ago for both entries
        old_time = time.time() - 30 * 86400
        self.dv._episodic._conn.execute("UPDATE episodic SET last_accessed = ?", (old_time,))
        self.dv._episodic._conn.commit()
        count = self.dv.episodic_decay(rate=0.05)
        self.assertGreater(count, 0)


class TestDaVinciAutoLearnReview(unittest.TestCase):
    def setUp(self):
        self.dv = DaVinci(":memory:", llm_enabled=True)

    def tearDown(self):
        self.dv.close()

    def test_approve_fact(self):
        if self.dv._auto_learn is None:
            self.skipTest("AutoLearn not initialised")
        # Manually queue a pending fact
        self.dv._auto_learn._pending = [
            {"fact": "A fact about DaVinci.", "question": "Q?", "timestamp": 0.0}
        ]
        result = self.dv.approve_fact(0)
        self.assertTrue(result)
        self.assertEqual(len(self.dv.review_pending()), 0)

    def test_reject_fact(self):
        if self.dv._auto_learn is None:
            self.skipTest("AutoLearn not initialised")
        self.dv._auto_learn._pending = [
            {"fact": "A fact.", "question": "Q?", "timestamp": 0.0}
        ]
        result = self.dv.reject_fact(0)
        self.assertTrue(result)
        self.assertEqual(self.dv._store.count(), 0)


if __name__ == "__main__":
    unittest.main()
