"""
Tests for davinci.llm.auto_learn — AutoLearn Pipeline
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock

from davinci.llm.auto_learn import AutoLearn
from davinci.llm.backend import LLMBackend
from davinci.memory.store import MemoryStore


def _make_llm(response: str) -> LLMBackend:
    backend = MagicMock(spec=LLMBackend)
    backend.chat.return_value = response
    return backend


class TestExtractFactsHeuristic(unittest.TestCase):
    """Fact extraction without an LLM (heuristic mode)."""

    def setUp(self):
        self.store = MemoryStore(":memory:")
        self.learn = AutoLearn(self.store)

    def tearDown(self):
        self.store.close()

    def test_extract_long_sentences(self):
        answer = (
            "Python is an interpreted, high-level, general-purpose programming language. "
            "It was created by Guido van Rossum and first released in 1991."
        )
        facts = self.learn.extract_facts("What is Python?", answer)
        self.assertGreater(len(facts), 0)

    def test_does_not_extract_short_sentences(self):
        answer = "Yes. No. OK."
        facts = self.learn.extract_facts("Are you sure?", answer)
        self.assertEqual(facts, [])

    def test_facts_are_strings(self):
        answer = "The Mandelbrot set is a well-known mathematical fractal defined in the complex plane."
        facts = self.learn.extract_facts("What is Mandelbrot?", answer)
        for f in facts:
            self.assertIsInstance(f, str)


class TestExtractFactsWithLLM(unittest.TestCase):
    """Fact extraction using a mock LLM."""

    def setUp(self):
        self.store = MemoryStore(":memory:")

    def tearDown(self):
        self.store.close()

    def test_extract_with_llm_response(self):
        facts_list = ["Python is a programming language.", "Python was created in 1991."]
        llm = _make_llm(json.dumps(facts_list))
        learn = AutoLearn(self.store, llm_backend=llm)
        facts = learn.extract_facts("What is Python?", "Python is a language created in 1991.")
        self.assertEqual(facts, facts_list)

    def test_extract_with_markdown_fenced_json(self):
        facts_list = ["Fact one.", "Fact two."]
        llm = _make_llm(f"```json\n{json.dumps(facts_list)}\n```")
        learn = AutoLearn(self.store, llm_backend=llm)
        facts = learn.extract_facts("Q?", "A.")
        self.assertEqual(facts, facts_list)

    def test_extract_falls_back_to_heuristic_on_bad_json(self):
        llm = _make_llm("not valid json")
        learn = AutoLearn(self.store, llm_backend=llm)
        answer = "Python is a high-level programming language known for readability."
        facts = learn.extract_facts("What is Python?", answer)
        self.assertIsInstance(facts, list)

    def test_extract_falls_back_on_llm_unavailable(self):
        llm = _make_llm("[LLM unavailable: connection refused]")
        learn = AutoLearn(self.store, llm_backend=llm)
        answer = "Python is a high-level language created in 1991 by Guido van Rossum."
        facts = learn.extract_facts("Q?", answer)
        self.assertIsInstance(facts, list)


class TestPendingFactsQueue(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(":memory:")
        llm = _make_llm(json.dumps(["Fact A is true.", "Fact B is also true."]))
        self.learn = AutoLearn(self.store, llm_backend=llm)

    def tearDown(self):
        self.store.close()

    def test_learn_auto_store_false_queues_pending(self):
        self.learn.learn("Q?", "A.", auto_store=False)
        self.assertEqual(len(self.learn.pending), 2)

    def test_learn_auto_store_true_stores_immediately(self):
        self.learn.learn("Q?", "A.", auto_store=True)
        self.assertEqual(len(self.learn.pending), 0)
        self.assertEqual(self.store.count(), 2)

    def test_pending_returns_copy(self):
        self.learn.learn("Q?", "A.", auto_store=False)
        pending = self.learn.pending
        self.assertIsInstance(pending, list)
        # Modifying the returned list should not affect internal state
        pending.clear()
        self.assertEqual(len(self.learn.pending), 2)


class TestApproveAndReject(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(":memory:")
        llm = _make_llm(json.dumps(["Fact one here.", "Fact two here.", "Fact three here."]))
        self.learn = AutoLearn(self.store, llm_backend=llm)
        self.learn.learn("Q?", "A.", auto_store=False)

    def tearDown(self):
        self.store.close()

    def test_approve_stores_fact(self):
        self.learn.approve(0)
        self.assertEqual(self.store.count(), 1)

    def test_approve_removes_from_pending(self):
        initial_count = len(self.learn.pending)
        self.learn.approve(0)
        self.assertEqual(len(self.learn.pending), initial_count - 1)

    def test_approve_invalid_index_returns_false(self):
        result = self.learn.approve(99)
        self.assertFalse(result)

    def test_reject_removes_from_pending(self):
        initial_count = len(self.learn.pending)
        self.learn.reject(0)
        self.assertEqual(len(self.learn.pending), initial_count - 1)

    def test_reject_does_not_store(self):
        self.learn.reject(0)
        self.assertEqual(self.store.count(), 0)

    def test_reject_invalid_index_returns_false(self):
        result = self.learn.reject(99)
        self.assertFalse(result)


class TestApproveAll(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(":memory:")
        llm = _make_llm(json.dumps(["F1.", "F2.", "F3."]))
        self.learn = AutoLearn(self.store, llm_backend=llm)
        self.learn.learn("Q?", "A.", auto_store=False)

    def tearDown(self):
        self.store.close()

    def test_approve_all_stores_all(self):
        count = self.learn.approve_all()
        self.assertEqual(count, 3)
        self.assertEqual(self.store.count(), 3)

    def test_approve_all_clears_pending(self):
        self.learn.approve_all()
        self.assertEqual(len(self.learn.pending), 0)


class TestPurge(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(":memory:")
        llm = _make_llm(json.dumps(["F1.", "F2."]))
        self.learn = AutoLearn(self.store, llm_backend=llm)
        self.learn.learn("Q?", "A.", auto_store=False)

    def tearDown(self):
        self.store.close()

    def test_purge_clears_pending(self):
        count = self.learn.purge()
        self.assertEqual(count, 2)
        self.assertEqual(len(self.learn.pending), 0)

    def test_purge_does_not_store(self):
        self.learn.purge()
        self.assertEqual(self.store.count(), 0)


if __name__ == "__main__":
    unittest.main()
