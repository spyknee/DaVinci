"""
Tests for davinci.memory.episodic — Episodic Memory Store
"""

from __future__ import annotations

import time
import unittest

from davinci.memory.episodic import EpisodicStore
from davinci.memory.store import MemoryStore


class TestEpisodicStoreSaveAndRetrieve(unittest.TestCase):
    def setUp(self):
        self.store = EpisodicStore(":memory:")

    def tearDown(self):
        self.store.close()

    def test_save_returns_id(self):
        eid = self.store.save("What is Python?", "A high-level language.")
        self.assertIsInstance(eid, str)
        self.assertTrue(len(eid) > 0)

    def test_save_multiple(self):
        id1 = self.store.save("Q1", "A1")
        id2 = self.store.save("Q2", "A2")
        self.assertNotEqual(id1, id2)
        self.assertEqual(self.store.count(), 2)

    def test_retrieve_by_question(self):
        self.store.save("What is Python?", "A high-level programming language.")
        results = self.store.retrieve("Python")
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["question"], "What is Python?")

    def test_retrieve_by_answer(self):
        self.store.save("Tell me about fractals", "Fractals are self-similar patterns.")
        results = self.store.retrieve("self-similar")
        self.assertGreater(len(results), 0)

    def test_retrieve_returns_dicts(self):
        self.store.save("Q", "A")
        results = self.store.retrieve("Q")
        self.assertIsInstance(results, list)
        if results:
            self.assertIn("question", results[0])
            self.assertIn("answer", results[0])
            self.assertIn("importance", results[0])

    def test_retrieve_boosts_importance(self):
        self.store.save("Q", "A")
        # Get initial importance
        results_before = self.store.retrieve("Q")
        initial_importance = results_before[0]["importance"]
        # Retrieve again — should boost
        results_after = self.store.retrieve("Q")
        self.assertGreaterEqual(results_after[0]["importance"], initial_importance)

    def test_retrieve_limit(self):
        for i in range(10):
            self.store.save(f"Question {i}", f"Answer {i}")
        results = self.store.retrieve("Question", limit=3)
        self.assertLessEqual(len(results), 3)

    def test_retrieve_no_match(self):
        self.store.save("Q", "A")
        results = self.store.retrieve("xyznotfoundxyz")
        self.assertEqual(results, [])


class TestEpisodicStoreDecay(unittest.TestCase):
    def setUp(self):
        self.store = EpisodicStore(":memory:")

    def tearDown(self):
        self.store.close()

    def test_decay_reduces_importance(self):
        eid = self.store.save("Q", "A", importance=0.9)
        # Manually set last_accessed to 30 days ago
        old_time = time.time() - 30 * 86400
        self.store._conn.execute(
            "UPDATE episodic SET last_accessed = ? WHERE id = ?",
            (old_time, eid),
        )
        self.store._conn.commit()
        count = self.store.decay(rate_per_day=0.05)
        self.assertEqual(count, 1)

    def test_decay_returns_count(self):
        for _ in range(3):
            self.store.save("Q", "A", importance=0.9)
        # Set all to old access time
        old_time = time.time() - 20 * 86400
        self.store._conn.execute("UPDATE episodic SET last_accessed = ?", (old_time,))
        self.store._conn.commit()
        count = self.store.decay(rate_per_day=0.05)
        self.assertEqual(count, 3)

    def test_decay_does_not_go_below_zero(self):
        eid = self.store.save("Q", "A", importance=0.1)
        old_time = time.time() - 1000 * 86400
        self.store._conn.execute(
            "UPDATE episodic SET last_accessed = ? WHERE id = ?",
            (old_time, eid),
        )
        self.store._conn.commit()
        self.store.decay(rate_per_day=0.05)
        row = self.store._conn.execute(
            "SELECT importance FROM episodic WHERE id = ?", (eid,)
        ).fetchone()
        self.assertGreaterEqual(row[0], 0.0)


class TestEpisodicStorePrune(unittest.TestCase):
    def setUp(self):
        self.store = EpisodicStore(":memory:")

    def tearDown(self):
        self.store.close()

    def test_prune_removes_low_importance(self):
        self.store.save("Q1", "A1", importance=0.1)
        self.store.save("Q2", "A2", importance=0.9)
        count = self.store.prune(threshold=0.2)
        self.assertEqual(count, 1)
        self.assertEqual(self.store.count(), 1)

    def test_prune_returns_count(self):
        for _ in range(4):
            self.store.save("Q", "A", importance=0.05)
        count = self.store.prune(threshold=0.2)
        self.assertEqual(count, 4)

    def test_prune_keeps_above_threshold(self):
        self.store.save("Q", "A", importance=0.5)
        count = self.store.prune(threshold=0.2)
        self.assertEqual(count, 0)
        self.assertEqual(self.store.count(), 1)


class TestEpisodicStoreCountAndStatus(unittest.TestCase):
    def setUp(self):
        self.store = EpisodicStore(":memory:")

    def tearDown(self):
        self.store.close()

    def test_count_empty(self):
        self.assertEqual(self.store.count(), 0)

    def test_count_after_save(self):
        self.store.save("Q", "A")
        self.assertEqual(self.store.count(), 1)

    def test_status_keys(self):
        s = self.store.status()
        self.assertIn("count", s)
        self.assertIn("avg_importance", s)
        self.assertIn("oldest_timestamp", s)

    def test_status_count_matches(self):
        self.store.save("Q", "A")
        self.store.save("Q2", "A2")
        self.assertEqual(self.store.status()["count"], 2)

    def test_status_avg_importance(self):
        self.store.save("Q1", "A1", importance=0.4)
        self.store.save("Q2", "A2", importance=0.6)
        s = self.store.status()
        self.assertAlmostEqual(s["avg_importance"], 0.5, places=5)


class TestEpisodicStoreClear(unittest.TestCase):
    def setUp(self):
        self.store = EpisodicStore(":memory:")

    def tearDown(self):
        self.store.close()

    def test_clear_returns_count(self):
        self.store.save("Q1", "A1")
        self.store.save("Q2", "A2")
        count = self.store.clear()
        self.assertEqual(count, 2)

    def test_clear_empties_store(self):
        self.store.save("Q", "A")
        self.store.clear()
        self.assertEqual(self.store.count(), 0)


class TestEpisodicStoreContextManager(unittest.TestCase):
    def test_context_manager(self):
        with EpisodicStore(":memory:") as es:
            eid = es.save("Q", "A")
            self.assertIsNotNone(eid)
            self.assertEqual(es.count(), 1)
        # After __exit__ the connection should be closed
        with self.assertRaises(Exception):
            es.count()


class TestEpisodicSharesDBWithMemoryStore(unittest.TestCase):
    def test_shared_db_path(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            # Both stores pointing at same file
            with MemoryStore(db_path=db_path) as ms:
                mid = ms.store("hello world")

            with EpisodicStore(db_path=db_path) as es:
                eid = es.save("Q", "A")
                self.assertEqual(es.count(), 1)

            # Re-open MemoryStore — episodic table should not affect memories
            with MemoryStore(db_path=db_path) as ms2:
                node = ms2.retrieve(mid)
                self.assertIsNotNone(node)
        finally:
            os.unlink(db_path)


if __name__ == "__main__":
    unittest.main()
