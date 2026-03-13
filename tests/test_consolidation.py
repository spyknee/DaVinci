"""
Unit tests for davinci.memory.consolidation — Layer 2 of DaVinci.

Run with:
    python -m pytest tests/test_consolidation.py -v
"""

import unittest

from davinci.memory.store import MemoryStore
from davinci.memory.consolidation import ConsolidationEngine


def _make_store() -> MemoryStore:
    """Return a fresh in-memory store for each test."""
    return MemoryStore(db_path=":memory:")


class TestConsolidate(unittest.TestCase):
    """consolidate() — frequency-based promotion."""

    def test_consolidate_returns_int(self):
        with _make_store() as store:
            engine = ConsolidationEngine(store)
            result = engine.consolidate()
            self.assertIsInstance(result, int)

    def test_consolidate_empty_store_returns_zero(self):
        with _make_store() as store:
            engine = ConsolidationEngine(store)
            self.assertEqual(engine.consolidate(), 0)

    def test_consolidate_promotes_high_frequency_memories(self):
        """Memories above threshold should have their zoom levels refined."""
        with _make_store() as store:
            long_content = "A" * 200  # longer than 100 chars
            mid = store.store(long_content)

            # Simulate high frequency by direct DB update
            with store._conn:
                store._conn.execute(
                    "UPDATE memories SET frequency = 10 WHERE id = ?", (mid,)
                )

            engine = ConsolidationEngine(store)
            promoted = engine.consolidate(strategy="frequency")
            self.assertEqual(promoted, 1)

            row = store._conn.execute(
                "SELECT zoom_level_1, zoom_level_2, zoom_level_3 FROM memories WHERE id = ?",
                (mid,),
            ).fetchone()

            # zoom_level_1 is first 50 chars, zoom_level_2 is first 100 chars
            self.assertEqual(row["zoom_level_1"], long_content[:50])
            self.assertEqual(row["zoom_level_2"], long_content[:100])
            self.assertEqual(row["zoom_level_3"], long_content)

    def test_consolidate_does_not_promote_low_frequency(self):
        """Memories at or below the threshold should not be changed."""
        with _make_store() as store:
            mid = store.store("low frequency content")
            # frequency defaults to 0, threshold is 5

            engine = ConsolidationEngine(store)
            promoted = engine.consolidate(strategy="frequency")
            self.assertEqual(promoted, 0)

    def test_consolidate_unknown_strategy_raises(self):
        with _make_store() as store:
            engine = ConsolidationEngine(store)
            with self.assertRaises(ValueError):
                engine.consolidate(strategy="unknown_strategy")

    def test_consolidate_multiple_promotions(self):
        with _make_store() as store:
            ids = [store.store(f"content {i} " + "x" * 150) for i in range(5)]
            # Set all to high frequency
            for mid in ids:
                with store._conn:
                    store._conn.execute(
                        "UPDATE memories SET frequency = 10 WHERE id = ?", (mid,)
                    )

            engine = ConsolidationEngine(store)
            count = engine.consolidate(strategy="frequency")
            self.assertEqual(count, 5)


class TestMergeSimilar(unittest.TestCase):
    """merge_similar() — duplicate content merging."""

    def test_merge_identical_content(self):
        """Two identical memories should be merged into one."""
        with _make_store() as store:
            store.store("the quick brown fox jumps over the lazy dog")
            store.store("the quick brown fox jumps over the lazy dog")

            engine = ConsolidationEngine(store)
            merges = engine.merge_similar(similarity_threshold=0.8)
            self.assertEqual(merges, 1)

            # Only one memory should remain
            s = store.stats()
            self.assertEqual(s["total"], 1)

    def test_merge_combines_frequencies(self):
        """Merged result should have the sum of both frequencies."""
        with _make_store() as store:
            mid1 = store.store("hello world how are you today")
            mid2 = store.store("hello world how are you today")

            # Give each a distinct frequency
            with store._conn:
                store._conn.execute(
                    "UPDATE memories SET frequency = 3 WHERE id = ?", (mid1,)
                )
                store._conn.execute(
                    "UPDATE memories SET frequency = 7 WHERE id = ?", (mid2,)
                )

            engine = ConsolidationEngine(store)
            engine.merge_similar(similarity_threshold=0.8)

            row = store._conn.execute("SELECT frequency FROM memories").fetchone()
            self.assertEqual(row["frequency"], 10)

    def test_merge_dissimilar_content_not_merged(self):
        """Memories with very different content should not be merged."""
        with _make_store() as store:
            store.store("apple orange banana grape")
            store.store("car truck motorcycle bicycle")

            engine = ConsolidationEngine(store)
            merges = engine.merge_similar(similarity_threshold=0.8)
            self.assertEqual(merges, 0)

            s = store.stats()
            self.assertEqual(s["total"], 2)

    def test_merge_empty_store_returns_zero(self):
        with _make_store() as store:
            engine = ConsolidationEngine(store)
            self.assertEqual(engine.merge_similar(), 0)

    def test_merge_no_similar_memories_returns_zero(self):
        with _make_store() as store:
            store.store("completely unique content about dogs")
            store.store("totally different words about astronomy")

            engine = ConsolidationEngine(store)
            result = engine.merge_similar(similarity_threshold=0.8)
            self.assertEqual(result, 0)

    def test_merge_returns_int(self):
        with _make_store() as store:
            store.store("test content one")
            engine = ConsolidationEngine(store)
            result = engine.merge_similar()
            self.assertIsInstance(result, int)

    def test_merge_threshold_at_boundary(self):
        """Similarity exactly at threshold triggers merge."""
        with _make_store() as store:
            # Both sentences share the same 4 words out of 4 unique words → J=1.0
            store.store("alpha beta gamma delta")
            store.store("alpha beta gamma delta")

            engine = ConsolidationEngine(store)
            merges = engine.merge_similar(similarity_threshold=1.0)
            self.assertEqual(merges, 1)

    def test_merge_multiple_duplicates(self):
        """Three identical memories: two merges should occur."""
        with _make_store() as store:
            for _ in range(3):
                store.store("same content repeated multiple times here")

            engine = ConsolidationEngine(store)
            merges = engine.merge_similar(similarity_threshold=0.8)
            self.assertEqual(merges, 2)

            # Only one memory should remain
            s = store.stats()
            self.assertEqual(s["total"], 1)


class TestConsolidationEngineInit(unittest.TestCase):
    """ConsolidationEngine initialisation."""

    def test_accepts_store_instance(self):
        with _make_store() as store:
            engine = ConsolidationEngine(store)
            self.assertIs(engine._store, store)


if __name__ == "__main__":
    unittest.main()
