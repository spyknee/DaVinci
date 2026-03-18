"""
Unit tests for davinci.interface.api — DaVinci Python API (Layer 3).

Run with:
    python -m pytest tests/test_api.py -v
"""

from __future__ import annotations

import unittest

from davinci import DaVinci
from davinci.core.fractal_engine import MemoryNode
from davinci.interface.base import BaseInterface


class TestDaVinciIsBaseInterface(unittest.TestCase):
    """DaVinci must implement BaseInterface."""

    def test_is_subclass(self):
        self.assertTrue(issubclass(DaVinci, BaseInterface))

    def test_instance_is_base_interface(self):
        with DaVinci(":memory:") as dv:
            self.assertIsInstance(dv, BaseInterface)


class TestRememberRecall(unittest.TestCase):
    """remember → recall round-trip."""

    def test_remember_returns_uuid_string(self):
        with DaVinci(":memory:") as dv:
            mid = dv.remember("hello world")
            self.assertIsInstance(mid, str)
            self.assertEqual(len(mid), 36)

    def test_recall_returns_memory_node(self):
        with DaVinci(":memory:") as dv:
            mid = dv.remember("test content")
            node = dv.recall(mid)
            self.assertIsNotNone(node)
            self.assertIsInstance(node, MemoryNode)

    def test_recall_content_matches(self):
        with DaVinci(":memory:") as dv:
            mid = dv.remember("unique fractal content xyz")
            node = dv.recall(mid)
            self.assertEqual(node.content, "unique fractal content xyz")

    def test_recall_increments_frequency(self):
        with DaVinci(":memory:") as dv:
            mid = dv.remember("freq test")
            node1 = dv.recall(mid)
            freq1 = node1.frequency
            node2 = dv.recall(mid)
            self.assertEqual(node2.frequency, freq1 + 1)

    def test_recall_nonexistent_id_returns_none(self):
        with DaVinci(":memory:") as dv:
            result = dv.recall("00000000-0000-0000-0000-000000000000")
            self.assertIsNone(result)

    def test_remember_with_zoom_levels(self):
        with DaVinci(":memory:") as dv:
            mid = dv.remember(
                "detailed content",
                zoom_levels={1: "summary", 2: "more detail", 3: "full content"},
            )
            node = dv.recall(mid)
            self.assertEqual(node.zoom_levels[1], "summary")
            self.assertEqual(node.zoom_levels[2], "more detail")
            self.assertEqual(node.zoom_levels[3], "full content")

    def test_remember_with_meta(self):
        with DaVinci(":memory:") as dv:
            mid = dv.remember("meta test", meta={"source": "unit_test"})
            self.assertIsNotNone(mid)


class TestSearch(unittest.TestCase):
    """search() returns relevant results."""

    def test_search_finds_stored_memory(self):
        with DaVinci(":memory:") as dv:
            dv.remember("the quick brown fox")
            results = dv.search("quick brown")
            self.assertGreater(len(results), 0)
            self.assertTrue(any("quick brown" in n.content for n in results))

    def test_search_returns_memory_nodes(self):
        with DaVinci(":memory:") as dv:
            dv.remember("fractal geometry")
            results = dv.search("fractal")
            for node in results:
                self.assertIsInstance(node, MemoryNode)

    def test_search_no_results(self):
        with DaVinci(":memory:") as dv:
            dv.remember("something completely unrelated")
            results = dv.search("xyzzy_no_match")
            self.assertEqual(results, [])

    def test_search_limit(self):
        with DaVinci(":memory:") as dv:
            for i in range(5):
                dv.remember(f"searchable memory number {i}")
            results = dv.search("searchable memory", limit=3)
            self.assertLessEqual(len(results), 3)

    def test_search_case_insensitive(self):
        with DaVinci(":memory:") as dv:
            dv.remember("Mandelbrot Set Theory")
            results = dv.search("mandelbrot")
            self.assertGreater(len(results), 0)


class TestForget(unittest.TestCase):
    """forget() prunes memories by classification."""

    def test_forget_returns_count(self):
        with DaVinci(":memory:") as dv:
            count = dv.forget()
            self.assertIsInstance(count, int)
            self.assertGreaterEqual(count, 0)

    def test_forget_removes_forget_classified_memories(self):
        with DaVinci(":memory:") as dv:
            # Store several memories — fresh stores with low frequency/recency
            # typically land in "forget" after a decay cycle
            for i in range(5):
                dv.remember(f"memory to potentially forget {i}")
            # Run decay to allow reclassification
            dv.decay()
            before = dv.stats()["by_classification"]["forget"]
            deleted = dv.forget("forget")
            self.assertEqual(deleted, before)
            after = dv.stats()["by_classification"]["forget"]
            self.assertEqual(after, 0)

    def test_forget_custom_classification(self):
        with DaVinci(":memory:") as dv:
            count = dv.forget("decay")
            self.assertIsInstance(count, int)


class TestDecay(unittest.TestCase):
    """decay() runs without error and returns a dict."""

    def test_decay_returns_dict(self):
        with DaVinci(":memory:") as dv:
            result = dv.decay()
            self.assertIsInstance(result, dict)

    def test_decay_on_empty_store(self):
        with DaVinci(":memory:") as dv:
            result = dv.decay()
            self.assertEqual(result, {})

    def test_decay_values_are_lists(self):
        with DaVinci(":memory:") as dv:
            for i in range(3):
                dv.remember(f"decay test {i}")
            result = dv.decay()
            for v in result.values():
                self.assertIsInstance(v, list)


class TestStats(unittest.TestCase):
    """stats() returns a valid statistics dict."""

    def test_stats_returns_dict(self):
        with DaVinci(":memory:") as dv:
            s = dv.stats()
            self.assertIsInstance(s, dict)

    def test_stats_has_required_keys(self):
        with DaVinci(":memory:") as dv:
            s = dv.stats()
            self.assertIn("total", s)
            self.assertIn("by_classification", s)
            self.assertIn("avg_frequency", s)

    def test_stats_total_on_empty_store(self):
        with DaVinci(":memory:") as dv:
            s = dv.stats()
            self.assertEqual(s["total"], 0)

    def test_stats_total_increments(self):
        with DaVinci(":memory:") as dv:
            dv.remember("stats memory 1")
            dv.remember("stats memory 2")
            s = dv.stats()
            self.assertEqual(s["total"], 2)

    def test_stats_by_classification_is_dict(self):
        with DaVinci(":memory:") as dv:
            s = dv.stats()
            bc = s["by_classification"]
            self.assertIsInstance(bc, dict)
            for cls in ("core", "boundary", "decay", "forget"):
                self.assertIn(cls, bc)


class TestConsolidate(unittest.TestCase):
    """consolidate() runs without error."""

    def test_consolidate_returns_int(self):
        with DaVinci(":memory:") as dv:
            result = dv.consolidate()
            self.assertIsInstance(result, int)

    def test_consolidate_on_empty_store(self):
        with DaVinci(":memory:") as dv:
            result = dv.consolidate()
            self.assertEqual(result, 0)

    def test_consolidate_frequency_strategy(self):
        with DaVinci(":memory:") as dv:
            for i in range(3):
                dv.remember(f"consolidate test {i}")
            result = dv.consolidate(strategy="frequency")
            self.assertIsInstance(result, int)


class TestMergeSimilar(unittest.TestCase):
    """merge_similar() merges near-duplicate memories."""

    def test_merge_returns_int(self):
        with DaVinci(":memory:") as dv:
            result = dv.merge_similar()
            self.assertIsInstance(result, int)

    def test_merge_on_empty_store(self):
        with DaVinci(":memory:") as dv:
            result = dv.merge_similar()
            self.assertEqual(result, 0)

    def test_merge_identical_content(self):
        with DaVinci(":memory:") as dv:
            dv.remember("the quick brown fox jumps over the lazy dog")
            dv.remember("the quick brown fox jumps over the lazy dog")
            merged = dv.merge_similar(threshold=0.9)
            self.assertGreaterEqual(merged, 1)


class TestMemories(unittest.TestCase):
    """memories() lists all (or filtered) memories."""

    def test_memories_returns_list(self):
        with DaVinci(":memory:") as dv:
            result = dv.memories()
            self.assertIsInstance(result, list)

    def test_memories_empty_store(self):
        with DaVinci(":memory:") as dv:
            result = dv.memories()
            self.assertEqual(result, [])

    def test_memories_contains_stored(self):
        with DaVinci(":memory:") as dv:
            dv.remember("listed memory")
            nodes = dv.memories()
            self.assertEqual(len(nodes), 1)
            self.assertEqual(nodes[0].content, "listed memory")

    def test_memories_filter_by_classification(self):
        with DaVinci(":memory:") as dv:
            for i in range(4):
                dv.remember(f"filter test {i}")
            dv.decay()
            # Filter by a valid classification — just check it returns a list
            for cls in ("core", "boundary", "decay", "forget"):
                result = dv.memories(classification=cls)
                self.assertIsInstance(result, list)

    def test_memories_nodes_have_id(self):
        with DaVinci(":memory:") as dv:
            dv.remember("node with id")
            nodes = dv.memories()
            self.assertTrue(hasattr(nodes[0], "id"))
            self.assertIsNotNone(nodes[0].id)


class TestContextManager(unittest.TestCase):
    """DaVinci supports the context manager protocol."""

    def test_context_manager_enter_returns_self(self):
        dv = DaVinci(":memory:")
        with dv as d:
            self.assertIs(d, dv)

    def test_context_manager_closes_on_exit(self):
        with DaVinci(":memory:") as dv:
            mid = dv.remember("context test")
            self.assertIsNotNone(mid)
        # After __exit__, the connection should be closed;
        # any further operation on the store should raise or be inert
        # (we just check no exception was raised during close)


if __name__ == "__main__":
    unittest.main()
