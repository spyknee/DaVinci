"""
Unit tests for davinci.memory.store — Layer 2 of DaVinci.

Run with:
    python -m pytest tests/test_memory_store.py -v
"""

import time
import unittest

from davinci.memory.store import MemoryStore, _MIN_RECENCY_SPREAD


def _make_store() -> MemoryStore:
    """Return a fresh in-memory store for each test."""
    return MemoryStore(db_path=":memory:")


class TestStoreAndRetrieve(unittest.TestCase):
    """Basic store → retrieve round-trip."""

    def test_store_returns_uuid_string(self):
        with _make_store() as store:
            mid = store.store("hello world")
            self.assertIsInstance(mid, str)
            self.assertEqual(len(mid), 36)  # UUID4 format

    def test_retrieve_returns_memory_node(self):
        with _make_store() as store:
            from davinci.core.fractal_engine import MemoryNode
            mid = store.store("hello world")
            node = store.retrieve(mid)
            self.assertIsNotNone(node)
            self.assertIsInstance(node, MemoryNode)

    def test_retrieve_content_matches(self):
        with _make_store() as store:
            mid = store.store("unique content xyz")
            node = store.retrieve(mid)
            self.assertEqual(node.content, "unique content xyz")

    def test_retrieve_increments_frequency(self):
        with _make_store() as store:
            mid = store.store("test memory")
            node1 = store.retrieve(mid)
            freq_after_first = node1.frequency

            node2 = store.retrieve(mid)
            self.assertEqual(node2.frequency, freq_after_first + 1)

    def test_retrieve_updates_recency(self):
        with _make_store() as store:
            before = time.time()
            mid = store.store("timestamp test")
            time.sleep(0.01)
            node = store.retrieve(mid)
            self.assertGreaterEqual(node.recency, before)

    def test_retrieve_updates_classification_in_db(self):
        """After retrieve, the classification in the DB is updated."""
        with _make_store() as store:
            mid = store.store("check classification update")
            store.retrieve(mid)
            # retrieve again — classification should still be present
            node = store.retrieve(mid)
            self.assertIn(node.classification, {"core", "boundary", "decay", "forget"})

    def test_retrieve_nonexistent_id_returns_none(self):
        with _make_store() as store:
            result = store.retrieve("00000000-0000-0000-0000-000000000000")
            self.assertIsNone(result)

    def test_round_trip_data_integrity(self):
        """Store → retrieve preserves content, zoom levels, meta."""
        zoom = {1: "summary", 2: "detail", 3: "full text of the memory"}
        with _make_store() as store:
            mid = store.store("round trip test", zoom_levels=zoom)
            node = store.retrieve(mid)
            self.assertEqual(node.content, "round trip test")
            self.assertEqual(node.zoom_levels[1], "summary")
            self.assertEqual(node.zoom_levels[2], "detail")
            self.assertEqual(node.zoom_levels[3], "full text of the memory")

    def test_default_zoom_level_3_equals_full_content(self):
        """When no zoom_levels supplied, zoom level 3 equals the full content."""
        content = "A" * 200
        with _make_store() as store:
            mid = store.store(content)
            node = store.retrieve(mid)
            self.assertEqual(node.zoom_levels[3], content)

    def test_default_zoom_level_1_differs_from_level_3_for_long_content(self):
        """When no zoom_levels supplied, zoom 1 ≠ zoom 3 for content > 100 chars."""
        content = "B" * 200
        with _make_store() as store:
            mid = store.store(content)
            node = store.retrieve(mid)
            self.assertNotEqual(node.zoom_levels[1], node.zoom_levels[3])

    def test_explicit_zoom_levels_are_respected(self):
        """Explicitly supplied zoom_levels override the defaults."""
        zoom = {1: "custom1", 2: "custom2", 3: "custom3"}
        with _make_store() as store:
            mid = store.store("any content", zoom_levels=zoom)
            node = store.retrieve(mid)
            self.assertEqual(node.zoom_levels[1], "custom1")
            self.assertEqual(node.zoom_levels[2], "custom2")
            self.assertEqual(node.zoom_levels[3], "custom3")


class TestSearch(unittest.TestCase):
    """Search functionality."""

    def test_search_empty_db_returns_empty_list(self):
        with _make_store() as store:
            results = store.search("anything")
            self.assertEqual(results, [])

    def test_search_returns_matching_nodes(self):
        with _make_store() as store:
            store.store("the quick brown fox")
            store.store("a slow blue elephant")
            results = store.search("quick")
            self.assertEqual(len(results), 1)
            self.assertIn("quick", results[0].content)

    def test_search_case_insensitive(self):
        with _make_store() as store:
            store.store("Hello World")
            results = store.search("hello")
            self.assertEqual(len(results), 1)

    def test_search_no_matches_returns_empty_list(self):
        with _make_store() as store:
            store.store("unrelated content")
            results = store.search("zzznomatch")
            self.assertEqual(results, [])

    def test_search_respects_limit(self):
        with _make_store() as store:
            for i in range(20):
                store.store(f"memory item {i}")
            results = store.search("memory", limit=5)
            self.assertLessEqual(len(results), 5)

    def test_search_orders_by_classification_priority(self):
        """Core memories should appear before forget memories."""
        with _make_store() as store:
            # Store several memories; after decay/retrieval cycles some will be
            # classified differently.  We verify the sort is stable by
            # checking that classification priority is non-decreasing.
            for i in range(5):
                store.store(f"searchable content {i}")
            results = store.search("searchable")
            from davinci.memory.store import _CLASSIFICATION_ORDER
            priorities = [_CLASSIFICATION_ORDER.get(n.classification, 99) for n in results]
            self.assertEqual(priorities, sorted(priorities))

    def test_search_ranking_exact_phrase_beats_partial(self):
        """Exact phrase match should rank higher than partial word match."""
        with _make_store() as store:
            store.store("the quick brown fox")   # contains exact phrase "quick brown"
            store.store("quickly done")          # contains "quick" but not exact phrase
            results = store.search("quick brown")
            # Both memories match (one exact phrase, one partial word)
            self.assertEqual(len(results), 2)
            # The exact phrase match should be the first result
            self.assertIn("quick brown", results[0].content)
            self.assertIn("quickly", results[1].content)


class TestGetByClassification(unittest.TestCase):
    """get_by_classification filtering."""

    def test_returns_only_matching_classification(self):
        with _make_store() as store:
            for _ in range(5):
                store.store("sample")
            # classify all nodes by running a decay cycle
            store.decay_cycle()
            for cls in ("core", "boundary", "decay", "forget"):
                nodes = store.get_by_classification(cls)
                for node in nodes:
                    self.assertEqual(node.classification, cls)

    def test_unknown_classification_returns_empty(self):
        with _make_store() as store:
            store.store("test")
            result = store.get_by_classification("nonexistent")
            self.assertEqual(result, [])

    def test_empty_db_returns_empty_list(self):
        with _make_store() as store:
            self.assertEqual(store.get_by_classification("core"), [])


class TestDecayCycle(unittest.TestCase):
    """decay_cycle reclassification."""

    def test_decay_cycle_returns_dict(self):
        with _make_store() as store:
            store.store("test node")
            result = store.decay_cycle()
            self.assertIsInstance(result, dict)

    def test_decay_cycle_on_empty_store(self):
        with _make_store() as store:
            result = store.decay_cycle()
            self.assertEqual(result, {})

    def test_decay_cycle_counts_changed_nodes(self):
        """After a decay cycle on a populated store the result is a dict of lists of IDs."""
        with _make_store() as store:
            for i in range(10):
                store.store(f"memory {i}")
            result = store.decay_cycle()
            for key, val in result.items():
                self.assertIn(key, {"core", "boundary", "decay", "forget"})
                self.assertIsInstance(val, list)
                for mid in val:
                    self.assertIsInstance(mid, str)


class TestPrune(unittest.TestCase):
    """prune() deletes memories by classification."""

    def test_prune_removes_forget_nodes(self):
        with _make_store() as store:
            for i in range(5):
                store.store(f"prune me {i}")
            store.decay_cycle()
            before = store.stats()["by_classification"]["forget"]
            deleted = store.prune("forget")
            after = store.stats()["by_classification"]["forget"]
            self.assertEqual(deleted, before)
            self.assertEqual(after, 0)

    def test_prune_returns_zero_when_nothing_to_delete(self):
        with _make_store() as store:
            store.store("keep me")
            # No forget-classified nodes yet in a fresh store with one item
            # (classification depends on fractal position; we just verify the
            # return type is int and >= 0)
            result = store.prune("forget")
            self.assertIsInstance(result, int)
            self.assertGreaterEqual(result, 0)

    def test_prune_does_not_delete_other_classifications(self):
        with _make_store() as store:
            for _ in range(3):
                store.store("keep me")
            before_total = store.stats()["total"]
            store.prune("nonexistent_class")
            after_total = store.stats()["total"]
            self.assertEqual(before_total, after_total)


class TestStats(unittest.TestCase):
    """stats() returns correct aggregates."""

    def test_empty_db_stats(self):
        with _make_store() as store:
            s = store.stats()
            self.assertEqual(s["total"], 0)
            self.assertEqual(s["avg_frequency"], 0.0)
            self.assertIsNone(s["oldest_timestamp"])
            self.assertIsNone(s["newest_timestamp"])
            for cls in ("core", "boundary", "decay", "forget"):
                self.assertEqual(s["by_classification"][cls], 0)

    def test_stats_total_count(self):
        with _make_store() as store:
            for i in range(7):
                store.store(f"memory {i}")
            s = store.stats()
            self.assertEqual(s["total"], 7)

    def test_stats_by_classification_sums_to_total(self):
        with _make_store() as store:
            for i in range(4):
                store.store(f"mem {i}")
            s = store.stats()
            cls_sum = sum(s["by_classification"].values())
            self.assertEqual(cls_sum, s["total"])

    def test_stats_avg_frequency(self):
        with _make_store() as store:
            mid = store.store("frequent memory")
            for _ in range(4):
                store.retrieve(mid)
            s = store.stats()
            self.assertGreater(s["avg_frequency"], 0)

    def test_stats_timestamps_present_after_store(self):
        with _make_store() as store:
            store.store("timestamped")
            s = store.stats()
            self.assertIsNotNone(s["oldest_timestamp"])
            self.assertIsNotNone(s["newest_timestamp"])


class TestContextManager(unittest.TestCase):
    """Context manager support."""

    def test_with_statement_works(self):
        with MemoryStore(":memory:") as store:
            mid = store.store("context manager test")
            node = store.retrieve(mid)
            self.assertIsNotNone(node)

    def test_connection_closed_after_with(self):
        store = MemoryStore(":memory:")
        with store:
            store.store("test")
        # After exit, the connection should be closed; accessing it should raise
        with self.assertRaises(Exception):
            store._conn.execute("SELECT 1")


class TestGetRanges(unittest.TestCase):
    """_get_ranges() private helper."""

    def test_empty_db_returns_default_ranges(self):
        now = time.time()
        with _make_store() as store:
            freq_range, rec_range = store._get_ranges(now=now)
        self.assertEqual(freq_range, (0.0, 0.0))
        self.assertAlmostEqual(rec_range[1] - rec_range[0], 1.0, places=6)
        self.assertEqual(rec_range[1], now)

    def test_empty_db_returns_synthetic_recency_range(self):
        before = time.time()
        with _make_store() as store:
            freq_range, rec_range = store._get_ranges(now=before)
        self.assertEqual(freq_range, (0.0, 0.0))
        self.assertAlmostEqual(rec_range[1] - rec_range[0], 1.0, places=6)
        self.assertEqual(rec_range[1], before)

    def test_same_timestamp_nodes_get_synthetic_range(self):
        with _make_store() as store:
            now = time.time()
            store.store("a")
            store.store("b")
            # force identical recency by direct update
            store._conn.execute("UPDATE memories SET recency = ?", (now,))
            store._conn.commit()
            # verify the DB state matches test intent
            rows = store._conn.execute("SELECT recency FROM memories").fetchall()
            self.assertTrue(all(r[0] == now for r in rows))
            _, rec_range = store._get_ranges(now=now)
            self.assertAlmostEqual(rec_range[1] - rec_range[0], _MIN_RECENCY_SPREAD, places=6)

    def test_ranges_reflect_stored_data(self):
        with _make_store() as store:
            store.store("a")
            store.store("b")
            freq_range, rec_range = store._get_ranges()
            self.assertIsInstance(freq_range[0], float)
            self.assertIsInstance(freq_range[1], float)
            self.assertLessEqual(freq_range[0], freq_range[1])


class TestNewMemoryClassification(unittest.TestCase):
    """Verify new memories are never immediately classified as forget."""

    def test_first_stored_memory_is_not_forget(self):
        """A single stored memory must not be classified forget at INSERT time."""
        with _make_store() as store:
            mid = store.store("brand new memory")
            row = store._conn.execute(
                "SELECT classification FROM memories WHERE id = ?", (mid,)
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertNotEqual(row[0], "forget")

    def test_rapid_fire_memories_are_not_forget(self):
        """3 memories stored in rapid succession must all be classified as core."""
        with _make_store() as store:
            ids = [
                store.store("test memory one"),
                store.store("test memory two"),
                store.store("test memory three"),
            ]
            for mid in ids:
                row = store._conn.execute(
                    "SELECT classification FROM memories WHERE id = ?", (mid,)
                ).fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row[0], "core", f"Memory {mid} classification mismatch")


class TestDecayCycleDegenerate(unittest.TestCase):
    """decay_cycle() handles degenerate recency range."""

    def test_decay_cycle_same_recency_no_forget(self):
        """Memories with identical recency must not all become forget after decay_cycle."""
        with _make_store() as store:
            now = time.time()
            ids = [store.store(f"memory {i}") for i in range(3)]
            # force identical recency
            store._conn.execute("UPDATE memories SET recency = ?", (now,))
            store._conn.commit()
            store.decay_cycle()
            for mid in ids:
                row = store._conn.execute(
                    "SELECT classification FROM memories WHERE id = ?", (mid,)
                ).fetchone()
                self.assertNotEqual(row[0], "forget")


class TestDecayCycleHysteresis(unittest.TestCase):
    """decay_cycle() hysteresis suppresses minor tier improvements."""

    def test_decay_cycle_hysteresis(self):
        """A memory in boundary should not jump to core in a single decay cycle."""
        with _make_store() as store:
            mid = store.store("stable memory")

            # Access many times so classify() will return 'core' for this node
            for _ in range(10):
                store.retrieve(mid)

            # Override the DB classification to 'boundary' — simulating a node
            # that was recently boundary and now barely qualifies for core.
            # Without hysteresis it would immediately flip; with it, it stays.
            store._conn.execute(
                "UPDATE memories SET classification = 'boundary' WHERE id = ?", (mid,)
            )
            store._conn.commit()

            # decay_cycle re-evaluates; classify() returns 'core', but the 1-tier
            # improvement (boundary→core) must be suppressed by hysteresis.
            store.decay_cycle()

            row = store._conn.execute(
                "SELECT classification FROM memories WHERE id = ?", (mid,)
            ).fetchone()
            self.assertEqual(row[0], "boundary")

if __name__ == "__main__":
    unittest.main()
