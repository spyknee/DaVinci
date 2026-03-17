"""Stress / torture tests for DaVinci memory subsystem.

Tests go beyond what ``tests/test_memory_store.py`` already covers: scale,
merge performance, decay graduation, full pipelines, concurrency, and edge
cases.  No production code is modified.

Run with::

    python -m pytest tests/test_torture.py -v

Important notes on the fractal model
--------------------------------------
``compute_c`` maps:
  real = normalize(freshness, 0, MAX_AGE, -2.0, 0.25)
  imag = normalize(frequency, freq_min, freq_max, -1.0, 1.0)

For a fresh memory to survive as ``'core'`` after ``decay_cycle()``:
  - real must be 0.25  (max freshness → just stored/accessed)
  - imag must be 0     (frequency at the *midpoint* of the range)
  → c = 0.25 + 0j  →  converges inside the Mandelbrot set  → ``'core'``

To achieve imag = 0 for memories at freq F, anchor memories must exist at
freq 2*F so that the global range becomes (0, 2F) and F normalises to 0.
Tests that verify "fresh memories survive decay" follow this pattern.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import unittest

from davinci.memory.consolidation import ConsolidationEngine
from davinci.memory.maintenance import MemoryMaintenance
from davinci.memory.store import MemoryStore, _MAX_RECENCY_AGE_SECONDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> MemoryStore:
    """Return a fresh in-memory store for each test."""
    return MemoryStore(db_path=":memory:")


def _old_ts() -> float:
    """Unix timestamp that is 31 days in the past."""
    return time.time() - 31 * 24 * 3600


class _ThreadSafeStore(MemoryStore):
    """MemoryStore that allows cross-thread SQLite access for concurrency tests."""

    def __init__(self) -> None:
        self._db_path = ":memory:"
        self._max_iter = 1000
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_schema()


# ---------------------------------------------------------------------------
# TestScale
# ---------------------------------------------------------------------------

class TestScale(unittest.TestCase):
    """High-volume store / search / decay operations."""

    def test_store_1000_memories_all_core(self):
        """1000 freshly stored memories must all have classification='core' in the DB."""
        with _make_store() as store:
            for i in range(1000):
                store.store(f"unique rapid fire memory number {i}")
            rows = store._conn.execute(
                "SELECT classification FROM memories"
            ).fetchall()
            self.assertEqual(len(rows), 1000)
            for row in rows:
                self.assertEqual(
                    row[0], "core",
                    f"Expected 'core', got {row[0]!r}",
                )

    def test_search_at_scale(self):
        """search('needle', limit=100) must return exactly 100 results, all containing 'needle'."""
        with _make_store() as store:
            for i in range(250):
                store.store(f"needle in a haystack memory number {i}")
            for i in range(250):
                store.store(f"haystack memory without the special term {i}")
            results = store.search("needle", limit=100)
            self.assertEqual(len(results), 100)
            for node in results:
                self.assertIn("needle", node.content.lower())

    def test_stats_by_classification_always_sums_to_total(self):
        """sum(by_classification.values()) == total at every step of a store/decay/prune cycle."""
        with _make_store() as store:
            for i in range(200):
                store.store(f"memory for stats scale test {i}")
            s = store.stats()
            self.assertEqual(sum(s["by_classification"].values()), s["total"])

            store.decay_cycle()
            s = store.stats()
            self.assertEqual(sum(s["by_classification"].values()), s["total"])

            store.prune("forget")
            s = store.stats()
            self.assertEqual(sum(s["by_classification"].values()), s["total"])

            for i in range(50):
                store.store(f"additional stats memory {i}")
            s = store.stats()
            self.assertEqual(sum(s["by_classification"].values()), s["total"])

            store.decay_cycle()
            s = store.stats()
            self.assertEqual(sum(s["by_classification"].values()), s["total"])

    def test_decay_cycle_at_scale(self):
        """100 old (freq=0) memories must become 'forget'; 100 fresh (freq=10) must not.

        Anchor memories at freq=20 are added so that the frequency range is
        (0, 20).  With that range:
          - old nodes  → imag=-1, real=-2  → c=-2-1j → escapes immediately → 'forget'
          - fresh nodes → imag=0,  real=0.25 → c=0.25+0j → inside set      → 'core'
        """
        with _make_store() as store:
            old_ids = [store.store(f"old memory to decay {i}") for i in range(100)]
            new_ids = [store.store(f"fresh memory to survive {i}") for i in range(100)]
            # Anchors establish the upper end of the frequency range.
            anchor1 = store.store("high frequency anchor alpha")
            anchor2 = store.store("high frequency anchor beta")

            old_ts = _old_ts()
            for mid in old_ids:
                store._conn.execute(
                    "UPDATE memories SET recency=?, frequency=0 WHERE id=?",
                    (old_ts, mid),
                )
            for mid in new_ids:
                store._conn.execute(
                    "UPDATE memories SET frequency=10 WHERE id=?", (mid,)
                )
            store._conn.execute(
                "UPDATE memories SET frequency=20 WHERE id=?", (anchor1,)
            )
            store._conn.execute(
                "UPDATE memories SET frequency=20 WHERE id=?", (anchor2,)
            )
            store._conn.commit()

            store.decay_cycle()

            for mid in old_ids:
                row = store._conn.execute(
                    "SELECT classification FROM memories WHERE id=?", (mid,)
                ).fetchone()
                self.assertEqual(
                    row[0], "forget",
                    f"Old memory {mid!r} expected 'forget', got {row[0]!r}",
                )

            for mid in new_ids:
                row = store._conn.execute(
                    "SELECT classification FROM memories WHERE id=?", (mid,)
                ).fetchone()
                self.assertNotEqual(
                    row[0], "forget",
                    f"Fresh memory {mid!r} must not be 'forget'",
                )


# ---------------------------------------------------------------------------
# TestMergeSimilarScale
# ---------------------------------------------------------------------------

class TestMergeSimilarScale(unittest.TestCase):
    """merge_similar() correctness and performance at scale."""

    def test_merge_similar_1000_unique_no_merges(self):
        """1000 unique memories → merge_similar(0.8) must return 0 and finish in < 30 s.

        Each memory uses only words specific to its index (e.g. ``item42``,
        ``node42``), so no two memories share any word and the pairwise Jaccard
        similarity is always 0.
        """
        with _make_store() as store:
            engine = ConsolidationEngine(store)
            for i in range(1000):
                store.store(
                    f"item{i} node{i} datum{i} entry{i} record{i} "
                    f"object{i} token{i} value{i}"
                )
            start = time.time()
            merges = engine.merge_similar(similarity_threshold=0.8)
            elapsed = time.time() - start

            self.assertEqual(merges, 0)
            self.assertLess(
                elapsed, 30.0,
                f"merge_similar took {elapsed:.1f}s, expected < 30s",
            )
            self.assertEqual(store.stats()["total"], 1000)

    def test_merge_similar_duplicates_merged(self):
        """10 identical memories → merge_similar(0.9) must return 9 merges; total = 1."""
        with _make_store() as store:
            engine = ConsolidationEngine(store)
            for _ in range(10):
                store.store("the cat sat on the mat")
            merges = engine.merge_similar(similarity_threshold=0.9)
            self.assertEqual(merges, 9)
            self.assertEqual(store.stats()["total"], 1)

    def test_merge_similar_partial_overlap(self):
        """Two similar cat sentences merge; the rocket sentence does not.

        Content is chosen so Jaccard similarity between the first two sentences
        is 0.75 (≥ threshold=0.6) and ≈ 0 between either cat sentence and the
        rocket sentence.

          A = "the cat sat on the mat by the window"
              words: {the, cat, sat, on, mat, by, window}  (7 unique)
          B = "the cat sat on the mat by the chair"
              words: {the, cat, sat, on, mat, by, chair}   (7 unique)
          A∩B = {the, cat, sat, on, mat, by} → 6
          A∪B = {the, cat, sat, on, mat, by, window, chair} → 8
          Jaccard = 6/8 = 0.75 ≥ 0.6  ✓
        """
        with _make_store() as store:
            engine = ConsolidationEngine(store)
            store.store("the cat sat on the mat by the window")
            store.store("the cat sat on the mat by the chair")
            store.store("completely different content about rockets")
            merges = engine.merge_similar(similarity_threshold=0.6)
            self.assertGreaterEqual(merges, 1, "First two sentences must merge")
            self.assertEqual(store.stats()["total"], 2)


# ---------------------------------------------------------------------------
# TestDecayGraduation
# ---------------------------------------------------------------------------

class TestDecayGraduation(unittest.TestCase):
    """Decay cycle graduation: core → forget and resurrection."""

    def test_decay_graduation_core_to_forget(self):
        """A memory that is 31 days old with freq=0 must decay to 'forget'.

        Two anchor memories at freq=50 give the frequency range (0, 50) so
        that the 31-day-old memory maps to c = -2 - 1j (escapes immediately).
        """
        with _make_store() as store:
            mid = store.store("ancient memory that should decay")
            # Anchor memories to establish a meaningful frequency range.
            store.store("anchor memory one")
            store.store("anchor memory two")

            old_ts = _old_ts()
            store._conn.execute(
                "UPDATE memories SET recency=?, frequency=0 WHERE id=?",
                (old_ts, mid),
            )
            store._conn.execute(
                "UPDATE memories SET frequency=50 "
                "WHERE id != ?", (mid,)
            )
            store._conn.commit()

            store.decay_cycle()

            row = store._conn.execute(
                "SELECT classification FROM memories WHERE id=?", (mid,)
            ).fetchone()
            self.assertEqual(row[0], "forget")

    def test_decay_idempotency(self):
        """Running decay_cycle() 10 times must produce a stable classification.

        Classifications must be identical after run 2 through run 10 — no
        flickering.
        """
        with _make_store() as store:
            old_ids = [store.store(f"old memory idempotency {i}") for i in range(5)]
            new_ids = [store.store(f"fresh memory idempotency {i}") for i in range(5)]

            old_ts = _old_ts()
            for mid in old_ids:
                store._conn.execute(
                    "UPDATE memories SET recency=?, frequency=0 WHERE id=?",
                    (old_ts, mid),
                )
            for mid in new_ids:
                store._conn.execute(
                    "UPDATE memories SET frequency=10 WHERE id=?", (mid,)
                )
            store._conn.commit()

            def _snapshot() -> dict:
                rows = store._conn.execute(
                    "SELECT id, classification FROM memories"
                ).fetchall()
                return {row["id"]: row["classification"] for row in rows}

            state_after_run = {}
            for run in range(1, 11):
                store.decay_cycle()
                state_after_run[run] = _snapshot()

            # From run 2 onward, classifications must be stable.
            for run in range(3, 11):
                self.assertEqual(
                    state_after_run[run], state_after_run[2],
                    f"Classification flickered between run 2 and run {run}",
                )

    def test_resurrection_after_retrieve(self):
        """A 'forget' memory must be resurrected to 'core' after a retrieve call.

        Anchor memories at freq=2 keep the frequency range (0, 2).  After
        retrieve(), the memory has freq=1 which normalises to imag=0
        (midpoint), giving c = 0.25 + 0j (inside the Mandelbrot set → 'core').
        A subsequent decay_cycle() must not demote it back to 'forget'.
        """
        with _make_store() as store:
            mid = store.store("memory to be resurrected")
            # Anchors at freq=2 so that after retrieve() (freq=1) the memory
            # maps to the midpoint of the frequency range → imag=0 → 'core'.
            anchor1 = store.store("resurrection anchor alpha")
            anchor2 = store.store("resurrection anchor beta")

            old_ts = _old_ts()
            store._conn.execute(
                "UPDATE memories SET recency=?, frequency=0 WHERE id=?",
                (old_ts, mid),
            )
            store._conn.execute(
                "UPDATE memories SET frequency=2 WHERE id=?", (anchor1,)
            )
            store._conn.execute(
                "UPDATE memories SET frequency=2 WHERE id=?", (anchor2,)
            )
            store._conn.commit()

            # Cycle 1: old memory must become 'forget'.
            store.decay_cycle()
            row = store._conn.execute(
                "SELECT classification FROM memories WHERE id=?", (mid,)
            ).fetchone()
            self.assertEqual(row[0], "forget", "Memory must be forget before resurrection")

            # Resurrect: retrieve() resets recency to MAX freshness and
            # recomputes classification using current global ranges.
            store.retrieve(mid)

            # Cycle 2: resurrected memory must NOT be 'forget'.
            store.decay_cycle()
            row = store._conn.execute(
                "SELECT classification FROM memories WHERE id=?", (mid,)
            ).fetchone()
            self.assertNotEqual(
                row[0], "forget",
                "Memory must not be 'forget' after resurrection via retrieve()",
            )


# ---------------------------------------------------------------------------
# TestFullPipeline
# ---------------------------------------------------------------------------

class TestFullPipeline(unittest.TestCase):
    """End-to-end store → decay → prune → store pipelines."""

    def test_store_decay_prune_cycle(self):
        """store(50) → age 20 → decay → prune → store(20) → total == 50.

        Anchor memories at freq=20 keep the frequency range (0, 20) so that
        fresh memories at freq=10 map to c = 0.25 + 0j (inside the Mandelbrot
        set) and survive the decay cycle.

        After prune, 30 fresh memories survive.  Storing 20 more brings the
        total back to 50.  At every step ``sum(by_classification.values()) ==
        total`` must hold.
        """
        with _make_store() as store:
            all_ids = [
                store.store(f"pipeline memory {i}") for i in range(50)
            ]
            old_ids = all_ids[:20]
            fresh_ids = all_ids[20:]   # 30 fresh

            # Anchors push fresh (freq=10) to imag=0 → c=0.25+0j → 'core'.
            anchor1 = store.store("pipeline anchor alpha")
            anchor2 = store.store("pipeline anchor beta")

            old_ts = _old_ts()
            for mid in old_ids:
                store._conn.execute(
                    "UPDATE memories SET recency=?, frequency=0 WHERE id=?",
                    (old_ts, mid),
                )
            for mid in fresh_ids:
                store._conn.execute(
                    "UPDATE memories SET frequency=10 WHERE id=?", (mid,)
                )
            store._conn.execute(
                "UPDATE memories SET frequency=20 WHERE id=?", (anchor1,)
            )
            store._conn.execute(
                "UPDATE memories SET frequency=20 WHERE id=?", (anchor2,)
            )
            store._conn.commit()

            store.decay_cycle()
            store.prune("forget")

            # 30 fresh memories survive (20 old + 2 anchors are pruned).
            self.assertEqual(store.stats()["total"], 30)

            new_ids = [
                store.store(f"new pipeline memory {i}") for i in range(20)
            ]

            # Total must be 30 (survivors) + 20 (new) = 50.
            self.assertEqual(store.stats()["total"], 50)

            # Verify all 20 newly stored IDs are present and retrievable.
            for mid in new_ids:
                node = store.retrieve(mid)
                self.assertIsNotNone(node, f"New memory {mid!r} must be retrievable")

            s = store.stats()
            self.assertEqual(sum(s["by_classification"].values()), s["total"])

    def test_full_pipeline_no_data_loss(self):
        """Content stored must be returned intact after decay + prune cycles.

        70 fresh memories survive; 30 aged ones are pruned.  Every surviving
        ID must return exactly the original content; every pruned ID must
        return None from retrieve().
        """
        with _make_store() as store:
            ids_and_content: dict[str, str] = {}
            for i in range(100):
                content = f"no data loss content memory index {i}"
                mid = store.store(content)
                ids_and_content[mid] = content

            # Separate aged vs fresh IDs.
            all_ids = list(ids_and_content)
            aged_ids = all_ids[:30]
            surviving_ids = all_ids[30:]

            # Anchors keep surviving (freq=10) in the midpoint of range (0,20).
            anchor1 = store.store("no data loss anchor alpha")
            anchor2 = store.store("no data loss anchor beta")

            old_ts = _old_ts()
            for mid in aged_ids:
                store._conn.execute(
                    "UPDATE memories SET recency=?, frequency=0 WHERE id=?",
                    (old_ts, mid),
                )
            for mid in surviving_ids:
                store._conn.execute(
                    "UPDATE memories SET frequency=10 WHERE id=?", (mid,)
                )
            store._conn.execute(
                "UPDATE memories SET frequency=20 WHERE id=?", (anchor1,)
            )
            store._conn.execute(
                "UPDATE memories SET frequency=20 WHERE id=?", (anchor2,)
            )
            store._conn.commit()

            store.decay_cycle()
            store.prune("forget")

            # Surviving memories must return their original content.
            for mid in surviving_ids:
                node = store.retrieve(mid)
                self.assertIsNotNone(node, f"Surviving memory {mid!r} must not be None")
                self.assertEqual(
                    node.content, ids_and_content[mid],
                    f"Content mismatch for {mid!r}",
                )

            # Pruned memories must return None.
            for mid in aged_ids:
                node = store.retrieve(mid)
                self.assertIsNone(node, f"Pruned memory {mid!r} must return None")


# ---------------------------------------------------------------------------
# TestConcurrency
# ---------------------------------------------------------------------------

class TestConcurrency(unittest.TestCase):
    """SQLite thread-safety: background maintenance + main-thread stores."""

    def test_concurrent_store_and_maintenance(self):
        """MemoryMaintenance running at 50 ms while main thread stores 100 memories.

        Uses a real MemoryStore (SQLite :memory:) with check_same_thread=False.
        Asserts:
          - no exception propagated to the main thread
          - at least one memory is present after all stores complete
          - stats()['by_classification'] sums to total
        """
        store = _ThreadSafeStore()
        error_flag = threading.Event()
        cycles_run = []

        def on_cycle(result: dict) -> None:
            cycles_run.append(result)

        maint = MemoryMaintenance(store, interval=0.05, on_cycle=on_cycle)
        maint.start()

        try:
            for i in range(100):
                store.store(f"concurrent memory {i}")
        except Exception:
            error_flag.set()
            raise
        finally:
            maint.stop()
            # Allow any in-flight maintenance tick to complete.
            time.sleep(0.15)

        self.assertFalse(
            error_flag.is_set(),
            "An exception was raised in the main thread during concurrent stores",
        )
        self.assertGreater(
            store.stats()["total"], 0,
            "At least one memory must remain after concurrent stores + maintenance",
        )
        s = store.stats()
        self.assertEqual(
            sum(s["by_classification"].values()), s["total"],
            "by_classification must sum to total after concurrent operation",
        )
        store.close()


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):
    """Edge cases: empty store, single memory, zero-frequency freshness."""

    def test_all_operations_on_empty_store(self):
        """Every public method on a fresh empty store must not raise."""
        with _make_store() as store:
            self.assertEqual(store.search("anything"), [])
            self.assertEqual(store.get_all(), [])
            self.assertEqual(store.get_by_classification("core"), [])
            self.assertEqual(store.decay_cycle(), {})
            self.assertEqual(store.prune("forget"), 0)
            s = store.stats()
            self.assertEqual(s["total"], 0)

    def test_single_memory_full_lifecycle(self):
        """A single memory survives search/get_all/decay/stats without error."""
        with _make_store() as store:
            mid = store.store("single lifecycle memory")

            results = store.search("single lifecycle")
            self.assertEqual(len(results), 1)

            all_nodes = store.get_all()
            self.assertEqual(len(all_nodes), 1)

            core_nodes = store.get_by_classification("core")
            self.assertGreaterEqual(len(core_nodes), 1)

            store.decay_cycle()

            s = store.stats()
            self.assertEqual(s["total"], 1)

            # Memory must survive until explicitly pruned.
            node = store.retrieve(mid)
            self.assertIsNotNone(node)

    def test_frequency_zero_fresh_is_core(self):
        """A freshly stored memory with frequency=0 must remain 'core' after decay_cycle().

        Freshness (recency = now) maps to real = 0.25 on the complex plane.
        With a degenerate frequency range (all memories have freq=0), imag = 0
        (midpoint), giving c = 0.25 + 0j which is inside the Mandelbrot set.
        """
        with _make_store() as store:
            mid = store.store("fresh zero frequency memory")
            # Confirm frequency is 0 in the DB.
            row = store._conn.execute(
                "SELECT frequency FROM memories WHERE id=?", (mid,)
            ).fetchone()
            self.assertEqual(row[0], 0)

            store.decay_cycle()

            row = store._conn.execute(
                "SELECT classification FROM memories WHERE id=?", (mid,)
            ).fetchone()
            self.assertNotEqual(
                row[0], "forget",
                "A freshly stored memory with freq=0 must not decay to 'forget'",
            )


if __name__ == "__main__":
    unittest.main()
