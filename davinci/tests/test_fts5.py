"""
Tests for FTS5 full-text search in davinci.memory.store — MemoryStore
"""

from __future__ import annotations

import unittest

from davinci.memory.store import MemoryStore


class TestFTS5TableCreation(unittest.TestCase):
    def test_fts5_table_exists(self):
        with MemoryStore(":memory:") as ms:
            row = ms._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'"
            ).fetchone()
            self.assertIsNotNone(row)

    def test_fts5_triggers_exist(self):
        with MemoryStore(":memory:") as ms:
            triggers = [
                row[0]
                for row in ms._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger'"
                ).fetchall()
            ]
            self.assertIn("memories_ai", triggers)
            self.assertIn("memories_ad", triggers)


class TestFTS5AutoPopulate(unittest.TestCase):
    def test_store_auto_populates_fts(self):
        with MemoryStore(":memory:") as ms:
            mid = ms.store("The Mandelbrot set is a beautiful fractal.")
            row = ms._conn.execute(
                "SELECT id FROM memories_fts WHERE id = ?", (mid,)
            ).fetchone()
            self.assertIsNotNone(row)

    def test_multiple_stores_populate_fts(self):
        with MemoryStore(":memory:") as ms:
            ids = [ms.store(f"Memory content number {i}") for i in range(5)]
            for mid in ids:
                row = ms._conn.execute(
                    "SELECT id FROM memories_fts WHERE id = ?", (mid,)
                ).fetchone()
                self.assertIsNotNone(row, f"FTS5 row missing for {mid}")


class TestSearchFTS(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(":memory:")
        self.store.store("Python is a high-level programming language")
        self.store.store("JavaScript runs in the browser")
        self.store.store("Rust is a systems programming language focused on safety")

    def tearDown(self):
        self.store.close()

    def test_search_fts_finds_keyword(self):
        results = self.store.search_fts("Python")
        self.assertGreater(len(results), 0)
        self.assertIn("Python", results[0].content)

    def test_search_fts_finds_in_any_field(self):
        results = self.store.search_fts("systems")
        self.assertGreater(len(results), 0)
        self.assertIn("systems", results[0].content)

    def test_search_fts_respects_limit(self):
        # Store many items
        for i in range(20):
            self.store.store(f"programming item number {i}")
        results = self.store.search_fts("programming", limit=5)
        self.assertLessEqual(len(results), 5)

    def test_search_fts_returns_memory_nodes(self):
        from davinci.core.fractal_engine import MemoryNode
        results = self.store.search_fts("Python")
        for r in results:
            self.assertIsInstance(r, MemoryNode)

    def test_search_fts_ordered_by_classification(self):
        results = self.store.search_fts("programming")
        if len(results) > 1:
            order = {"core": 0, "boundary": 1, "decay": 2, "forget": 3}
            priorities = [order.get(n.classification, 99) for n in results]
            self.assertEqual(priorities, sorted(priorities))

    def test_search_fts_no_match_returns_empty(self):
        results = self.store.search_fts("xyznotfoundabcdef")
        self.assertEqual(results, [])


class TestSearchFTSFallback(unittest.TestCase):
    def test_invalid_fts_query_falls_back_to_like(self):
        with MemoryStore(":memory:") as ms:
            ms.store("hello world content here")
            # An empty or syntactically invalid FTS5 query causes an error —
            # the method should fall back to LIKE search
            # We trigger the fallback by passing a query with unclosed quote
            try:
                results = ms.search_fts('"unclosed')
                # Either returns results via fallback or empty list — no exception
                self.assertIsInstance(results, list)
            except Exception as exc:
                self.fail(f"search_fts raised unexpectedly: {exc}")


class TestFTS5Prune(unittest.TestCase):
    def test_prune_removes_from_fts(self):
        with MemoryStore(":memory:") as ms:
            # A single freshly-stored item is always classified "forget"
            pruned_mid = ms.store("item to prune test")
            row_before = ms._conn.execute(
                "SELECT id FROM memories_fts WHERE id = ?", (pruned_mid,)
            ).fetchone()
            self.assertIsNotNone(row_before)
            ms.prune("forget")
            row = ms._conn.execute(
                "SELECT id FROM memories_fts WHERE id = ?", (pruned_mid,)
            ).fetchone()
            self.assertIsNone(row)

    def test_prune_clears_fts_table(self):
        with MemoryStore(":memory:") as ms:
            # Store a single item (classified "forget"), prune it, FTS5 should be empty
            ms.store("single item to prune")
            ms.prune("forget")
            count = ms._conn.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
            self.assertEqual(count, 0)


class TestFTS5Backfill(unittest.TestCase):
    def test_backfill_populates_existing_rows(self):
        """Verify _backfill_fts inserts rows that were not yet in FTS5."""
        with MemoryStore(":memory:") as ms:
            # Manually insert a row into memories without going through FTS5 trigger
            import uuid, time, json
            mid = str(uuid.uuid4())
            now = time.time()
            ms._conn.execute(
                """
                INSERT INTO memories
                    (id, content, frequency, recency, created_at,
                     classification, c_real, c_imag, iteration_count,
                     zoom_level_1, zoom_level_2, zoom_level_3, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (mid, "backfill test content", 0, now, now, "forget",
                 0.0, 0.0, 0, "content", "content", "content", "{}"),
            )
            # Manually delete from FTS5 to simulate pre-FTS5 database
            ms._conn.execute("DELETE FROM memories_fts WHERE id = ?", (mid,))
            ms._conn.commit()

            # Now run backfill
            ms._backfill_fts()

            row = ms._conn.execute(
                "SELECT id FROM memories_fts WHERE id = ?", (mid,)
            ).fetchone()
            self.assertIsNotNone(row)


if __name__ == "__main__":
    unittest.main()
