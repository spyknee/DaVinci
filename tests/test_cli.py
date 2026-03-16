"""
Integration tests for the DaVinci CLI — ``python -m davinci``.

Tests use subprocess to invoke the CLI via a temporary SQLite database file so
they are fully isolated.

Run with:
    python -m pytest tests/test_cli.py -v
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest


def _run(args: list[str], db_path: str) -> subprocess.CompletedProcess:
    """Run ``python -m davinci --db <db_path> <args>`` and return the result."""
    cmd = [sys.executable, "-m", "davinci", "--db", db_path] + args
    return subprocess.run(cmd, capture_output=True, text=True)


class TestRememberCommand(unittest.TestCase):
    """``remember`` command stores memories."""

    def test_remember_succeeds(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            result = _run(["remember", "fractal memory test"], db)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("Stored memory", result.stdout)
        finally:
            os.unlink(db)

    def test_remember_outputs_uuid(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            result = _run(["remember", "uuid output test"], db)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            # UUID is 36 chars; check it appears in output
            parts = result.stdout.strip().split()
            uuid_part = parts[-1]
            self.assertEqual(len(uuid_part), 36)
        finally:
            os.unlink(db)

    def test_remember_with_zoom_levels(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            result = _run(
                ["remember", "zoom test", "--zoom1", "summary", "--zoom2", "detail", "--zoom3", "full"],
                db,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
        finally:
            os.unlink(db)


class TestRecallCommand(unittest.TestCase):
    """``recall`` command retrieves memories by UUID."""

    def _store_and_get_id(self, db: str, content: str = "recall test content") -> str:
        result = _run(["remember", content], db)
        # Last token on the output line is the UUID
        return result.stdout.strip().split()[-1]

    def test_recall_known_id(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            mid = self._store_and_get_id(db)
            result = _run(["recall", mid], db)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("recall test content", result.stdout)
        finally:
            os.unlink(db)

    def test_recall_unknown_id_exits_nonzero(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            result = _run(["recall", "00000000-0000-0000-0000-000000000000"], db)
            self.assertNotEqual(result.returncode, 0)
        finally:
            os.unlink(db)


class TestSearchCommand(unittest.TestCase):
    """``search`` command finds stored memories."""

    def test_search_finds_stored_memory(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            _run(["remember", "Mandelbrot set exploration"], db)
            result = _run(["search", "Mandelbrot"], db)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("Mandelbrot", result.stdout)
        finally:
            os.unlink(db)

    def test_search_no_results(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            result = _run(["search", "xyzzy_no_match_ever"], db)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("0 found", result.stdout)
        finally:
            os.unlink(db)

    def test_search_limit_flag(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            for i in range(5):
                _run(["remember", f"limit test memory {i}"], db)
            result = _run(["search", "limit test", "--limit", "2"], db)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
        finally:
            os.unlink(db)


class TestStatsCommand(unittest.TestCase):
    """``stats`` command prints memory statistics."""

    def test_stats_returns_output(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            result = _run(["stats"], db)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("Total memories", result.stdout)
        finally:
            os.unlink(db)

    def test_stats_shows_classifications(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            _run(["remember", "stats content"], db)
            result = _run(["stats"], db)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("core", result.stdout)
            self.assertIn("forget", result.stdout)
        finally:
            os.unlink(db)


class TestForgetCommand(unittest.TestCase):
    """``forget`` command prunes memories."""

    def test_forget_runs_without_error(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            result = _run(["forget"], db)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("Deleted", result.stdout)
        finally:
            os.unlink(db)

    def test_forget_custom_classification(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            result = _run(["forget", "--classification", "decay"], db)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
        finally:
            os.unlink(db)


class TestDecayCommand(unittest.TestCase):
    """``decay`` command runs the decay cycle."""

    def test_decay_runs_without_error(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            result = _run(["decay"], db)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("Decay cycle complete", result.stdout)
        finally:
            os.unlink(db)

    def test_decay_with_memories(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            _run(["remember", "decay test memory"], db)
            result = _run(["decay"], db)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
        finally:
            os.unlink(db)


class TestConsolidateCommand(unittest.TestCase):
    """``consolidate`` command runs consolidation."""

    def test_consolidate_runs_without_error(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            result = _run(["consolidate"], db)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("Consolidation complete", result.stdout)
        finally:
            os.unlink(db)

    def test_consolidate_strategy_flag(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            result = _run(["consolidate", "--strategy", "frequency"], db)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
        finally:
            os.unlink(db)


class TestMergeCommand(unittest.TestCase):
    """``merge`` command merges similar memories."""

    def test_merge_runs_without_error(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            result = _run(["merge"], db)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("Merge complete", result.stdout)
        finally:
            os.unlink(db)

    def test_merge_threshold_flag(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            result = _run(["merge", "--threshold", "0.9"], db)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
        finally:
            os.unlink(db)


class TestMemoriesCommand(unittest.TestCase):
    """``memories`` command lists stored memories."""

    def test_memories_runs_without_error(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            result = _run(["memories"], db)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
        finally:
            os.unlink(db)

    def test_memories_shows_stored_content(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            _run(["remember", "listed content preview"], db)
            result = _run(["memories"], db)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("listed content preview", result.stdout)
        finally:
            os.unlink(db)

    def test_memories_classification_filter(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            _run(["remember", "filter me"], db)
            result = _run(["memories", "--classification", "forget"], db)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
        finally:
            os.unlink(db)


class TestDbFlag(unittest.TestCase):
    """``--db`` flag uses the specified database file."""

    def test_db_flag_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "custom.db")
            result = _run(["stats"], db_path)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(os.path.exists(db_path))

    def test_db_flag_persists_across_calls(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            _run(["remember", "persistent memory"], db)
            result = _run(["search", "persistent"], db)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("persistent", result.stdout)
        finally:
            os.unlink(db)


class TestUnknownCommand(unittest.TestCase):
    """Unknown or missing commands show help."""

    def test_no_command_shows_help(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            result = _run([], db)
            # Should exit 0 and print help (argparse behavior)
            combined = result.stdout + result.stderr
            self.assertIn("davinci", combined.lower())
        finally:
            os.unlink(db)


if __name__ == "__main__":
    unittest.main()
