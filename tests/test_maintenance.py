"""
Unit tests for davinci.memory.maintenance — MemoryMaintenance.

All tests are synchronous; no real timers fire.

Run with:
    python -m pytest tests/test_maintenance.py -v
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, call, patch

from davinci.memory.maintenance import MemoryMaintenance


def _make_store_mock() -> MagicMock:
    """Return a mock MemoryStore with sensible default return values."""
    store = MagicMock()
    store.decay_cycle.return_value = {}
    store.prune.return_value = 0
    return store


class TestRunOnce(unittest.TestCase):
    """run_once() calls the correct operations and returns a stats dict."""

    @patch("davinci.memory.maintenance.ConsolidationEngine")
    def test_run_once_calls_decay_cycle(self, MockEngine):
        store = _make_store_mock()
        MockEngine.return_value.merge_similar.return_value = 0
        m = MemoryMaintenance(store=store)
        m.run_once()
        store.decay_cycle.assert_called_once()

    @patch("davinci.memory.maintenance.ConsolidationEngine")
    def test_run_once_calls_merge_similar(self, MockEngine):
        store = _make_store_mock()
        mock_engine = MockEngine.return_value
        mock_engine.merge_similar.return_value = 0
        m = MemoryMaintenance(store=store, similarity_threshold=0.75)
        m.run_once()
        mock_engine.merge_similar.assert_called_once_with(0.75)

    @patch("davinci.memory.maintenance.ConsolidationEngine")
    def test_run_once_calls_prune(self, MockEngine):
        store = _make_store_mock()
        MockEngine.return_value.merge_similar.return_value = 0
        m = MemoryMaintenance(store=store)
        m.run_once()
        store.prune.assert_called_once_with("forget")

    @patch("davinci.memory.maintenance.ConsolidationEngine")
    def test_run_once_returns_stats_dict(self, MockEngine):
        store = _make_store_mock()
        store.decay_cycle.return_value = {"decay": ["id1"]}
        store.prune.return_value = 3
        MockEngine.return_value.merge_similar.return_value = 2
        m = MemoryMaintenance(store=store)
        stats = m.run_once()
        self.assertIn("decayed", stats)
        self.assertIn("merged", stats)
        self.assertIn("pruned", stats)
        self.assertEqual(stats["decayed"], {"decay": ["id1"]})
        self.assertEqual(stats["merged"], 2)
        self.assertEqual(stats["pruned"], 3)


class TestStartStop(unittest.TestCase):
    """start() and stop() manage the timer correctly."""

    @patch("davinci.memory.maintenance.ConsolidationEngine")
    @patch("davinci.memory.maintenance.threading.Timer")
    def test_start_schedules_timer(self, MockTimer, MockEngine):
        store = _make_store_mock()
        m = MemoryMaintenance(store=store, interval=60)
        m.start()
        MockTimer.assert_called_once_with(60, m._tick)
        MockTimer.return_value.start.assert_called_once()
        m.stop()

    @patch("davinci.memory.maintenance.ConsolidationEngine")
    @patch("davinci.memory.maintenance.threading.Timer")
    def test_stop_cancels_timer(self, MockTimer, MockEngine):
        store = _make_store_mock()
        m = MemoryMaintenance(store=store)
        m.start()
        m.stop()
        MockTimer.return_value.cancel.assert_called_once()
        self.assertFalse(m._running)


class TestTickBehaviour(unittest.TestCase):
    """_tick() callbacks and rescheduling."""

    @patch("davinci.memory.maintenance.ConsolidationEngine")
    @patch("davinci.memory.maintenance.threading.Timer")
    def test_on_cycle_callback_called(self, MockTimer, MockEngine):
        store = _make_store_mock()
        store.decay_cycle.return_value = {}
        store.prune.return_value = 0
        MockEngine.return_value.merge_similar.return_value = 1
        callback = MagicMock()
        m = MemoryMaintenance(store=store, on_cycle=callback)
        m._running = True  # simulate started state without real timer
        m._tick()
        callback.assert_called_once()
        stats = callback.call_args[0][0]
        self.assertIn("decayed", stats)
        self.assertIn("merged", stats)
        self.assertIn("pruned", stats)

    @patch("davinci.memory.maintenance.ConsolidationEngine")
    @patch("davinci.memory.maintenance.threading.Timer")
    def test_tick_reschedules_after_exception(self, MockTimer, MockEngine):
        store = _make_store_mock()
        m = MemoryMaintenance(store=store)
        m._running = True  # simulate started state without real timer
        with patch.object(m, "run_once", side_effect=RuntimeError("boom")):
            with patch.object(m, "_schedule_next") as mock_schedule:
                m._tick()
                mock_schedule.assert_called_once()


class TestContextManager(unittest.TestCase):
    """Context manager starts and stops the loop."""

    @patch("davinci.memory.maintenance.ConsolidationEngine")
    @patch("davinci.memory.maintenance.threading.Timer")
    def test_context_manager_starts_and_stops(self, MockTimer, MockEngine):
        store = _make_store_mock()
        m = MemoryMaintenance(store=store)
        with m:
            self.assertTrue(m._running)
        self.assertFalse(m._running)


if __name__ == "__main__":
    unittest.main()
