"""
Unit tests for davinci.voice.session — VoiceSession (Layer 4).

Run with:
    python -m pytest tests/test_voice_session.py -v
"""

from __future__ import annotations

import unittest

from davinci.voice.interface import VoiceInterface
from davinci.voice.session import VoiceSession
from davinci.voice.stt import StubSTT
from davinci.voice.tts import StubTTS


def _make_session(responses=None, spoken=None):
    """Helper: create VoiceInterface + VoiceSession with stub backends."""
    vi = VoiceInterface(db_path=":memory:")
    vi._stt = StubSTT(responses=responses or [])
    vi._tts = StubTTS(output=spoken if spoken is not None else [])
    session = VoiceSession(vi)
    return vi, session


class TestSessionHistoryTracking(unittest.TestCase):
    """Session tracks conversation history."""

    def test_history_empty_before_start(self):
        vi, session = _make_session(responses=["quit"])
        self.assertEqual(session.history(), [])
        vi.close()

    def test_history_populated_after_start(self):
        vi, session = _make_session(responses=["quit"])
        session.start()
        self.assertGreater(len(session.history()), 0)
        vi.close()

    def test_history_contains_user_and_davinci_roles(self):
        vi, session = _make_session(responses=["quit"])
        session.start()
        roles = {entry["role"] for entry in session.history()}
        self.assertIn("user", roles)
        self.assertIn("davinci", roles)
        vi.close()

    def test_history_records_user_text(self):
        vi, session = _make_session(responses=["quit"])
        session.start()
        user_texts = [e["text"] for e in session.history() if e["role"] == "user"]
        self.assertIn("quit", user_texts)
        vi.close()

    def test_history_records_davinci_text(self):
        vi, session = _make_session(responses=["quit"])
        session.start()
        davinci_texts = [e["text"] for e in session.history() if e["role"] == "davinci"]
        self.assertTrue(len(davinci_texts) >= 1)
        vi.close()


class TestSessionHistoryTimestamps(unittest.TestCase):
    """Session history has correct roles and timestamps."""

    def test_each_entry_has_timestamp(self):
        vi, session = _make_session(responses=["quit"])
        session.start()
        for entry in session.history():
            self.assertIn("timestamp", entry)
            self.assertIsInstance(entry["timestamp"], float)
            self.assertGreater(entry["timestamp"], 0)
        vi.close()

    def test_each_entry_has_role(self):
        vi, session = _make_session(responses=["quit"])
        session.start()
        for entry in session.history():
            self.assertIn("role", entry)
            self.assertIn(entry["role"], ("user", "davinci"))
        vi.close()

    def test_each_entry_has_text(self):
        vi, session = _make_session(responses=["quit"])
        session.start()
        for entry in session.history():
            self.assertIn("text", entry)
            self.assertIsInstance(entry["text"], str)
        vi.close()

    def test_timestamps_are_monotonically_nondecreasing(self):
        vi, session = _make_session(responses=["remember something", "quit"])
        session.start()
        history = session.history()
        if len(history) >= 2:
            for i in range(1, len(history)):
                self.assertGreaterEqual(
                    history[i]["timestamp"],
                    history[i - 1]["timestamp"],
                )
        vi.close()


class TestLastResponse(unittest.TestCase):
    """last_response returns most recent DaVinci response."""

    def test_last_response_none_before_start(self):
        vi, session = _make_session()
        self.assertIsNone(session.last_response())
        vi.close()

    def test_last_response_after_start(self):
        vi, session = _make_session(responses=["quit"])
        session.start()
        last = session.last_response()
        self.assertIsNotNone(last)
        self.assertIsInstance(last, str)
        vi.close()

    def test_last_response_is_most_recent_davinci_entry(self):
        vi, session = _make_session(responses=["remember hello session", "quit"])
        session.start()
        last = session.last_response()
        # History should have at least one davinci entry; last_response must
        # match the last davinci entry in history.
        davinci_entries = [e["text"] for e in session.history() if e["role"] == "davinci"]
        self.assertEqual(last, davinci_entries[-1])
        vi.close()


class TestClearHistory(unittest.TestCase):
    """clear_history empties the list."""

    def test_clear_history_empties_list(self):
        vi, session = _make_session(responses=["quit"])
        session.start()
        self.assertGreater(len(session.history()), 0)
        session.clear_history()
        self.assertEqual(session.history(), [])
        vi.close()

    def test_clear_history_on_empty_is_safe(self):
        vi, session = _make_session()
        session.clear_history()  # Should not raise
        self.assertEqual(session.history(), [])
        vi.close()

    def test_history_repopulates_after_clear(self):
        vi, session = _make_session(responses=["quit", "quit"])
        session.start()
        session.clear_history()
        # Re-run the loop
        vi._stt = StubSTT(responses=["quit"])
        session.start()
        self.assertGreater(len(session.history()), 0)
        vi.close()


if __name__ == "__main__":
    unittest.main()
