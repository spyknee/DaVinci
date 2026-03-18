"""
Unit tests for davinci.voice.interface — VoiceInterface (Layer 4).

Run with:
    python -m pytest tests/test_voice_interface.py -v
"""

from __future__ import annotations

import io
import sys
import unittest

from davinci.interface.base import BaseInterface
from davinci.voice.interface import VoiceInterface
from davinci.voice.stt import StubSTT
from davinci.voice.tts import StubTTS


def _make_vi(responses=None, spoken=None, db_path=":memory:"):
    """Helper: create a VoiceInterface with stub backends."""
    from davinci.voice.stt import STTRegistry
    from davinci.voice.tts import TTSRegistry

    # Register temporary backends with the specific instances we want.
    _stt_instance = StubSTT(responses=responses or [])
    _tts_instance = StubTTS(output=spoken if spoken is not None else [])

    # Bypass the registry by directly assigning backend instances to the
    # VoiceInterface so tests are decoupled from registry state.
    vi = VoiceInterface(db_path=db_path)
    vi._stt = _stt_instance
    vi._tts = _tts_instance
    return vi


class TestVoiceInterfaceCreation(unittest.TestCase):
    """VoiceInterface creates successfully with stub backends."""

    def test_creates_without_error(self):
        with VoiceInterface(db_path=":memory:") as vi:
            self.assertIsNotNone(vi)

    def test_is_base_interface_subclass(self):
        self.assertTrue(issubclass(VoiceInterface, BaseInterface))

    def test_instance_is_base_interface(self):
        with VoiceInterface(db_path=":memory:") as vi:
            self.assertIsInstance(vi, BaseInterface)


class TestVoiceInterfaceMemoryOps(unittest.TestCase):
    """remember/recall/search work through VoiceInterface."""

    def test_remember_returns_uuid(self):
        with _make_vi() as vi:
            mid = vi.remember("fractal memory")
            self.assertIsInstance(mid, str)
            self.assertEqual(len(mid), 36)

    def test_recall_returns_node(self):
        with _make_vi() as vi:
            mid = vi.remember("hello voice")
            node = vi.recall(mid)
            self.assertIsNotNone(node)
            self.assertEqual(node.content, "hello voice")

    def test_search_returns_results(self):
        with _make_vi() as vi:
            vi.remember("orange tree")
            results = vi.search("orange")
            self.assertTrue(len(results) >= 1)
            self.assertEqual(results[0].content, "orange tree")


class TestVoiceInterfaceListen(unittest.TestCase):
    """listen() delegates to STT backend."""

    def test_listen_returns_stt_response(self):
        spoken = []
        vi = _make_vi(responses=["test utterance"], spoken=spoken)
        result = vi.listen()
        vi.close()
        self.assertEqual(result, "test utterance")

    def test_listen_cycles_responses(self):
        vi = _make_vi(responses=["one", "two"])
        self.assertEqual(vi.listen(), "one")
        self.assertEqual(vi.listen(), "two")
        self.assertEqual(vi.listen(), "one")
        vi.close()


class TestVoiceInterfaceSpeak(unittest.TestCase):
    """speak() delegates to TTS backend."""

    def test_speak_uses_tts(self):
        spoken = []
        vi = _make_vi(responses=[], spoken=spoken)
        vi.speak("hello from voice")
        vi.close()
        self.assertIn("hello from voice", spoken)

    def test_speak_prints_to_stdout(self):
        vi = _make_vi(responses=[])
        captured = io.StringIO()
        sys.stdout = captured
        try:
            vi.speak("stdout check")
        finally:
            sys.stdout = sys.__stdout__
            vi.close()
        self.assertIn("stdout check", captured.getvalue())


class TestParseIntent(unittest.TestCase):
    """parse_intent correctly identifies commands."""

    def setUp(self):
        self.vi = VoiceInterface(db_path=":memory:")

    def tearDown(self):
        self.vi.close()

    def test_remember_command(self):
        cmd, arg = self.vi.parse_intent("remember the quick brown fox")
        self.assertEqual(cmd, "remember")
        self.assertEqual(arg, "the quick brown fox")

    def test_search_for_command(self):
        cmd, arg = self.vi.parse_intent("search for cats")
        self.assertEqual(cmd, "search")
        self.assertEqual(arg, "cats")

    def test_search_command(self):
        cmd, arg = self.vi.parse_intent("search dogs")
        self.assertEqual(cmd, "search")
        self.assertEqual(arg, "dogs")

    def test_recall_command(self):
        cmd, arg = self.vi.parse_intent("recall abc-123")
        self.assertEqual(cmd, "search")
        self.assertEqual(arg, "abc-123")

    def test_forget_command(self):
        cmd, arg = self.vi.parse_intent("forget")
        self.assertEqual(cmd, "forget")
        self.assertEqual(arg, "")

    def test_stats_command(self):
        cmd, arg = self.vi.parse_intent("stats")
        self.assertEqual(cmd, "stats")
        self.assertEqual(arg, "")

    def test_decay_command(self):
        cmd, arg = self.vi.parse_intent("decay")
        self.assertEqual(cmd, "decay")
        self.assertEqual(arg, "")

    def test_quit_command(self):
        cmd, arg = self.vi.parse_intent("quit")
        self.assertEqual(cmd, "quit")
        self.assertEqual(arg, "")

    def test_exit_command(self):
        cmd, arg = self.vi.parse_intent("exit")
        self.assertEqual(cmd, "quit")
        self.assertEqual(arg, "")

    def test_stop_command(self):
        cmd, arg = self.vi.parse_intent("stop")
        self.assertEqual(cmd, "quit")
        self.assertEqual(arg, "")

    def test_random_text_defaults_to_remember(self):
        text = "this is just some random text"
        cmd, arg = self.vi.parse_intent(text)
        self.assertEqual(cmd, "remember")
        self.assertEqual(arg, text)

    def test_whitespace_stripped(self):
        cmd, arg = self.vi.parse_intent("  quit  ")
        self.assertEqual(cmd, "quit")


class TestConversationLoop(unittest.TestCase):
    """conversation_loop runs with pre-canned STT responses and exits on 'quit'."""

    def test_loop_exits_on_quit(self):
        spoken = []
        vi = _make_vi(responses=["quit"], spoken=spoken)
        vi.conversation_loop()
        vi.close()
        # Should have spoken the prompt and the goodbye
        self.assertTrue(len(spoken) >= 1)

    def test_loop_remembers_content(self):
        spoken = []
        vi = _make_vi(responses=["remember hello voice world", "quit"], spoken=spoken)
        vi.conversation_loop()
        vi.close()
        # Confirmation should mention the memory id
        confirmations = [s for s in spoken if "Remembered" in s]
        self.assertTrue(len(confirmations) >= 1)

    def test_loop_searches(self):
        spoken = []
        vi = _make_vi(responses=["search fractal", "quit"], spoken=spoken)
        vi.remember("fractal engine")
        vi.conversation_loop()
        vi.close()
        results_msgs = [s for s in spoken if "Found" in s or "No memories" in s]
        self.assertTrue(len(results_msgs) >= 1)

    def test_loop_stats(self):
        spoken = []
        vi = _make_vi(responses=["stats", "quit"], spoken=spoken)
        vi.conversation_loop()
        vi.close()
        stats_msgs = [s for s in spoken if "memor" in s.lower()]
        self.assertTrue(len(stats_msgs) >= 1)

    def test_loop_decay(self):
        spoken = []
        vi = _make_vi(responses=["decay", "quit"], spoken=spoken)
        vi.conversation_loop()
        vi.close()
        decay_msgs = [s for s in spoken if "Decay" in s or "reclassified" in s]
        self.assertTrue(len(decay_msgs) >= 1)

    def test_loop_forget(self):
        spoken = []
        vi = _make_vi(responses=["forget", "quit"], spoken=spoken)
        vi.conversation_loop()
        vi.close()
        forget_msgs = [s for s in spoken if "Forgot" in s]
        self.assertTrue(len(forget_msgs) >= 1)


class TestVoiceInterfaceContextManager(unittest.TestCase):
    """Context manager works."""

    def test_context_manager_closes_cleanly(self):
        with VoiceInterface(db_path=":memory:") as vi:
            mid = vi.remember("test context manager")
            self.assertIsInstance(mid, str)
        # After __exit__ the DB should be closed — no error expected.

    def test_context_manager_enter_returns_self(self):
        vi = VoiceInterface(db_path=":memory:")
        result = vi.__enter__()
        self.assertIs(result, vi)
        vi.close()


class TestVoiceInterfaceAdvanced(unittest.TestCase):
    """Stats/decay/consolidate work through voice interface."""

    def test_stats_returns_dict(self):
        with _make_vi() as vi:
            s = vi.stats()
            self.assertIsInstance(s, dict)
            self.assertIn("total", s)

    def test_decay_returns_dict(self):
        with _make_vi() as vi:
            result = vi.decay()
            self.assertIsInstance(result, dict)

    def test_consolidate_returns_int(self):
        with _make_vi() as vi:
            count = vi.consolidate()
            self.assertIsInstance(count, int)

    def test_memories_returns_list(self):
        with _make_vi() as vi:
            vi.remember("memory one")
            vi.remember("memory two")
            all_mems = vi.memories()
            self.assertEqual(len(all_mems), 2)

    def test_merge_similar_returns_int(self):
        with _make_vi() as vi:
            vi.remember("the quick brown fox")
            vi.remember("the quick brown fox jumps")
            count = vi.merge_similar(threshold=0.5)
            self.assertIsInstance(count, int)

    def test_forget_returns_int(self):
        with _make_vi() as vi:
            count = vi.forget()
            self.assertIsInstance(count, int)


if __name__ == "__main__":
    unittest.main()
