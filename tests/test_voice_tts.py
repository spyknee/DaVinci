"""
Unit tests for davinci.voice.tts — TTS backend system (Layer 4).

Run with:
    python -m pytest tests/test_voice_tts.py -v
"""

from __future__ import annotations

import io
import sys
import unittest

from davinci.voice.tts import TTSBackend, TTSRegistry, StubTTS


class TestStubTTSOutput(unittest.TestCase):
    """StubTTS prints to stdout."""

    def test_speak_prints_to_stdout(self):
        tts = StubTTS()
        captured = io.StringIO()
        sys.stdout = captured
        try:
            tts.speak("Hello, world!")
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn("Hello, world!", captured.getvalue())

    def test_speak_multiple_lines(self):
        tts = StubTTS()
        captured = io.StringIO()
        sys.stdout = captured
        try:
            tts.speak("first")
            tts.speak("second")
        finally:
            sys.stdout = sys.__stdout__
        output = captured.getvalue()
        self.assertIn("first", output)
        self.assertIn("second", output)


class TestStubTTSCollectsOutput(unittest.TestCase):
    """StubTTS collects output in list."""

    def test_appends_to_output_list(self):
        spoken: list[str] = []
        tts = StubTTS(output=spoken)
        tts.speak("alpha")
        tts.speak("beta")
        self.assertEqual(spoken, ["alpha", "beta"])

    def test_output_list_grows_with_each_speak(self):
        spoken: list[str] = []
        tts = StubTTS(output=spoken)
        for i in range(5):
            tts.speak(f"item {i}")
        self.assertEqual(len(spoken), 5)

    def test_no_output_list_does_not_error(self):
        tts = StubTTS()  # output not provided
        tts.speak("no list provided")  # should not raise


class TestStubTTSProperties(unittest.TestCase):
    """StubTTS availability and name."""

    def test_is_available_true(self):
        tts = StubTTS()
        self.assertTrue(tts.is_available())

    def test_name_is_stub(self):
        tts = StubTTS()
        self.assertEqual(tts.name(), "stub")

    def test_is_ttsbackend_subclass(self):
        self.assertTrue(issubclass(StubTTS, TTSBackend))


class TestTTSRegistry(unittest.TestCase):
    """TTSRegistry registers and retrieves backends."""

    def test_stub_is_preregistered(self):
        self.assertIn("stub", TTSRegistry.available())

    def test_get_stub_returns_stub_tts(self):
        tts = TTSRegistry.get("stub")
        self.assertIsInstance(tts, StubTTS)

    def test_available_lists_registered_backends(self):
        names = TTSRegistry.available()
        self.assertIsInstance(names, list)
        self.assertIn("stub", names)

    def test_get_unknown_raises_key_error(self):
        with self.assertRaises(KeyError):
            TTSRegistry.get("nonexistent_backend_xyz")

    def test_get_forwards_config_to_backend(self):
        spoken: list[str] = []
        tts = TTSRegistry.get("stub", output=spoken)
        self.assertIsInstance(tts, StubTTS)
        tts.speak("from registry")
        self.assertIn("from registry", spoken)


class TestCustomTTSRegistration(unittest.TestCase):
    """Custom backend registration works."""

    def test_register_and_retrieve_custom_backend(self):
        class SilentTTS(TTSBackend):
            def __init__(self, **config):
                super().__init__(**config)
                self.spoken: list[str] = []

            def speak(self, text: str) -> None:
                self.spoken.append(text)  # silent — no stdout

            def is_available(self) -> bool:
                return True

            def name(self) -> str:
                return "silent"

        TTSRegistry.register("silent_test", SilentTTS)
        self.assertIn("silent_test", TTSRegistry.available())
        tts = TTSRegistry.get("silent_test")
        self.assertIsInstance(tts, SilentTTS)
        tts.speak("quiet")
        self.assertEqual(tts.spoken, ["quiet"])

    def test_registered_backend_appears_in_available(self):
        class DummyTTS(TTSBackend):
            def speak(self, text: str) -> None:
                pass

            def is_available(self) -> bool:
                return True

            def name(self) -> str:
                return "dummy"

        TTSRegistry.register("dummy_tts_test", DummyTTS)
        self.assertIn("dummy_tts_test", TTSRegistry.available())


if __name__ == "__main__":
    unittest.main()
