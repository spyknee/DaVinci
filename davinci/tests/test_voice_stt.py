"""
Unit tests for davinci.voice.stt — STT backend system (Layer 4).

Run with:
    python -m pytest tests/test_voice_stt.py -v
"""

from __future__ import annotations

import unittest

from davinci.voice.stt import STTBackend, STTRegistry, StubSTT


class TestStubSTTResponses(unittest.TestCase):
    """StubSTT returns pre-canned responses in order."""

    def test_returns_responses_in_order(self):
        stt = StubSTT(responses=["alpha", "beta", "gamma"])
        self.assertEqual(stt.listen(), "alpha")
        self.assertEqual(stt.listen(), "beta")
        self.assertEqual(stt.listen(), "gamma")

    def test_cycles_through_responses(self):
        stt = StubSTT(responses=["one", "two"])
        self.assertEqual(stt.listen(), "one")
        self.assertEqual(stt.listen(), "two")
        self.assertEqual(stt.listen(), "one")   # wraps around
        self.assertEqual(stt.listen(), "two")

    def test_single_response_cycles(self):
        stt = StubSTT(responses=["only"])
        for _ in range(5):
            self.assertEqual(stt.listen(), "only")


class TestStubSTTFallback(unittest.TestCase):
    """StubSTT with empty / no responses falls back to input()."""

    def test_empty_list_uses_input(self):
        """With an empty responses list, listen() should call input()."""
        stt = StubSTT(responses=[])
        # Patch input() so we don't block the test.
        import builtins
        original_input = builtins.input
        try:
            builtins.input = lambda _prompt="": "keyboard input"
            result = stt.listen()
            self.assertEqual(result, "keyboard input")
        finally:
            builtins.input = original_input

    def test_no_responses_uses_input(self):
        stt = StubSTT()  # no responses kwarg
        import builtins
        original_input = builtins.input
        try:
            builtins.input = lambda _prompt="": "typed text"
            result = stt.listen()
            self.assertEqual(result, "typed text")
        finally:
            builtins.input = original_input


class TestStubSTTProperties(unittest.TestCase):
    """StubSTT availability and name."""

    def test_is_available_true(self):
        stt = StubSTT(responses=["x"])
        self.assertTrue(stt.is_available())

    def test_name_is_stub(self):
        stt = StubSTT()
        self.assertEqual(stt.name(), "stub")

    def test_is_sttbackend_subclass(self):
        self.assertTrue(issubclass(StubSTT, STTBackend))


class TestSTTRegistry(unittest.TestCase):
    """STTRegistry registers and retrieves backends."""

    def test_stub_is_preregistered(self):
        self.assertIn("stub", STTRegistry.available())

    def test_get_stub_returns_stub_stt(self):
        stt = STTRegistry.get("stub")
        self.assertIsInstance(stt, StubSTT)

    def test_available_lists_registered_backends(self):
        names = STTRegistry.available()
        self.assertIsInstance(names, list)
        self.assertIn("stub", names)

    def test_get_unknown_raises_key_error(self):
        with self.assertRaises(KeyError):
            STTRegistry.get("nonexistent_backend_xyz")

    def test_get_forwards_config_to_backend(self):
        stt = STTRegistry.get("stub", responses=["hello", "world"])
        self.assertIsInstance(stt, StubSTT)
        self.assertEqual(stt.listen(), "hello")
        self.assertEqual(stt.listen(), "world")


class TestCustomSTTRegistration(unittest.TestCase):
    """Custom backend registration works."""

    def test_register_and_retrieve_custom_backend(self):
        class EchoSTT(STTBackend):
            def __init__(self, **config):
                super().__init__(**config)
                self._msg = config.get("message", "echo")

            def listen(self) -> str:
                return self._msg

            def is_available(self) -> bool:
                return True

            def name(self) -> str:
                return "echo"

        STTRegistry.register("echo_test", EchoSTT)
        self.assertIn("echo_test", STTRegistry.available())
        stt = STTRegistry.get("echo_test", message="hello from echo")
        self.assertIsInstance(stt, EchoSTT)
        self.assertEqual(stt.listen(), "hello from echo")
        self.assertEqual(stt.name(), "echo")

    def test_registered_backend_is_available_via_list(self):
        class DummySTT(STTBackend):
            def listen(self) -> str:
                return ""

            def is_available(self) -> bool:
                return True

            def name(self) -> str:
                return "dummy"

        STTRegistry.register("dummy_stt_test", DummySTT)
        self.assertIn("dummy_stt_test", STTRegistry.available())


if __name__ == "__main__":
    unittest.main()
