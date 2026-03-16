"""
Unit tests for davinci.llm.client — LMStudioClient.

All tests mock ``lmstudio.Client`` so no real network connection is required.

Run with:
    python -m pytest tests/test_llm_client.py -v
"""

from __future__ import annotations

import unittest
import warnings
from unittest.mock import MagicMock, patch

from davinci.llm.client import LMStudioClient
from davinci.memory.store import MemoryStore


def _make_store() -> MemoryStore:
    """Return a fresh in-memory store for each test."""
    return MemoryStore(db_path=":memory:")


def _make_model(identifier: str, display_name: str) -> MagicMock:
    """Return a mock model object with .identifier and .displayName."""
    m = MagicMock()
    m.identifier = identifier
    m.displayName = display_name
    return m


def _make_chunk(content: str) -> MagicMock:
    """Return a mock stream chunk with a .content attribute."""
    c = MagicMock()
    c.content = content
    return c


class TestNegotiateModel(unittest.TestCase):
    """Model auto-negotiation on construction."""

    @patch("davinci.llm.client.Client")
    def test_auto_negotiate_single_model(self, MockClient):
        model = _make_model("llama-3-8b-instruct", "LLaMA 3 8B Instruct")
        MockClient.return_value.llm.list_loaded.return_value = [model]
        client = LMStudioClient(store=_make_store())
        self.assertEqual(client.model_name, "LLaMA 3 8B Instruct")
        self.assertEqual(client.model_id, "llama-3-8b-instruct")

    @patch("davinci.llm.client.Client")
    def test_auto_negotiate_no_models_raises(self, MockClient):
        MockClient.return_value.llm.list_loaded.return_value = []
        with self.assertRaises(RuntimeError) as ctx:
            LMStudioClient(store=_make_store())
        self.assertIn("No model loaded", str(ctx.exception))

    @patch("davinci.llm.client.Client")
    def test_auto_negotiate_multiple_models_warns(self, MockClient):
        model_a = _make_model("llama-a", "LLaMA A")
        model_b = _make_model("llama-b", "LLaMA B")
        MockClient.return_value.llm.list_loaded.return_value = [model_a, model_b]
        with self.assertWarns(UserWarning):
            client = LMStudioClient(store=_make_store())
        # First model should be selected
        self.assertEqual(client.model_name, "LLaMA A")


class TestModelSize(unittest.TestCase):
    """Model size detection via hints."""

    def _client_with_name(self, MockClient, display_name: str) -> LMStudioClient:
        model = _make_model("test-id", display_name)
        MockClient.return_value.llm.list_loaded.return_value = [model]
        return LMStudioClient(store=_make_store())

    @patch("davinci.llm.client.Client")
    def test_model_size_large(self, MockClient):
        client = self._client_with_name(MockClient, "llama-70b-instruct")
        self.assertEqual(client.model_size, "large")

    @patch("davinci.llm.client.Client")
    def test_model_size_small(self, MockClient):
        client = self._client_with_name(MockClient, "llama-9b-chat")
        self.assertEqual(client.model_size, "small")

    @patch("davinci.llm.client.Client")
    def test_model_size_unknown(self, MockClient):
        client = self._client_with_name(MockClient, "mystery-model")
        self.assertEqual(client.model_size, "unknown")


class TestWarnIfWrongSize(unittest.TestCase):
    """warn_if_wrong_size emits when size mismatches."""

    @patch("davinci.llm.client.Client")
    def test_warn_if_wrong_size(self, MockClient):
        model = _make_model("llama-70b", "LLaMA 70B")
        MockClient.return_value.llm.list_loaded.return_value = [model]
        client = LMStudioClient(store=_make_store())
        self.assertEqual(client.model_size, "large")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            client.warn_if_wrong_size("small")
        self.assertTrue(any("large" in str(w.message) for w in caught))


class TestIngest(unittest.TestCase):
    """ingest() streaming and memory storage."""

    @patch("davinci.llm.client.Client")
    def test_ingest_yields_tokens(self, MockClient):
        raw_chunks = [_make_chunk("Hello"), _make_chunk(" world"), _make_chunk("!")]
        model = _make_model("tiny-7b", "Tiny 7B")
        mock_instance = MockClient.return_value
        mock_instance.llm.list_loaded.return_value = [model]
        mock_instance.llm.respond_stream.return_value = iter(raw_chunks)
        client = LMStudioClient(store=_make_store())
        tokens = list(client.ingest("some text"))
        # First 3 tokens, then the sentinel
        self.assertIn("Hello", tokens)
        self.assertIn(" world", tokens)
        self.assertIn("!", tokens)
        # Sentinel is the last item
        self.assertTrue(tokens[-1].startswith("\n[memory:"))

    @patch("davinci.llm.client.Client")
    def test_ingest_stores_memory(self, MockClient):
        raw_chunks = [_make_chunk("Summary text")]
        model = _make_model("tiny-7b", "Tiny 7B")
        store = _make_store()
        mock_instance = MockClient.return_value
        mock_instance.llm.list_loaded.return_value = [model]
        mock_instance.llm.respond_stream.return_value = iter(raw_chunks)
        client = LMStudioClient(store=store)
        tokens = list(client.ingest("input text"))
        # The sentinel contains the memory UUID
        sentinel = tokens[-1]
        self.assertTrue(sentinel.startswith("\n[memory:"))
        memory_id = sentinel[len("\n[memory:"):-1]
        # The memory should now be in the store
        node = store.retrieve(memory_id)
        self.assertIsNotNone(node)
        self.assertEqual(node.content, "Summary text")


class TestReason(unittest.TestCase):
    """reason() streaming and no-memory fallback."""

    @patch("davinci.llm.client.Client")
    def test_reason_yields_tokens(self, MockClient):
        raw_chunks = [_make_chunk("Token1"), _make_chunk(" Token2")]
        store = _make_store()
        store.store("The Mandelbrot set is beautiful.")
        model = _make_model("llama-70b", "LLaMA 70B")
        mock_instance = MockClient.return_value
        mock_instance.llm.list_loaded.return_value = [model]
        mock_instance.llm.respond_stream.return_value = iter(raw_chunks)
        client = LMStudioClient(store=store)
        tokens = list(client.reason("Mandelbrot"))
        self.assertIn("Token1", tokens)
        self.assertIn(" Token2", tokens)

    @patch("davinci.llm.client.Client")
    def test_reason_no_memories(self, MockClient):
        store = _make_store()  # empty store
        model = _make_model("llama-70b", "LLaMA 70B")
        MockClient.return_value.llm.list_loaded.return_value = [model]
        client = LMStudioClient(store=store)
        tokens = list(client.reason("anything"))
        self.assertEqual(tokens, ["(No relevant memories found.)"])


if __name__ == "__main__":
    unittest.main()
