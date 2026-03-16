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


def _make_client_cls(models: list, chunks: list | None = None) -> MagicMock:
    """Return a mock ``lmstudio.Client`` class.

    Parameters
    ----------
    models:
        List of mock model objects returned by ``list_loaded()``.
    chunks:
        Optional list of mock chunk objects yielded by ``respond_stream()``.
    """
    mock_client_instance = MagicMock()
    mock_client_instance.llm.list_loaded.return_value = models
    if chunks is not None:
        mock_client_instance.llm.respond_stream.return_value = iter(chunks)
    mock_client_cls = MagicMock(return_value=mock_client_instance)
    return mock_client_cls


class TestNegotiateModel(unittest.TestCase):
    """Model auto-negotiation on construction."""

    def test_auto_negotiate_single_model(self):
        model = _make_model("llama-3-8b-instruct", "LLaMA 3 8B Instruct")
        mock_cls = _make_client_cls([model])
        with patch("davinci.llm.client.Client", mock_cls):
            from davinci.llm.client import LMStudioClient
            client = LMStudioClient(store=_make_store())
        self.assertEqual(client.model_name, "LLaMA 3 8B Instruct")
        self.assertEqual(client.model_id, "llama-3-8b-instruct")

    def test_auto_negotiate_no_models_raises(self):
        mock_cls = _make_client_cls([])
        with patch("davinci.llm.client.Client", mock_cls):
            from davinci.llm.client import LMStudioClient
            with self.assertRaises(RuntimeError) as ctx:
                LMStudioClient(store=_make_store())
        self.assertIn("No model loaded", str(ctx.exception))

    def test_auto_negotiate_multiple_models_warns(self):
        model_a = _make_model("llama-a", "LLaMA A")
        model_b = _make_model("llama-b", "LLaMA B")
        mock_cls = _make_client_cls([model_a, model_b])
        with patch("davinci.llm.client.Client", mock_cls):
            from davinci.llm.client import LMStudioClient
            with self.assertWarns(UserWarning):
                client = LMStudioClient(store=_make_store())
        # First model should be selected
        self.assertEqual(client.model_name, "LLaMA A")


class TestModelSize(unittest.TestCase):
    """Model size detection via hints."""

    def _client_with_name(self, display_name: str):
        model = _make_model("test-id", display_name)
        mock_cls = _make_client_cls([model])
        with patch("davinci.llm.client.Client", mock_cls):
            from davinci.llm.client import LMStudioClient
            return LMStudioClient(store=_make_store())

    def test_model_size_large(self):
        client = self._client_with_name("llama-70b-instruct")
        self.assertEqual(client.model_size, "large")

    def test_model_size_small(self):
        client = self._client_with_name("llama-9b-chat")
        self.assertEqual(client.model_size, "small")

    def test_model_size_unknown(self):
        client = self._client_with_name("mystery-model")
        self.assertEqual(client.model_size, "unknown")


class TestWarnIfWrongSize(unittest.TestCase):
    """warn_if_wrong_size emits when size mismatches."""

    def test_warn_if_wrong_size(self):
        model = _make_model("llama-70b", "LLaMA 70B")
        mock_cls = _make_client_cls([model])
        with patch("davinci.llm.client.Client", mock_cls):
            from davinci.llm.client import LMStudioClient
            client = LMStudioClient(store=_make_store())
        self.assertEqual(client.model_size, "large")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            client.warn_if_wrong_size("small")
        self.assertTrue(any("large" in str(w.message) for w in caught))


class TestIngest(unittest.TestCase):
    """ingest() streaming and memory storage."""

    def _make_ingest_client(self, chunks: list):
        model = _make_model("tiny-7b", "Tiny 7B")
        mock_cls = _make_client_cls([model], chunks)
        with patch("davinci.llm.client.Client", mock_cls):
            from davinci.llm.client import LMStudioClient
            return LMStudioClient(store=_make_store()), mock_cls

    def test_ingest_yields_tokens(self):
        raw_chunks = [_make_chunk("Hello"), _make_chunk(" world"), _make_chunk("!")]
        model = _make_model("tiny-7b", "Tiny 7B")
        mock_cls = _make_client_cls([model])
        with patch("davinci.llm.client.Client", mock_cls):
            from davinci.llm.client import LMStudioClient
            client = LMStudioClient(store=_make_store())
        # Set up respond_stream to return our chunks
        client._client.llm.respond_stream.return_value = iter(raw_chunks)
        tokens = list(client.ingest("some text"))
        # First 3 tokens, then the sentinel
        self.assertIn("Hello", tokens)
        self.assertIn(" world", tokens)
        self.assertIn("!", tokens)
        # Sentinel is the last item
        self.assertTrue(tokens[-1].startswith("\n[memory:"))

    def test_ingest_stores_memory(self):
        raw_chunks = [_make_chunk("Summary text")]
        model = _make_model("tiny-7b", "Tiny 7B")
        store = _make_store()
        mock_cls = _make_client_cls([model])
        with patch("davinci.llm.client.Client", mock_cls):
            from davinci.llm.client import LMStudioClient
            client = LMStudioClient(store=store)
        client._client.llm.respond_stream.return_value = iter(raw_chunks)
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

    def test_reason_yields_tokens(self):
        raw_chunks = [_make_chunk("Token1"), _make_chunk(" Token2")]
        store = _make_store()
        store.store("The Mandelbrot set is beautiful.")
        model = _make_model("llama-70b", "LLaMA 70B")
        mock_cls = _make_client_cls([model])
        with patch("davinci.llm.client.Client", mock_cls):
            from davinci.llm.client import LMStudioClient
            client = LMStudioClient(store=store)
        client._client.llm.respond_stream.return_value = iter(raw_chunks)
        tokens = list(client.reason("Mandelbrot"))
        self.assertIn("Token1", tokens)
        self.assertIn(" Token2", tokens)

    def test_reason_no_memories(self):
        store = _make_store()  # empty store
        model = _make_model("llama-70b", "LLaMA 70B")
        mock_cls = _make_client_cls([model])
        with patch("davinci.llm.client.Client", mock_cls):
            from davinci.llm.client import LMStudioClient
            client = LMStudioClient(store=store)
        tokens = list(client.reason("anything"))
        self.assertEqual(tokens, ["(No relevant memories found.)"])


if __name__ == "__main__":
    unittest.main()
