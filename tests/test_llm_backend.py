"""
Tests for davinci.llm.backend — LLM Backend System
"""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import MagicMock, patch

from davinci.llm.backend import (
    GitHubModelsBackend,
    LLMBackend,
    LLMRegistry,
    LMStudioBackend,
)


class TestLMStudioBackendConstruction(unittest.TestCase):
    def test_defaults(self):
        backend = LMStudioBackend()
        self.assertEqual(backend.name(), "lmstudio")
        self.assertEqual(backend.model_name(), "qwen/qwen3-coder-next")

    def test_custom_params(self):
        backend = LMStudioBackend(
            base_url="http://192.168.1.1:5678",
            model="mymodel/v1",
            api_key="mykey",
        )
        self.assertEqual(backend.model_name(), "mymodel/v1")
        self.assertEqual(backend._base_url, "http://192.168.1.1:5678")
        self.assertEqual(backend._api_key, "mykey")

    def test_name(self):
        self.assertEqual(LMStudioBackend().name(), "lmstudio")

    def test_is_abstract_base(self):
        # LMStudioBackend is a concrete subclass of LLMBackend
        self.assertIsInstance(LMStudioBackend(), LLMBackend)


class TestGitHubModelsBackendConstruction(unittest.TestCase):
    def test_defaults(self):
        backend = GitHubModelsBackend()
        self.assertEqual(backend.name(), "github")
        self.assertEqual(backend.model_name(), "PLACEHOLDER")
        self.assertEqual(backend._api_key_env, "GITHUB_TOKEN")

    def test_custom_params(self):
        backend = GitHubModelsBackend(
            base_url="https://custom.endpoint",
            model="gpt-4o",
            api_key_env="MY_TOKEN",
        )
        self.assertEqual(backend.model_name(), "gpt-4o")
        self.assertEqual(backend._api_key_env, "MY_TOKEN")

    def test_name(self):
        self.assertEqual(GitHubModelsBackend().name(), "github")

    def test_is_abstract_base(self):
        self.assertIsInstance(GitHubModelsBackend(), LLMBackend)


class TestLMStudioBackendChat(unittest.TestCase):
    def _make_mock_response(self, content: str, status: int = 200):
        """Build a mock HTTPResponse that returns the given content."""
        response_data = json.dumps(
            {"choices": [{"message": {"content": content}}]}
        ).encode("utf-8")
        mock_response = MagicMock()
        mock_response.status = status
        mock_response.read.return_value = response_data
        return mock_response

    def test_chat_success(self):
        backend = LMStudioBackend()
        mock_response = self._make_mock_response("Hello from LLM")
        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_response

        with patch("http.client.HTTPConnection", return_value=mock_conn):
            result = backend.chat([{"role": "user", "content": "Hi"}])

        self.assertEqual(result, "Hello from LLM")

    def test_chat_connection_error(self):
        backend = LMStudioBackend()
        with patch("http.client.HTTPConnection", side_effect=ConnectionRefusedError("refused")):
            result = backend.chat([{"role": "user", "content": "Hi"}])

        self.assertTrue(result.startswith("[LLM unavailable:"))

    def test_chat_sends_correct_payload(self):
        backend = LMStudioBackend(model="test/model")
        mock_response = self._make_mock_response("ok")
        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_response

        with patch("http.client.HTTPConnection", return_value=mock_conn):
            backend.chat([{"role": "user", "content": "test"}], max_tokens=500, temperature=0.5)

        # Verify request was made with POST
        call_args = mock_conn.request.call_args
        self.assertEqual(call_args[0][0], "POST")
        body = json.loads(call_args[1]["body"].decode("utf-8"))
        self.assertEqual(body["model"], "test/model")
        self.assertEqual(body["max_tokens"], 500)
        self.assertAlmostEqual(body["temperature"], 0.5)
        self.assertFalse(body["stream"])

    def test_chat_malformed_json_response(self):
        backend = LMStudioBackend()
        mock_response = MagicMock()
        mock_response.read.return_value = b"not json"
        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_response

        with patch("http.client.HTTPConnection", return_value=mock_conn):
            result = backend.chat([{"role": "user", "content": "Hi"}])

        self.assertTrue(result.startswith("[LLM unavailable:"))


class TestLMStudioBackendIsAvailable(unittest.TestCase):
    def test_is_available_true(self):
        backend = LMStudioBackend()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_response

        with patch("http.client.HTTPConnection", return_value=mock_conn):
            self.assertTrue(backend.is_available())

    def test_is_available_false_non_200(self):
        backend = LMStudioBackend()
        mock_response = MagicMock()
        mock_response.status = 404
        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_response

        with patch("http.client.HTTPConnection", return_value=mock_conn):
            self.assertFalse(backend.is_available())

    def test_is_available_false_connection_error(self):
        backend = LMStudioBackend()
        with patch("http.client.HTTPConnection", side_effect=OSError("no connection")):
            self.assertFalse(backend.is_available())


class TestGitHubModelsBackendIsAvailable(unittest.TestCase):
    def test_available_with_env_var(self):
        backend = GitHubModelsBackend(api_key_env="TEST_GH_TOKEN")
        with patch.dict(os.environ, {"TEST_GH_TOKEN": "ghp_test123"}):
            self.assertTrue(backend.is_available())

    def test_not_available_without_env_var(self):
        backend = GitHubModelsBackend(api_key_env="TEST_GH_TOKEN_MISSING")
        env = {k: v for k, v in os.environ.items() if k != "TEST_GH_TOKEN_MISSING"}
        with patch.dict(os.environ, env, clear=True):
            self.assertFalse(backend.is_available())

    def test_not_available_empty_env_var(self):
        backend = GitHubModelsBackend(api_key_env="TEST_GH_TOKEN_EMPTY")
        with patch.dict(os.environ, {"TEST_GH_TOKEN_EMPTY": ""}):
            self.assertFalse(backend.is_available())


class TestGitHubModelsBackendChat(unittest.TestCase):
    def test_chat_no_api_key_returns_error(self):
        backend = GitHubModelsBackend(api_key_env="MISSING_TOKEN_XYZ")
        env = {k: v for k, v in os.environ.items() if k != "MISSING_TOKEN_XYZ"}
        with patch.dict(os.environ, env, clear=True):
            result = backend.chat([{"role": "user", "content": "Hi"}])
        self.assertTrue(result.startswith("[LLM unavailable:"))

    def test_chat_with_api_key_success(self):
        backend = GitHubModelsBackend(
            base_url="http://127.0.0.1:9999",
            api_key_env="TEST_GH_TOKEN",
        )
        response_data = json.dumps(
            {"choices": [{"message": {"content": "GitHub response"}}]}
        ).encode("utf-8")
        mock_response = MagicMock()
        mock_response.read.return_value = response_data
        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_response

        with patch.dict(os.environ, {"TEST_GH_TOKEN": "tok123"}):
            with patch("http.client.HTTPConnection", return_value=mock_conn):
                result = backend.chat([{"role": "user", "content": "Hi"}])

        self.assertEqual(result, "GitHub response")


class TestLLMRegistry(unittest.TestCase):
    def test_pre_registered_backends(self):
        registry = LLMRegistry()
        available = registry.available()
        self.assertIn("lmstudio", available)
        self.assertIn("github", available)

    def test_register_and_get(self):
        registry = LLMRegistry()

        class DummyBackend(LLMBackend):
            def chat(self, messages, max_tokens=900, temperature=0.7):
                return "dummy"

            def is_available(self):
                return True

            def name(self):
                return "dummy"

            def model_name(self):
                return "dummy/v1"

        registry.register("dummy", DummyBackend)
        self.assertIn("dummy", registry.available())
        backend = registry.get("dummy")
        self.assertIsInstance(backend, DummyBackend)

    def test_get_lmstudio(self):
        registry = LLMRegistry()
        backend = registry.get("lmstudio")
        self.assertIsInstance(backend, LMStudioBackend)

    def test_get_github(self):
        registry = LLMRegistry()
        backend = registry.get("github")
        self.assertIsInstance(backend, GitHubModelsBackend)

    def test_get_unknown_raises(self):
        registry = LLMRegistry()
        with self.assertRaises(KeyError):
            registry.get("nonexistent_backend")

    def test_get_with_kwargs(self):
        registry = LLMRegistry()
        backend = registry.get("lmstudio", model="custom/model")
        self.assertEqual(backend.model_name(), "custom/model")


if __name__ == "__main__":
    unittest.main()
