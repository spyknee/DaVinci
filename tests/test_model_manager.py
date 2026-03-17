"""
Tests for davinci.llm.manager — Model Manager
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from davinci.llm.backend import LLMBackend, LMStudioBackend, GitHubModelsBackend
from davinci.llm.manager import ModelManager


class TestModelManagerDefaults(unittest.TestCase):
    def test_loads_default_config(self):
        mm = ModelManager()
        self.assertIn("qwen", mm.available_models())
        self.assertIn("qwen35", mm.available_models())
        self.assertIn("model3", mm.available_models())

    def test_default_active_model(self):
        mm = ModelManager()
        self.assertEqual(mm.active_name(), "qwen")

    def test_active_returns_backend(self):
        mm = ModelManager()
        backend = mm.active()
        self.assertIsInstance(backend, LLMBackend)

    def test_active_model_name(self):
        mm = ModelManager()
        self.assertEqual(mm.active_model_name(), "qwen/qwen3-coder-next")


class TestModelManagerFromProfile(unittest.TestCase):
    def setUp(self):
        self.profile = {
            "active_model": "qwen35",
            "models": {
                "qwen35": {
                    "base_url": "http://127.0.0.1:1234",
                    "model": "qwen/qwen3.5-9b",
                    "api_key": "lm-studio",
                    "provider": "lmstudio",
                },
                "model3": {
                    "base_url": "https://models.inference.ai.azure.com",
                    "model": "PLACEHOLDER",
                    "api_key_env": "GITHUB_TOKEN",
                    "provider": "github",
                },
            },
        }

    def test_loads_from_profile(self):
        mm = ModelManager(profile=self.profile)
        self.assertEqual(mm.active_name(), "qwen35")
        self.assertEqual(mm.active_model_name(), "qwen/qwen3.5-9b")

    def test_available_models_from_profile(self):
        mm = ModelManager(profile=self.profile)
        models = mm.available_models()
        self.assertIn("qwen35", models)
        self.assertIn("model3", models)
        self.assertNotIn("qwen", models)


class TestModelManagerSwitch(unittest.TestCase):
    def test_switch_success(self):
        mm = ModelManager()
        result = mm.switch("qwen35")
        self.assertTrue(result)
        self.assertEqual(mm.active_name(), "qwen35")

    def test_switch_unknown_returns_false(self):
        mm = ModelManager()
        result = mm.switch("nonexistent_model")
        self.assertFalse(result)
        # Active should remain unchanged
        self.assertEqual(mm.active_name(), "qwen")

    def test_switch_changes_backend(self):
        mm = ModelManager()
        mm.switch("qwen35")
        backend = mm.active()
        self.assertEqual(backend.model_name(), "qwen/qwen3.5-9b")


class TestModelManagerToggle(unittest.TestCase):
    def test_toggle_cycles(self):
        mm = ModelManager()
        initial = mm.active_name()
        names = mm.available_models()
        initial_idx = names.index(initial)
        expected_next = names[(initial_idx + 1) % len(names)]
        new_name = mm.toggle()
        self.assertEqual(new_name, expected_next)
        self.assertEqual(mm.active_name(), expected_next)

    def test_toggle_wraps_around(self):
        mm = ModelManager()
        names = mm.available_models()
        # Toggle through all models
        for _ in range(len(names)):
            mm.toggle()
        # Should be back at original after full cycle
        original_idx = names.index("qwen")
        # After len(names) toggles we are back at qwen
        self.assertEqual(mm.active_name(), "qwen")

    def test_toggle_in_two_model_profile(self):
        mm = ModelManager(profile={
            "active_model": "a",
            "models": {
                "a": {"base_url": "http://127.0.0.1:1234", "model": "m/a", "provider": "lmstudio"},
                "b": {"base_url": "http://127.0.0.1:1234", "model": "m/b", "provider": "lmstudio"},
            },
        })
        self.assertEqual(mm.toggle(), "b")
        self.assertEqual(mm.toggle(), "a")


class TestModelManagerStatus(unittest.TestCase):
    def test_status_keys(self):
        mm = ModelManager()
        status = mm.status()
        self.assertIn("active", status)
        self.assertIn("model", status)
        self.assertIn("base_url", status)
        self.assertIn("available", status)

    def test_status_active_name(self):
        mm = ModelManager()
        self.assertEqual(mm.status()["active"], "qwen")

    def test_status_model_name(self):
        mm = ModelManager()
        self.assertEqual(mm.status()["model"], "qwen/qwen3-coder-next")

    def test_status_base_url(self):
        mm = ModelManager()
        self.assertEqual(mm.status()["base_url"], "http://127.0.0.1:1234")


class TestModelManagerLazyInstantiation(unittest.TestCase):
    def test_backend_not_created_until_used(self):
        mm = ModelManager()
        # No backends created yet
        self.assertEqual(len(mm._backends), 0)
        # Access active() to trigger creation
        _ = mm.active()
        self.assertEqual(len(mm._backends), 1)

    def test_same_instance_returned_on_second_access(self):
        mm = ModelManager()
        b1 = mm.active()
        b2 = mm.active()
        self.assertIs(b1, b2)


if __name__ == "__main__":
    unittest.main()
