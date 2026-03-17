"""
Tests for davinci.llm.profile — Profile System
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from davinci.llm.profile import Profile


class TestProfileDefaults(unittest.TestCase):
    def test_load_default_profile(self):
        p = Profile()
        self.assertIn("llm", p.as_dict())
        self.assertIn("memory", p.as_dict())

    def test_default_active_model(self):
        p = Profile()
        self.assertEqual(p.get("llm.active_model"), "qwen")

    def test_default_models_exist(self):
        p = Profile()
        models = p.get("llm.models", {})
        self.assertIn("qwen", models)
        self.assertIn("qwen35", models)
        self.assertIn("model3", models)

    def test_default_qwen_model_id(self):
        p = Profile()
        self.assertEqual(
            p.get("llm.models.qwen.model"),
            "qwen/qwen3-coder-next",
        )

    def test_default_qwen35_model_id(self):
        p = Profile()
        self.assertEqual(
            p.get("llm.models.qwen35.model"),
            "qwen/qwen3.5-9b",
        )

    def test_default_model3_provider(self):
        p = Profile()
        self.assertEqual(p.get("llm.models.model3.provider"), "github")

    def test_default_returns_none_for_missing(self):
        p = Profile()
        self.assertIsNone(p.get("nonexistent.key"))

    def test_default_returns_given_default(self):
        p = Profile()
        self.assertEqual(p.get("nonexistent.key", "fallback"), "fallback")


class TestProfileDotPathAccess(unittest.TestCase):
    def test_get_top_level(self):
        p = Profile()
        self.assertIsNotNone(p.get("llm"))

    def test_get_nested_two_levels(self):
        p = Profile()
        active = p.get("llm.active_model")
        self.assertEqual(active, "qwen")

    def test_get_nested_three_levels(self):
        p = Profile()
        url = p.get("llm.models.qwen.base_url")
        self.assertEqual(url, "http://127.0.0.1:1234")

    def test_set_top_level(self):
        p = Profile()
        p.set("custom_key", "custom_value")
        self.assertEqual(p.get("custom_key"), "custom_value")

    def test_set_nested(self):
        p = Profile()
        p.set("llm.active_model", "qwen35")
        self.assertEqual(p.get("llm.active_model"), "qwen35")

    def test_set_creates_intermediate_dicts(self):
        p = Profile()
        p.set("new.nested.key", 42)
        self.assertEqual(p.get("new.nested.key"), 42)


class TestProfileSaveLoad(unittest.TestCase):
    def test_save_and_reload(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmppath = f.name

        try:
            p = Profile(tmppath)
            p.set("llm.active_model", "qwen35")
            p.save()

            p2 = Profile(tmppath)
            self.assertEqual(p2.get("llm.active_model"), "qwen35")
        finally:
            os.unlink(tmppath)

    def test_save_does_nothing_without_path(self):
        p = Profile()
        p.set("llm.active_model", "qwen35")
        p.save()  # Should not raise

    def test_load_missing_file_uses_defaults(self):
        p = Profile("/tmp/nonexistent_davinci_profile_xyz.json")
        self.assertEqual(p.get("llm.active_model"), "qwen")

    def test_load_merges_with_defaults(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"llm": {"active_model": "qwen35"}}, f)
            tmppath = f.name

        try:
            p = Profile(tmppath)
            # Custom value is loaded
            self.assertEqual(p.get("llm.active_model"), "qwen35")
            # Default values still present
            self.assertIn("qwen", p.get("llm.models", {}))
        finally:
            os.unlink(tmppath)


class TestProfileLLMConfig(unittest.TestCase):
    def test_llm_config_returns_dict(self):
        p = Profile()
        cfg = p.llm_config()
        self.assertIsInstance(cfg, dict)

    def test_llm_config_has_active_model(self):
        p = Profile()
        cfg = p.llm_config()
        self.assertIn("active_model", cfg)

    def test_llm_config_has_models(self):
        p = Profile()
        cfg = p.llm_config()
        self.assertIn("models", cfg)
        self.assertIn("qwen", cfg["models"])


class TestProfileContains(unittest.TestCase):
    def test_contains_existing_key(self):
        p = Profile()
        self.assertIn("llm", p)

    def test_not_contains_missing_key(self):
        p = Profile()
        self.assertNotIn("nonexistent_xyz", p)


if __name__ == "__main__":
    unittest.main()
