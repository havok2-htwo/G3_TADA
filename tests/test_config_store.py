from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend.config_store import ConfigStore


class ConfigStoreTests(unittest.TestCase):
    def test_settings_are_seeded_and_clamped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous = os.environ.get("TADA_ADMIN_KEY")
            os.environ["TADA_ADMIN_KEY"] = "admin-test-key"
            try:
                store = ConfigStore(Path(temp_dir))
                snapshot = store.export_settings_snapshot()
                self.assertEqual(snapshot["active_model"], "HumeAI/tada-3b-ml")
                self.assertEqual(snapshot["model_precision"], "fp16")
                self.assertIsNone(snapshot["deterministic_seed"])
                self.assertFalse(snapshot["persist_generated_wavs"])
                self.assertEqual(snapshot["prompt_start_trim_steps"], 0)

                updated = store.update_settings(
                    {
                        "max_batch_size": 4,
                        "max_parallel_requests": 9,
                        "steps": 200,
                        "model_precision": "bf16",
                        "deterministic_seed": 1234,
                        "persist_generated_wavs": True,
                        "short_sentence_merge_max_chars": 999,
                        "following_sentence_merge_min_chars": -5,
                        "allow_lan_access": True,
                    }
                )
                self.assertEqual(updated["max_batch_size"], 4)
                self.assertEqual(updated["max_parallel_requests"], 4)
                self.assertEqual(updated["steps"], 128)
                self.assertEqual(updated["model_precision"], "bf16")
                self.assertEqual(updated["deterministic_seed"], 1234)
                self.assertTrue(updated["persist_generated_wavs"])
                self.assertEqual(updated["short_sentence_merge_max_chars"], 200)
                self.assertEqual(updated["following_sentence_merge_min_chars"], 0)
                self.assertTrue(updated["allow_lan_access"])
                self.assertTrue(updated["restart_required"])
                self.assertTrue(store.verify_admin_key("admin-test-key"))
            finally:
                if previous is None:
                    os.environ.pop("TADA_ADMIN_KEY", None)
                else:
                    os.environ["TADA_ADMIN_KEY"] = previous

    def test_admin_key_can_be_rotated_and_persists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous = os.environ.get("TADA_ADMIN_KEY")
            os.environ["TADA_ADMIN_KEY"] = "admin-test-key"
            try:
                project_root = Path(temp_dir)
                store = ConfigStore(project_root)
                self.assertTrue(store.verify_admin_key("admin-test-key"))

                rotated = store.rotate_admin_key(label="Rotated Master")
                self.assertEqual(rotated["label"], "Rotated Master")
                self.assertTrue(store.verify_admin_key(rotated["token"]))
                self.assertFalse(store.verify_admin_key("admin-test-key"))

                listed = store.list_keys()
                self.assertEqual(listed["admin_key"]["label"], "Rotated Master")
                reloaded_store = ConfigStore(project_root)
                self.assertTrue(reloaded_store.verify_admin_key(rotated["token"]))
                self.assertEqual(reloaded_store.list_keys()["admin_key"]["label"], "Rotated Master")
            finally:
                if previous is None:
                    os.environ.pop("TADA_ADMIN_KEY", None)
                else:
                    os.environ["TADA_ADMIN_KEY"] = previous

    def test_startup_admin_key_is_accepted_until_it_expires(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_admin = os.environ.get("TADA_ADMIN_KEY")
            previous_startup = os.environ.get("TADA_STARTUP_ADMIN_KEY")
            previous_ttl = os.environ.get("TADA_STARTUP_ADMIN_KEY_TTL_SECONDS")
            os.environ["TADA_ADMIN_KEY"] = "admin-test-key"
            os.environ["TADA_STARTUP_ADMIN_KEY"] = "startup-admin-test-key"
            os.environ["TADA_STARTUP_ADMIN_KEY_TTL_SECONDS"] = "60"
            try:
                store = ConfigStore(Path(temp_dir))
                self.assertTrue(store.verify_admin_key("admin-test-key"))
                self.assertTrue(store.verify_admin_key("startup-admin-test-key"))

                store._startup_admin_key_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
                self.assertFalse(store.verify_admin_key("startup-admin-test-key"))
            finally:
                if previous_admin is None:
                    os.environ.pop("TADA_ADMIN_KEY", None)
                else:
                    os.environ["TADA_ADMIN_KEY"] = previous_admin
                if previous_startup is None:
                    os.environ.pop("TADA_STARTUP_ADMIN_KEY", None)
                else:
                    os.environ["TADA_STARTUP_ADMIN_KEY"] = previous_startup
                if previous_ttl is None:
                    os.environ.pop("TADA_STARTUP_ADMIN_KEY_TTL_SECONDS", None)
                else:
                    os.environ["TADA_STARTUP_ADMIN_KEY_TTL_SECONDS"] = previous_ttl

    def test_settings_can_be_saved_and_applied_as_presets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous = os.environ.get("TADA_ADMIN_KEY")
            os.environ["TADA_ADMIN_KEY"] = "admin-test-key"
            try:
                store = ConfigStore(Path(temp_dir))
                store.update_settings(
                    {
                        "model_precision": "fp32",
                        "persist_generated_wavs": True,
                        "steps": 17,
                    }
                )
                preset = store.save_settings_preset("My Fast Preset")
                self.assertEqual(preset["name"], "my-fast-preset")

                store.update_settings({"model_precision": "fp16", "persist_generated_wavs": False, "steps": 4})
                applied = store.apply_settings_preset("my-fast-preset")
                self.assertEqual(applied["model_precision"], "fp32")
                self.assertTrue(applied["persist_generated_wavs"])
                self.assertEqual(applied["steps"], 17)

                presets = store.list_settings_presets()
                self.assertEqual(len(presets), 1)
                self.assertEqual(presets[0]["label"], "My Fast Preset")
            finally:
                if previous is None:
                    os.environ.pop("TADA_ADMIN_KEY", None)
                else:
                    os.environ["TADA_ADMIN_KEY"] = previous

if __name__ == "__main__":
    unittest.main()
