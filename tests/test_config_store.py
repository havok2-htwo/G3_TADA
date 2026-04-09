from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend.config_store import ConfigStore


class CoordinatedConfigStore(ConfigStore):
    def __init__(self, project_root: Path):
        super().__init__(project_root)
        self.pause_verify_load = threading.Event()
        self.resume_verify_load = threading.Event()
        self.pause_verify_thread_name: str | None = None

    def _load_secrets(self) -> dict[str, object]:
        payload = super()._load_secrets()
        if (
            self.pause_verify_thread_name
            and threading.current_thread().name == self.pause_verify_thread_name
            and not self.pause_verify_load.is_set()
        ):
            self.pause_verify_load.set()
            self.resume_verify_load.wait(timeout=5)
        return payload


class ConfigStoreTests(unittest.TestCase):
    def test_settings_are_seeded_and_clamped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous = os.environ.get("TADA_ADMIN_KEY")
            os.environ["TADA_ADMIN_KEY"] = "admin-test-key"
            try:
                store = ConfigStore(Path(temp_dir))
                snapshot = store.export_settings_snapshot()
                self.assertEqual(snapshot["active_model"], "HumeAI/tada-3b-ml")

                updated = store.update_settings(
                    {
                        "max_batch_size": 4,
                        "max_parallel_requests": 9,
                        "steps": 200,
                        "short_sentence_merge_max_chars": 999,
                        "following_sentence_merge_min_chars": -5,
                        "allow_lan_access": True,
                    }
                )
                self.assertEqual(updated["max_batch_size"], 4)
                self.assertEqual(updated["max_parallel_requests"], 4)
                self.assertEqual(updated["steps"], 128)
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

    def test_client_keys_can_be_created_verified_and_deleted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous = os.environ.get("TADA_ADMIN_KEY")
            os.environ["TADA_ADMIN_KEY"] = "admin-test-key"
            try:
                store = ConfigStore(Path(temp_dir))
                created = store.create_key(label="Demo", kind="client")
                verified = store.verify_client_key(created["token"])
                self.assertIsNotNone(verified)
                self.assertEqual(verified["label"], "Demo")
                self.assertTrue(store.delete_client_key(created["id"]))
                self.assertIsNone(store.verify_client_key(created["token"]))
            finally:
                if previous is None:
                    os.environ.pop("TADA_ADMIN_KEY", None)
                else:
                    os.environ["TADA_ADMIN_KEY"] = previous

    def test_client_keys_persist_across_store_reloads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous = os.environ.get("TADA_ADMIN_KEY")
            os.environ["TADA_ADMIN_KEY"] = "admin-test-key"
            try:
                project_root = Path(temp_dir)
                store = ConfigStore(project_root)
                created = store.create_key(label="Persistent Demo", kind="client")

                reloaded_store = ConfigStore(project_root)
                verified = reloaded_store.verify_client_key(created["token"])

                self.assertIsNotNone(verified)
                self.assertEqual(verified["label"], "Persistent Demo")
                listed_labels = {item["label"] for item in reloaded_store.list_keys()["client_keys"]}
                self.assertIn("Persistent Demo", listed_labels)
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

    def test_concurrent_client_key_use_does_not_drop_newly_created_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous = os.environ.get("TADA_ADMIN_KEY")
            os.environ["TADA_ADMIN_KEY"] = "admin-test-key"
            try:
                store = CoordinatedConfigStore(Path(temp_dir))
                existing = store.create_key(label="Existing", kind="client")
                store.pause_verify_thread_name = "verify-client-thread"
                results: dict[str, object] = {}

                def verify_existing_key() -> None:
                    results["verified"] = store.verify_client_key(existing["token"])

                verify_thread = threading.Thread(
                    target=verify_existing_key,
                    name="verify-client-thread",
                )
                verify_thread.start()
                self.assertTrue(store.pause_verify_load.wait(timeout=2))

                def create_new_key() -> None:
                    results["created"] = store.create_key(label="Fresh", kind="client")

                create_thread = threading.Thread(target=create_new_key, name="create-client-thread")
                create_thread.start()
                time.sleep(0.2)
                self.assertTrue(create_thread.is_alive())

                store.resume_verify_load.set()
                verify_thread.join(timeout=2)
                create_thread.join(timeout=2)

                self.assertFalse(verify_thread.is_alive())
                self.assertFalse(create_thread.is_alive())
                created = results["created"]
                self.assertIsInstance(created, dict)
                self.assertIsNotNone(store.verify_client_key(created["token"]))

                client_keys = store.list_keys()["client_keys"]
                self.assertEqual(len(client_keys), 2)
                self.assertEqual({item["label"] for item in client_keys}, {"Existing", "Fresh"})
            finally:
                if previous is None:
                    os.environ.pop("TADA_ADMIN_KEY", None)
                else:
                    os.environ["TADA_ADMIN_KEY"] = previous


if __name__ == "__main__":
    unittest.main()
