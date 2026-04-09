from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools import build_release_bundle


class ReleaseBundleTests(unittest.TestCase):
    def test_copy_model_caches_uses_configured_storage_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            backend_dir = project_root / "backend"
            data_dir = backend_dir / "data"
            cache_root = project_root / "custom-cache"
            repo_root = cache_root / "models--HumeAI--tada-1b"
            snapshot_dir = repo_root / "snapshots" / "snapshot-123"

            snapshot_dir.mkdir(parents=True, exist_ok=True)
            (repo_root / "refs").mkdir(parents=True, exist_ok=True)
            (repo_root / "refs" / "main").write_text("snapshot-123", encoding="utf-8")
            (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")

            data_dir.mkdir(parents=True, exist_ok=True)
            (data_dir / "server_settings.json").write_text(
                json.dumps({"model_storage_path": str(cache_root)}),
                encoding="utf-8",
            )

            previous_project_root = build_release_bundle.PROJECT_ROOT
            previous_backend_dir = build_release_bundle.BACKEND_DIR
            try:
                build_release_bundle.PROJECT_ROOT = project_root
                build_release_bundle.BACKEND_DIR = backend_dir
                bundle_root = project_root / "bundle"
                included = build_release_bundle.copy_model_caches(bundle_root)
            finally:
                build_release_bundle.PROJECT_ROOT = previous_project_root
                build_release_bundle.BACKEND_DIR = previous_backend_dir

            model_entry = next(item for item in included if item["model_id"] == "HumeAI/tada-1b")
            self.assertEqual(model_entry["status"], "ready")
            copied_config = bundle_root / ".hf_cache" / "hub" / "models--HumeAI--tada-1b" / "snapshots" / "snapshot-123" / "config.json"
            self.assertTrue(copied_config.exists())


if __name__ == "__main__":
    unittest.main()
