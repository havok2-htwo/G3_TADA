from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class GenerateStartupAdminKeyTests(unittest.TestCase):
    def test_script_prints_urlsafe_startup_admin_key(self):
        project_root = Path(__file__).resolve().parents[1]
        script_path = project_root / "tools" / "generate_startup_admin_key.py"

        completed = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            check=True,
        )

        key = completed.stdout.strip()
        self.assertTrue(key.startswith("tada_startup_admin_"))
        self.assertRegex(key, r"^tada_startup_admin_[A-Za-z0-9_-]+$")
        self.assertGreater(len(key), len("tada_startup_admin_") + 10)


if __name__ == "__main__":
    unittest.main()
