from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config_store import ConfigStore


def main() -> int:
    store = ConfigStore(ROOT)
    settings = store.get_settings()

    host = (os.getenv("TADA_SERVER_HOST") or "").strip()
    if not host:
        host = "0.0.0.0" if settings.allow_lan_access else "127.0.0.1"

    port = (os.getenv("TADA_SERVER_PORT") or "7878").strip() or "7878"
    allow_lan = "true" if settings.allow_lan_access else "false"

    print(f"TADA_SERVER_HOST={host}")
    print(f"TADA_SERVER_PORT={port}")
    print(f"TADA_ALLOW_LAN_ACCESS={allow_lan}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
