from __future__ import annotations

import os
import sys
import inspect
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
    sys.path.append(str(VENDOR_DIR))


def patch_dataclass_slots_compat() -> None:
    import dataclasses

    if "slots" in inspect.signature(dataclasses.dataclass).parameters:
        return
    original = dataclasses.dataclass

    def compat_dataclass(_cls=None, /, *args, **kwargs):
        kwargs.pop("slots", None)
        if _cls is None:
            return lambda cls: original(cls, *args, **kwargs)
        return original(_cls, *args, **kwargs)

    dataclasses.dataclass = compat_dataclass


def _strip_optional_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_project_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key.startswith("#"):
            continue
        os.environ.setdefault(key, _strip_optional_quotes(value.strip()))


patch_dataclass_slots_compat()
load_project_dotenv()
