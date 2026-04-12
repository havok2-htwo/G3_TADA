from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def _clamp_int(value: Any, *, minimum: int, maximum: int, default: int) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = default
    return max(minimum, min(maximum, numeric))


def _clean_path(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    return text or fallback


@dataclass
class ServerSettings:
    active_model: str
    steps: int
    sentence_chunking: bool
    short_sentence_merge_max_chars: int
    following_sentence_merge_min_chars: int
    allow_lan_access: bool
    stream_start_buffer_ms: int
    stream_chunk_ms: int
    batch_wait_ms: int
    max_batch_size: int
    max_parallel_requests: int
    max_queue_size: int
    model_storage_path: str
    whisper_base_url: str
    
    vad_trimming: bool
    duration_multiplier: float
    vad_threshold_pct: float
    vad_padding_ms: int
    vad_fade_ms: int

    @classmethod
    def from_payload(cls, payload: dict[str, Any], *, fallback_storage_path: str) -> "ServerSettings":
        max_batch_size = _clamp_int(payload.get("max_batch_size"), minimum=1, maximum=128, default=8)
        max_parallel_requests = _clamp_int(
            payload.get("max_parallel_requests"),
            minimum=1,
            maximum=max_batch_size,
            default=min(8, max_batch_size),
        )
        return cls(
            active_model=str(payload.get("active_model") or "HumeAI/tada-3b-ml").strip() or "HumeAI/tada-3b-ml",
            steps=_clamp_int(payload.get("steps"), minimum=1, maximum=128, default=10),
            sentence_chunking=bool(payload.get("sentence_chunking", True)),
            short_sentence_merge_max_chars=_clamp_int(
                payload.get("short_sentence_merge_max_chars"),
                minimum=0,
                maximum=200,
                default=30,
            ),
            following_sentence_merge_min_chars=_clamp_int(
                payload.get("following_sentence_merge_min_chars"),
                minimum=0,
                maximum=500,
                default=20,
            ),
            allow_lan_access=bool(payload.get("allow_lan_access", False)),
            stream_start_buffer_ms=_clamp_int(
                payload.get("stream_start_buffer_ms"),
                minimum=0,
                maximum=5000,
                default=500,
            ),
            stream_chunk_ms=_clamp_int(payload.get("stream_chunk_ms"), minimum=50, maximum=5000, default=500),
            batch_wait_ms=_clamp_int(payload.get("batch_wait_ms"), minimum=0, maximum=5000, default=500),
            max_batch_size=max_batch_size,
            max_parallel_requests=max_parallel_requests,
            max_queue_size=_clamp_int(payload.get("max_queue_size"), minimum=1, maximum=2048, default=256),
            model_storage_path=_clean_path(payload.get("model_storage_path"), fallback_storage_path),
            whisper_base_url=str(payload.get("whisper_base_url") or "").strip(),
            vad_trimming=bool(payload.get("vad_trimming", True)),
            duration_multiplier=float(payload.get("duration_multiplier", 1.1)),
            vad_threshold_pct=float(payload.get("vad_threshold_pct", 0.01)),
            vad_padding_ms=_clamp_int(payload.get("vad_padding_ms"), minimum=0, maximum=5000, default=150),
            vad_fade_ms=_clamp_int(payload.get("vad_fade_ms"), minimum=0, maximum=2000, default=50),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConfigStore:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.backend_dir = project_root / "backend"
        self.data_dir = self.backend_dir / "data"
        self.settings_path = self.data_dir / "server_settings.json"
        self.secrets_path = self.data_dir / "server_secrets.json"
        self._lock = threading.RLock()
        self._startup_admin_key = str(os.getenv("TADA_STARTUP_ADMIN_KEY") or "").strip()
        self._startup_admin_key_expires_at: datetime | None = None
        if self._startup_admin_key:
            ttl_seconds = _clamp_int(
                os.getenv("TADA_STARTUP_ADMIN_KEY_TTL_SECONDS"),
                minimum=1,
                maximum=3600,
                default=300,
            )
            self._startup_admin_key_expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._seed_files()

    def _default_storage_path(self) -> str:
        return str((self.project_root / ".hf_cache" / "hub").resolve(strict=False))

    def _seed_files(self) -> None:
        with self._lock:
            settings = _json_load(self.settings_path)
            if not settings:
                seeded_settings = ServerSettings.from_payload(
                    {
                        "active_model": os.getenv("TADA_MODEL_NAME", "HumeAI/tada-3b-ml"),
                        "steps": os.getenv("TADA_DEFAULT_STEPS", "10"),
                        "sentence_chunking": True,
                        "short_sentence_merge_max_chars": 30,
                        "following_sentence_merge_min_chars": 20,
                        "allow_lan_access": os.getenv("TADA_ALLOW_LAN_ACCESS", "0").lower() in {"1", "true", "yes"},
                        "stream_start_buffer_ms": 500,
                        "stream_chunk_ms": 500,
                        "batch_wait_ms": 500,
                        "max_batch_size": 8,
                        "max_parallel_requests": 8,
                        "max_queue_size": 256,
                        "model_storage_path": os.getenv("HF_HUB_CACHE", self._default_storage_path()),
                        "whisper_base_url": "",
                        "vad_trimming": True,
                        "duration_multiplier": 1.1,
                        "vad_threshold_pct": 0.015,
                        "vad_padding_ms": 150,
                        "vad_fade_ms": 50,
                    },
                    fallback_storage_path=self._default_storage_path(),
                )
                _json_dump(self.settings_path, seeded_settings.to_dict())

            secrets_payload = _json_load(self.secrets_path)
            if not secrets_payload:
                secrets_payload = {
                    "hf_token": os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or "",
                    "whisper_api_key": "",
                    "admin_key": None,
                }
            else:
                secrets_payload.pop("client_keys", None)
            if not secrets_payload.get("admin_key"):
                bootstrap_token = os.getenv("TADA_ADMIN_KEY") or self._generate_raw_key(prefix="tada_admin")
                secrets_payload["admin_key"] = {
                    "id": "admin",
                    "label": "Master Admin Key",
                    "hash": hash_token(bootstrap_token),
                    "created_at": now_iso(),
                    "last_used_at": None,
                }
                print(
                    "TADA admin key initialized. Save this key now because it is only shown once:",
                    bootstrap_token,
                    flush=True,
                )
            _json_dump(self.secrets_path, secrets_payload)

    def apply_runtime_environment(self) -> None:
        settings = self.get_settings()
        os.environ["HF_HUB_CACHE"] = str(Path(settings.model_storage_path).expanduser())
        os.environ["TADA_MODEL_NAME"] = settings.active_model
        os.environ["TADA_DEFAULT_STEPS"] = str(settings.steps)
        
        # Audio VAD Overrides
        os.environ["TADA_VAD_TRIMMING"] = "1" if settings.vad_trimming else "0"
        os.environ["TADA_DURATION_MULTIPLIER"] = str(settings.duration_multiplier)
        os.environ["TADA_VAD_THRESHOLD_PCT"] = str(settings.vad_threshold_pct)
        os.environ["TADA_VAD_PADDING_MS"] = str(settings.vad_padding_ms)
        os.environ["TADA_VAD_FADE_MS"] = str(settings.vad_fade_ms)
        
        hf_token = self.get_hf_token()
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

    def get_settings(self) -> ServerSettings:
        with self._lock:
            return ServerSettings.from_payload(
                _json_load(self.settings_path),
                fallback_storage_path=self._default_storage_path(),
            )

    def update_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            current = self.get_settings()
            merged = current.to_dict()
            merged.update({key: value for key, value in payload.items() if key in merged})
            next_settings = ServerSettings.from_payload(
                merged,
                fallback_storage_path=self._default_storage_path(),
            )
            restart_required = (
                current.model_storage_path != next_settings.model_storage_path
                or current.allow_lan_access != next_settings.allow_lan_access
            )
            _json_dump(self.settings_path, next_settings.to_dict())

            next_hf_token = payload.get("hf_token")
            if next_hf_token is not None:
                self._update_secret_value("hf_token", str(next_hf_token).strip())
            next_whisper_api_key = payload.get("whisper_api_key")
            if next_whisper_api_key is not None:
                self._update_secret_value("whisper_api_key", str(next_whisper_api_key).strip())

        return self.export_settings_snapshot(restart_required=restart_required)

    def export_settings_snapshot(self, *, restart_required: bool = False) -> dict[str, Any]:
        settings = self.get_settings()
        secrets_payload = self._load_secrets()
        return {
            **settings.to_dict(),
            "hf_token_present": bool(secrets_payload.get("hf_token")),
            "whisper_api_key_present": bool(secrets_payload.get("whisper_api_key")),
            "restart_required": restart_required,
        }

    def _load_secrets(self) -> dict[str, Any]:
        with self._lock:
            return _json_load(self.secrets_path)

    def _save_secrets(self, payload: dict[str, Any]) -> None:
        with self._lock:
            _json_dump(self.secrets_path, payload)

    def _mutate_secrets(self, mutator: Any) -> Any:
        with self._lock:
            payload = self._load_secrets()
            result = mutator(payload)
            self._save_secrets(payload)
            return result

    def _update_secret_value(self, field_name: str, value: str) -> None:
        def apply_update(payload: dict[str, Any]) -> None:
            payload[field_name] = value

        self._mutate_secrets(apply_update)

    def get_hf_token(self) -> str:
        return str(self._load_secrets().get("hf_token") or "")

    def get_whisper_api_key(self) -> str:
        return str(self._load_secrets().get("whisper_api_key") or "")

    def list_keys(self) -> dict[str, Any]:
        payload = self._load_secrets()
        admin_key = payload["admin_key"]
        return {
            "admin_key": {
                "id": admin_key["id"],
                "label": admin_key.get("label") or "Master Admin Key",
                "created_at": admin_key["created_at"],
                "last_used_at": admin_key.get("last_used_at"),
            }
        }

    def rotate_admin_key(self, *, label: str = "Master Admin Key") -> dict[str, Any]:
        clean_label = label.strip() or "Master Admin Key"
        plaintext = self._generate_raw_key(prefix="tada_admin")

        def apply_rotate(payload: dict[str, Any]) -> dict[str, Any]:
            payload["admin_key"] = {
                "id": "admin",
                "label": clean_label,
                "hash": hash_token(plaintext),
                "created_at": now_iso(),
                "last_used_at": None,
            }
            return {
                "id": "admin",
                "label": clean_label,
                "token": plaintext,
                "created_at": payload["admin_key"]["created_at"],
            }

        return self._mutate_secrets(apply_rotate)

    def _generate_raw_key(self, *, prefix: str) -> str:
        return f"{prefix}_{secrets.token_urlsafe(24)}"

    def verify_admin_key(self, raw_key: str) -> bool:
        if not raw_key:
            return False

        def apply_verify(payload: dict[str, Any]) -> bool:
            record = payload.get("admin_key") or {}
            is_valid = secrets.compare_digest(record.get("hash") or "", hash_token(raw_key))
            if not is_valid and self._startup_admin_key:
                not_expired = (
                    self._startup_admin_key_expires_at is None
                    or datetime.now(timezone.utc) <= self._startup_admin_key_expires_at
                )
                is_valid = not_expired and secrets.compare_digest(self._startup_admin_key, raw_key)
            if is_valid:
                record["last_used_at"] = now_iso()
                payload["admin_key"] = record
            return is_valid

        return self._mutate_secrets(apply_verify)
