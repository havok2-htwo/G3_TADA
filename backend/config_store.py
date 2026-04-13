from __future__ import annotations

import hashlib
import json
import os
import re
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


def _clamp_float(value: Any, *, minimum: float, maximum: float, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(minimum, min(maximum, numeric))


def _optional_int(value: Any, *, minimum: int, maximum: int) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        numeric = int(text)
    except (TypeError, ValueError):
        return None
    return max(minimum, min(maximum, numeric))


def _normalize_model_precision(value: Any, *, default: str = "fp16") -> str:
    candidate = str(value or default).strip().lower() or default
    if candidate not in {"fp16", "bf16", "fp32"}:
        return default
    return candidate


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default


def _slugify_preset_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip()).strip("-._").lower()
    return cleaned or "preset"


def _clean_path(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    return text or fallback


@dataclass
class ServerSettings:
    active_model: str
    model_precision: str
    deterministic_seed: int | None
    persist_generated_wavs: bool
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
    prompt_start_trim_steps: int
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
            model_precision=_normalize_model_precision(payload.get("model_precision"), default="fp16"),
            deterministic_seed=_optional_int(payload.get("deterministic_seed"), minimum=0, maximum=2**63 - 1),
            persist_generated_wavs=_coerce_bool(payload.get("persist_generated_wavs"), default=False),
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
            prompt_start_trim_steps=_clamp_int(
                payload.get("prompt_start_trim_steps"),
                minimum=0,
                maximum=12,
                default=0,
            ),
            vad_threshold_pct=_clamp_float(payload.get("vad_threshold_pct"), minimum=0.0, maximum=10.0, default=0.015),
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
        self.presets_dir = self.data_dir / "presets"
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
        self.presets_dir.mkdir(parents=True, exist_ok=True)
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
                        "model_precision": os.getenv("TADA_MODEL_PRECISION", "fp16"),
                        "deterministic_seed": os.getenv("TADA_DETERMINISTIC_SEED"),
                        "persist_generated_wavs": False,
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
                        "prompt_start_trim_steps": 0,
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
        os.environ["TADA_MODEL_PRECISION"] = settings.model_precision
        os.environ["TADA_DEFAULT_STEPS"] = str(settings.steps)

        if settings.deterministic_seed is None:
            os.environ.pop("TADA_DETERMINISTIC_SEED", None)
        else:
            os.environ["TADA_DETERMINISTIC_SEED"] = str(settings.deterministic_seed)

        os.environ["TADA_VAD_TRIMMING"] = "1" if settings.vad_trimming else "0"
        os.environ["TADA_PROMPT_START_TRIM_STEPS"] = str(settings.prompt_start_trim_steps)
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

    @staticmethod
    def _settings_field_names() -> set[str]:
        return set(ServerSettings.__dataclass_fields__.keys())

    def export_settings_payload(self) -> dict[str, Any]:
        return self.get_settings().to_dict()

    def _extract_settings_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Settings payload must be a JSON object.")
        candidate = payload.get("settings") if isinstance(payload.get("settings"), dict) else payload
        if not isinstance(candidate, dict):
            raise ValueError("Imported preset does not contain a valid 'settings' object.")
        fields = self._settings_field_names()
        filtered = {key: candidate[key] for key in fields if key in candidate}
        if not filtered:
            raise ValueError("No known server settings were found in the imported payload.")
        return filtered

    def import_settings_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.update_settings(self._extract_settings_payload(payload))

    def list_settings_presets(self) -> list[dict[str, Any]]:
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        presets: list[dict[str, Any]] = []
        for path in sorted(self.presets_dir.glob("*.json"), key=lambda item: item.name.lower()):
            try:
                payload = _json_load(path)
                settings_payload = self._extract_settings_payload(payload)
            except Exception:
                continue
            presets.append(
                {
                    "name": path.stem,
                    "label": str(payload.get("preset_name") or path.stem),
                    "file_name": path.name,
                    "updated_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
                    "settings": settings_payload,
                }
            )
        return presets

    def save_settings_preset(self, name: str) -> dict[str, Any]:
        label = str(name or "").strip()
        if not label:
            raise ValueError("Preset name must not be empty.")
        file_stem = _slugify_preset_name(label)
        path = self.presets_dir / f"{file_stem}.json"
        payload = {
            "preset_name": label,
            "saved_at": now_iso(),
            "settings": self.export_settings_payload(),
        }
        _json_dump(path, payload)
        return {
            "name": file_stem,
            "label": label,
            "file_name": path.name,
            "updated_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
        }

    def apply_settings_preset(self, name: str) -> dict[str, Any]:
        preset_name = str(name or "").strip()
        if not preset_name:
            raise ValueError("Preset name must not be empty.")
        candidate_names = []
        if preset_name.lower().endswith(".json"):
            candidate_names.append(preset_name)
        candidate_names.append(f"{preset_name}.json")
        candidate_names.append(f"{_slugify_preset_name(preset_name)}.json")

        preset_path: Path | None = None
        for candidate in candidate_names:
            path = self.presets_dir / candidate
            if path.exists():
                preset_path = path
                break
        if preset_path is None:
            raise FileNotFoundError(f"Preset '{preset_name}' was not found in {self.presets_dir}.")

        payload = _json_load(preset_path)
        return self.import_settings_payload(payload)

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
