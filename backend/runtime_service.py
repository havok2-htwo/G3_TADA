from __future__ import annotations

import io
import gc
import hashlib
import importlib.util
import json
import os
import shutil
import threading
import time
import types
import uuid
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import get_token, snapshot_download

from backend.config_store import ConfigStore, now_iso
from backend.hf_cache import resolve_cached_repo_root, resolve_cached_snapshot, resolve_pretrained_source
from backend.prompt_batch import ensure_directory, merge_encoder_outputs

try:
    from tada.modules.encoder import Encoder, EncoderOutput
    from tada.modules.tada import InferenceOptions, TadaForCausalLM

    TADA_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - runtime dependency availability varies.
    Encoder = None  # type: ignore[assignment]
    EncoderOutput = None  # type: ignore[assignment]
    InferenceOptions = None  # type: ignore[assignment]
    TadaForCausalLM = None  # type: ignore[assignment]
    TADA_IMPORT_ERROR = exc


DEFAULT_FLOW_STEPS = max(1, int(os.getenv("TADA_DEFAULT_STEPS", "10")))
DEFAULT_PROGRESS_INTERVAL_STEPS = max(1, int(os.getenv("TADA_PROGRESS_INTERVAL_STEPS", "12")))
MIN_REFERENCE_AUDIO_SECONDS = 3.0
MAX_REFERENCE_AUDIO_SECONDS = 14.5
DEFAULT_REFERENCE_AUDIO_SECONDS = 10.0
REFERENCE_TAIL_SILENCE_SECONDS = 0.5
REFERENCE_TAIL_FADE_OUT_MS = 30
BENCHMARK_MAX_SOURCE_SECONDS = 14.0
BENCHMARK_FIXTURES_DIR = Path("tests") / "fixtures" / "benchmark_voices"
GENERATED_AUDIO_MEMORY_CACHE_LIMIT = 100

AVAILABLE_MODELS: dict[str, dict[str, str]] = {
    "HumeAI/tada-3b-ml": {
        "label": "TADA 3B ML",
        "description": "Beste Qualitaet, aber am langsamsten.",
    },
    "HumeAI/tada-1b": {
        "label": "TADA 1B",
        "description": "Leichter und deutlich besser fuer Speed/8 GB VRAM.",
    },
}

MODEL_DOWNLOAD_TARGETS: dict[str, dict[str, Any]] = {
    "HumeAI/tada-3b-ml": {"label": "TADA 3B ML", "kind": "model", "approx_size_gb": 6.5},
    "HumeAI/tada-1b": {"label": "TADA 1B", "kind": "model", "approx_size_gb": 2.3},
    "HumeAI/tada-codec": {"label": "TADA Codec", "kind": "codec", "approx_size_gb": 0.6},
    "meta-llama/Llama-3.2-1B": {"label": "Meta Llama 3.2 1B Tokenizer", "kind": "tokenizer", "approx_size_gb": 2.4},
}

MODEL_NAME_ALIASES = {
    "1b": "HumeAI/tada-1b",
    "tada-1b": "HumeAI/tada-1b",
    "humeai/tada-1b": "HumeAI/tada-1b",
    "3b": "HumeAI/tada-3b-ml",
    "tada-3b": "HumeAI/tada-3b-ml",
    "tada-3b-ml": "HumeAI/tada-3b-ml",
    "humeai/tada-3b-ml": "HumeAI/tada-3b-ml",
}

SUPPORTED_LANGUAGES = {
    "en": "English",
    "ar": "Arabic",
    "ch": "Chinese",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "pl": "Polish",
    "pt": "Portuguese",
}

SUPPORTED_MODEL_PRECISIONS = ("fp16", "bf16", "fp32")
BYTES_PER_GB = 1024 ** 3


def slugify(value: str) -> str:
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return slug.strip("-") or "voice"


def normalize_voice_name(value: str) -> str:
    return " ".join(value.strip().split()).casefold()


def _resolve_storage_root(project_root: Path, storage_path: str) -> Path:
    path = Path(storage_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False)


def _directory_size_gb(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None

    total_bytes = 0
    for current_root, _, files in os.walk(path):
        for file_name in files:
            file_path = Path(current_root) / file_name
            try:
                total_bytes += file_path.stat().st_size
            except OSError:
                continue

    return round(total_bytes / BYTES_PER_GB, 2)


class DuplicateVoiceNameError(ValueError):
    def __init__(self, *, cleaned_name: str, existing_voice_id: str, existing_voice_name: str):
        super().__init__(f"A voice named '{cleaned_name}' already exists. Please choose a different name.")
        self.cleaned_name = cleaned_name
        self.existing_voice_id = existing_voice_id
        self.existing_voice_name = existing_voice_name


def safe_json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def load_audio_tensor(path: Path) -> tuple[torch.Tensor, int]:
    try:
        audio, sample_rate = sf.read(str(path), always_2d=True)
        waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).transpose(0, 1)
        return waveform.contiguous(), int(sample_rate)
    except Exception:
        import librosa

        audio, sample_rate = librosa.load(str(path), sr=None, mono=False)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        return waveform.contiguous(), int(sample_rate)


def save_audio_tensor(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    audio = waveform.detach().float().cpu()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    sf.write(str(path), audio.transpose(0, 1).numpy(), int(sample_rate))


def resample_audio_tensor(waveform: torch.Tensor, *, sample_rate: int, target_sample_rate: int) -> torch.Tensor:
    audio = waveform.detach().float().cpu().contiguous()
    if sample_rate == target_sample_rate:
        return audio
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2 or int(audio.shape[-1]) <= 0:
        raise ValueError("Audio is empty and cannot be resampled.")

    target_length = max(1, int(round(audio.shape[-1] * float(target_sample_rate) / float(sample_rate))))
    resampled = torch.nn.functional.interpolate(
        audio.unsqueeze(0),
        size=target_length,
        mode="linear",
        align_corners=False,
    )
    return resampled.squeeze(0).contiguous()


def audio_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trim_audio_tensor(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    trim_start_ms: int | float | None = None,
    trim_end_ms: int | float | None = None,
    min_duration_seconds: float = MIN_REFERENCE_AUDIO_SECONDS,
    max_duration_seconds: float = MAX_REFERENCE_AUDIO_SECONDS,
) -> tuple[torch.Tensor, dict[str, Any]]:
    audio = waveform.detach().float().cpu().contiguous()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2 or int(audio.shape[-1]) <= 0:
        raise ValueError("Reference audio is empty.")

    total_samples = int(audio.shape[-1])
    source_duration_seconds = float(total_samples) / float(sample_rate)
    start_ms = max(0, int(round(float(trim_start_ms or 0))))
    requested_end_ms = int(round(float(trim_end_ms))) if trim_end_ms is not None and str(trim_end_ms) != "" else 0
    min_duration_ms = max(1, int(round(float(min_duration_seconds) * 1000.0)))
    max_duration_ms = max(1, int(round(float(max_duration_seconds) * 1000.0)))
    source_duration_ms = max(1, int(round(source_duration_seconds * 1000.0)))
    if source_duration_ms < min_duration_ms:
        raise ValueError(
            f"Reference audio must be at least {min_duration_seconds:.0f} seconds long. "
            f"Current file duration is {source_duration_seconds:.2f} seconds."
        )
    end_ms = requested_end_ms if requested_end_ms > 0 else source_duration_ms
    end_ms = min(end_ms, source_duration_ms)
    was_auto_trimmed = False
    if end_ms <= start_ms:
        raise ValueError("Trim end must be greater than trim start.")
    if end_ms - start_ms > max_duration_ms:
        end_ms = start_ms + max_duration_ms
        was_auto_trimmed = True

    start_sample = min(total_samples - 1, max(0, int(round(sample_rate * start_ms / 1000.0))))
    end_sample = min(total_samples, max(start_sample + 1, int(round(sample_rate * end_ms / 1000.0))))
    trimmed = audio[:, start_sample:end_sample].contiguous()
    if int(trimmed.shape[-1]) <= 0:
        raise ValueError("Trimmed reference audio is empty.")
    trimmed_duration_ms = int(round(float(trimmed.shape[-1]) / float(sample_rate) * 1000.0))
    if trimmed_duration_ms < min_duration_ms:
        raise ValueError(
            f"Reference selection must be at least {min_duration_seconds:.0f} seconds long. "
            f"Current selection is {trimmed_duration_ms / 1000.0:.2f} seconds."
        )

    metadata = {
        "trim_start_ms": start_ms,
        "trim_end_ms": end_ms,
        "source_duration_seconds": round(source_duration_seconds, 4),
        "duration_seconds": round(float(trimmed.shape[-1]) / float(sample_rate), 4),
        "min_duration_seconds": min_duration_seconds,
        "was_auto_trimmed": was_auto_trimmed,
        "max_duration_seconds": max_duration_seconds,
    }
    return trimmed, metadata


def prepare_reference_prompt_audio(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    tail_silence_seconds: float = REFERENCE_TAIL_SILENCE_SECONDS,
    fade_out_ms: int = REFERENCE_TAIL_FADE_OUT_MS,
) -> tuple[torch.Tensor, dict[str, Any]]:
    audio = waveform.detach().float().cpu().contiguous()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2 or int(audio.shape[-1]) <= 0:
        raise ValueError("Reference audio is empty.")

    fade_samples = min(int(audio.shape[-1]), max(0, int(round(sample_rate * fade_out_ms / 1000.0))))
    if fade_samples > 0:
        fade_curve = torch.linspace(1.0, 0.0, fade_samples, dtype=audio.dtype)
        audio[:, -fade_samples:] = audio[:, -fade_samples:] * fade_curve

    silence_samples = max(0, int(round(sample_rate * tail_silence_seconds)))
    if silence_samples > 0:
        audio = torch.nn.functional.pad(audio, (0, silence_samples), value=0.0)

    metadata = {
        "tail_silence_ms": int(round(silence_samples * 1000.0 / float(sample_rate))) if sample_rate > 0 else 0,
        "fade_out_ms": int(round(fade_samples * 1000.0 / float(sample_rate))) if sample_rate > 0 else 0,
    }
    return audio.contiguous(), metadata


def patched_decode_wav(model: Any, encoded: torch.Tensor, time_before: torch.Tensor) -> torch.Tensor:
    decoder_device = next(model.decoder.parameters()).device
    decoder_dtype = next(model.decoder.parameters()).dtype
    time_before = time_before[: encoded.shape[0] + 1].to(decoder_device)
    if time_before.shape[0] == 0:
        return torch.zeros(encoded.shape[0], 0, device=decoder_device)

    encoded = encoded.to(decoder_device, dtype=decoder_dtype)
    encoded_expanded = []
    for pos in range(encoded.shape[0]):
        encoded_expanded.append(
            torch.zeros(
                (time_before[pos] - 1).clamp(min=0),
                encoded.shape[-1],
                device=decoder_device,
                dtype=decoder_dtype,
            )
        )
        encoded_expanded.append(encoded[pos].unsqueeze(0))

    encoded_expanded.append(
        torch.zeros(time_before[-1], encoded.shape[-1], device=decoder_device, dtype=decoder_dtype)
    )

    encoded_expanded = torch.cat(encoded_expanded, dim=0).unsqueeze(0)
    token_masks = (torch.norm(encoded_expanded, dim=-1) != 0).long()
    return model.decoder.generate(encoded_expanded, token_masks=token_masks)


def _multipart_form_data(fields: dict[str, str], file_field_name: str, file_path: Path) -> tuple[bytes, str]:
    boundary = f"----TADA{uuid.uuid4().hex}"
    line_break = b"\r\n"
    parts: list[bytes] = []
    for key, value in fields.items():
        parts.extend(
            [
                f"--{boundary}".encode("utf-8"),
                f'Content-Disposition: form-data; name="{key}"'.encode("utf-8"),
                b"",
                value.encode("utf-8"),
            ]
        )

    parts.extend(
        [
            f"--{boundary}".encode("utf-8"),
            f'Content-Disposition: form-data; name="{file_field_name}"; filename="{file_path.name}"'.encode("utf-8"),
            b"Content-Type: application/octet-stream",
            b"",
            file_path.read_bytes(),
        ]
    )
    parts.append(f"--{boundary}--".encode("utf-8"))
    return line_break.join(parts) + line_break, boundary


def _safe_http_error_detail(exc: urllib_error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


@dataclass
class VoiceRecord:
    voice_id: str
    name: str
    language: str
    transcript: str
    created_at: str
    reference_audio: str
    prompt_cache: str
    sample_rate: int
    duration_seconds: float
    trim_start_ms: int = 0
    trim_end_ms: int = 0
    source_duration_seconds: float = 0.0
    was_auto_trimmed: bool = False
    tail_silence_ms: int = 0

    @classmethod
    def from_file(cls, metadata_path: Path) -> "VoiceRecord":
        payload = safe_json_load(metadata_path)
        duration_seconds = float(payload["duration_seconds"])
        return cls(
            voice_id=payload["voice_id"],
            name=payload["name"],
            language=payload["language"],
            transcript=payload["transcript"],
            created_at=payload["created_at"],
            reference_audio=payload["reference_audio"],
            prompt_cache=payload["prompt_cache"],
            sample_rate=int(payload["sample_rate"]),
            duration_seconds=duration_seconds,
            trim_start_ms=int(payload.get("trim_start_ms", 0) or 0),
            trim_end_ms=int(payload.get("trim_end_ms", round(duration_seconds * 1000.0)) or 0),
            source_duration_seconds=float(payload.get("source_duration_seconds", duration_seconds) or duration_seconds),
            was_auto_trimmed=bool(payload.get("was_auto_trimmed", False)),
            tail_silence_ms=int(payload.get("tail_silence_ms", 0) or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "voice_id": self.voice_id,
            "name": self.name,
            "language": self.language,
            "transcript": self.transcript,
            "created_at": self.created_at,
            "reference_audio": self.reference_audio,
            "prompt_cache": self.prompt_cache,
            "sample_rate": self.sample_rate,
            "duration_seconds": self.duration_seconds,
            "trim_start_ms": self.trim_start_ms,
            "trim_end_ms": self.trim_end_ms,
            "source_duration_seconds": self.source_duration_seconds,
            "was_auto_trimmed": self.was_auto_trimmed,
            "tail_silence_ms": self.tail_silence_ms,
        }


@dataclass
class BatchGenerationItem:
    request_id: str
    voice_id: str
    text: str
    sentence_index: int
    derived_seed: int | None = None


@dataclass
class BatchGenerationResult:
    request_id: str
    voice_id: str
    text: str
    sentence_index: int
    waveform: torch.Tensor
    sample_rate: int
    duration_seconds: float


@dataclass
class BatchPreviewUpdate:
    request_id: str
    voice_id: str
    text: str
    sentence_index: int
    waveform_delta: torch.Tensor
    sample_rate: int
    progress_step: int


@dataclass
class BenchmarkFixtureRecord:
    fixture_id: str
    source_name: str
    source_path: str
    source_sha256: str
    source_duration_seconds: float
    prepared_duration_seconds: float | None
    sample_rate: int | None
    channels: int | None
    transcript: str | None
    language: str | None
    output_path: str | None
    skipped: bool
    skip_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "source_name": self.source_name,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "source_duration_seconds": self.source_duration_seconds,
            "prepared_duration_seconds": self.prepared_duration_seconds,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "transcript": self.transcript,
            "language": self.language,
            "output_path": self.output_path,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


@dataclass
class _PreviewState:
    item: BatchGenerationItem
    emitted_samples: int = 0


class TadaRuntimeService:
    def __init__(self, project_root: Path, config_store: ConfigStore):
        self.project_root = project_root
        self.config_store = config_store
        self.backend_dir = project_root / "backend"
        self.data_dir = ensure_directory(self.backend_dir / "data")
        self.voices_dir = ensure_directory(self.data_dir / "voices")
        self.generated_history_path = self.data_dir / "generated_history.jsonl"
        self.generated_dir = ensure_directory(self.backend_dir / "generated")
        self.offload_dir = ensure_directory(self.backend_dir / "offload")
        self._generated_audio_cache: OrderedDict[str, bytes] = OrderedDict()
        self._generated_audio_cache_lock = threading.Lock()
        self.codec_name = os.getenv("TADA_CODEC_NAME", "HumeAI/tada-codec")
        self._encoders: dict[str, Any] = {}
        self._model: Any | None = None
        self._model_name: str | None = None
        self._model_lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self._model_load_error: str | None = None
        self._download_lock = threading.Lock()
        self._download_jobs: dict[tuple[str, str], dict[str, Any]] = {}

        if torch.cuda.is_available():
            self.device = torch.device(os.getenv("TADA_DEVICE", "cuda:0"))
            total_memory = torch.cuda.get_device_properties(0).total_memory
            self.gpu_memory_gib = round(total_memory / (1024**3), 2)
            self.gpu_name = torch.cuda.get_device_name(0)
        else:
            self.device = torch.device("cpu")
            self.gpu_memory_gib = 0.0
            self.gpu_name = None

        self.encoder_device = torch.device(
            os.getenv(
                "TADA_ENCODER_DEVICE",
                "cpu" if self.device.type == "cuda" and self.gpu_memory_gib < 10 else str(self.device),
            )
        )
        self.prompt_device = torch.device("cuda:0" if self.device.type == "cuda" else "cpu")
        initial_settings = self._settings()
        self.configured_model_precision = initial_settings.model_precision
        self.model_precision, self.model_dtype = self._resolve_model_precision(initial_settings.model_precision)
        self._apply_env_vars(initial_settings)
        self.enable_cpu_offload = os.getenv("TADA_ENABLE_CPU_OFFLOAD", "0").lower() in {"1", "true", "yes"}
        self.decoder_device = torch.device(
            os.getenv("TADA_DECODER_DEVICE", "cpu" if self.enable_cpu_offload else str(self.device))
        )

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        self.enable_torch_compile = (
            self.device.type == "cuda"
            and not self.enable_cpu_offload
            and importlib.util.find_spec("triton") is not None
            and os.getenv("TADA_DISABLE_TORCH_COMPILE", "0").lower() not in {"1", "true", "yes"}
        )

    def _settings(self):
        return self.config_store.get_settings()

    def _resolve_model_precision(self, configured_precision: str) -> tuple[str, torch.dtype]:
        normalized = str(configured_precision or "fp16").strip().lower() or "fp16"
        if normalized not in SUPPORTED_MODEL_PRECISIONS:
            raise ValueError(
                f"Unsupported model precision '{configured_precision}'. Available: {', '.join(SUPPORTED_MODEL_PRECISIONS)}."
            )
        if self.device.type != "cuda":
            return "fp32", torch.float32
        if normalized == "bf16":
            if not torch.cuda.is_bf16_supported():
                raise ValueError("BF16 was requested but this GPU/runtime does not support bfloat16.")
            return "bf16", torch.bfloat16
        if normalized == "fp16":
            return "fp16", torch.float16
        return "fp32", torch.float32

    def _hf_token(self) -> str:
        return self.config_store.get_hf_token() or get_token() or ""

    def _hf_kwargs(self) -> dict[str, Any]:
        token = self._hf_token()
        return {"token": token} if token else {}

    def _cache_roots(self) -> list[Path]:
        settings = self._settings()
        return [Path(settings.model_storage_path).expanduser()]

    def status(self) -> dict[str, Any]:
        settings = self._settings()
        voices = self.list_voices()
        return {
            "ok": TADA_IMPORT_ERROR is None,
            "cuda_available": torch.cuda.is_available(),
            "device": str(self.device),
            "encoder_device": str(self.encoder_device),
            "decoder_device": str(self.decoder_device),
            "gpu_name": self.gpu_name,
            "gpu_memory_gib": self.gpu_memory_gib,
            "configured_model_name": settings.active_model,
            "configured_model_precision": settings.model_precision,
            "effective_model_precision": self.model_precision,
            "loaded_model_name": self._model_name,
            "available_models": self.available_models(),
            "model_loaded": self._model is not None,
            "cpu_offload": self.enable_cpu_offload,
            "dtype": str(self.model_dtype).replace("torch.", ""),
            "default_flow_steps": DEFAULT_FLOW_STEPS,
            "torch_compile_enabled": self.enable_torch_compile,
            "voices_count": len(voices),
            "languages": SUPPORTED_LANGUAGES,
            "hf_token_present": bool(self._hf_token()),
            "import_error": str(TADA_IMPORT_ERROR) if TADA_IMPORT_ERROR else None,
            "model_error": self._model_load_error,
        }

    def list_voices(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for metadata_path in self.voices_dir.glob("*/metadata.json"):
            record = VoiceRecord.from_file(metadata_path)
            payload = record.to_dict()
            payload["reference_url"] = f"/api/assets/voices/{record.voice_id}/reference"
            records.append(payload)
        return sorted(records, key=lambda item: item["created_at"], reverse=True)

    def get_voice(self, voice_id: str) -> dict[str, Any]:
        metadata_path = self.voices_dir / voice_id / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Voice '{voice_id}' was not found.")
        record = VoiceRecord.from_file(metadata_path)
        payload = record.to_dict()
        payload["reference_url"] = f"/api/assets/voices/{record.voice_id}/reference"
        return payload

    def _append_generation_history(self, payload: dict[str, Any]) -> None:
        self.generated_history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.generated_history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _has_cached_generated_audio(self, file_name: str) -> bool:
        with self._generated_audio_cache_lock:
            return file_name in self._generated_audio_cache

    def _store_generated_audio_in_memory(self, file_name: str, audio: torch.Tensor, sample_rate: int) -> None:
        buffer = io.BytesIO()
        clip = audio.detach().float().cpu()
        if clip.ndim == 1:
            clip = clip.unsqueeze(0)
        sf.write(buffer, clip.transpose(0, 1).numpy(), int(sample_rate), format="WAV")
        with self._generated_audio_cache_lock:
            self._generated_audio_cache[file_name] = buffer.getvalue()
            self._generated_audio_cache.move_to_end(file_name)
            while len(self._generated_audio_cache) > GENERATED_AUDIO_MEMORY_CACHE_LIMIT:
                self._generated_audio_cache.popitem(last=False)

    def _hydrate_generation_record(self, record: dict[str, Any]) -> dict[str, Any]:
        hydrated = dict(record)
        file_name = str(hydrated.get("audio_file_name") or "").strip()
        audio_storage = str(hydrated.get("audio_storage") or ("disk" if file_name else "")).strip().lower()
        audio_url = None
        if file_name:
            if audio_storage == "disk":
                if (self.generated_dir / file_name).exists():
                    audio_url = f"/api/assets/generated/{file_name}"
            elif audio_storage == "memory":
                if self._has_cached_generated_audio(file_name):
                    audio_url = f"/api/assets/generated/{file_name}"
        hydrated["audio_storage"] = audio_storage or None
        hydrated["audio_url"] = audio_url
        return hydrated

    def list_recent_generations(self, *, limit: int = 60) -> list[dict[str, Any]]:
        if not self.generated_history_path.exists():
            return []
        lines = self.generated_history_path.read_text(encoding="utf-8").splitlines()
        records: list[dict[str, Any]] = []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(self._hydrate_generation_record(record))
            if len(records) >= limit:
                break
        return records

    def reference_audio_path(self, voice_id: str) -> Path:
        voice = self.get_voice(voice_id)
        return self.voices_dir / voice_id / voice["reference_audio"]

    def delete_voice(self, voice_id: str) -> None:
        voice_dir = self.voices_dir / voice_id
        if not voice_dir.exists():
            raise FileNotFoundError(f"Voice '{voice_id}' was not found.")
        shutil.rmtree(voice_dir, ignore_errors=False)

    def available_models(self) -> list[dict[str, str]]:
        return [
            {"id": model_name, "label": metadata["label"], "description": metadata["description"]}
            for model_name, metadata in AVAILABLE_MODELS.items()
        ]

    @staticmethod
    def normalize_model_name(model_name: str) -> str:
        candidate = model_name.strip()
        if not candidate:
            raise ValueError("Model name must not be empty.")
        normalized = MODEL_NAME_ALIASES.get(candidate.lower(), candidate)
        if normalized not in AVAILABLE_MODELS:
            supported = ", ".join(AVAILABLE_MODELS.keys())
            raise ValueError(f"Unsupported model '{model_name}'. Available: {supported}.")
        return normalized

    def _dispose_model(self, model: Any | None) -> None:
        if model is not None:
            del model
        gc.collect()
        if self.device.type == "cuda":
            try:
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
            except RuntimeError as exc:
                print(f"Warning: CUDA cache cleanup failed during model disposal: {exc}", flush=True)

    def _apply_env_vars(self, settings) -> None:
        if settings.model_storage_path:
            os.environ["TADA_MODEL_STORAGE_PATH"] = str(Path(settings.model_storage_path).expanduser().resolve(strict=False))
        elif "TADA_MODEL_STORAGE_PATH" in os.environ:
            del os.environ["TADA_MODEL_STORAGE_PATH"]
            
        token = self._hf_token()
        if token:
            os.environ["HF_TOKEN"] = token
        elif "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]

    def sync_runtime_settings(self) -> None:
        settings = self._settings()
        self._apply_env_vars(settings)
        normalized_model_name = self.normalize_model_name(settings.active_model)
        desired_precision, desired_dtype = self._resolve_model_precision(settings.model_precision)
        model_to_dispose: Any | None = None
        with self._generation_lock:
            with self._model_lock:
                precision_changed = (
                    desired_dtype != self.model_dtype
                    or desired_precision != self.model_precision
                    or settings.model_precision != self.configured_model_precision
                )
                if normalized_model_name != self._model_name or precision_changed:
                    model_to_dispose = self._model
                    self._model = None
                    self._model_name = normalized_model_name
                    self._model_load_error = None
                self.configured_model_precision = settings.model_precision
                self.model_precision = desired_precision
                self.model_dtype = desired_dtype
            self._dispose_model(model_to_dispose)

    def _ensure_runtime_ready(self) -> None:
        if TADA_IMPORT_ERROR is not None:
            raise RuntimeError(
                "TADA is not installed yet. Run install.bat first so the local vendor packages are available."
            ) from TADA_IMPORT_ERROR

    def _load_model(self) -> Any:
        self._ensure_runtime_ready()
        settings = self._settings()
        desired_model_name = self.normalize_model_name(settings.active_model)
        if self._model is not None and self._model_name == desired_model_name:
            return self._model

        with self._model_lock:
            if self._model is not None and self._model_name == desired_model_name:
                return self._model
            try:
                model_to_dispose = self._model
                self._model = None
                self._model_name = desired_model_name
                model_source, model_is_local = resolve_pretrained_source(
                    desired_model_name,
                    self.project_root,
                    extra_roots=self._cache_roots(),
                )
                load_kwargs: dict[str, Any] = {
                    "dtype": self.model_dtype,
                    "low_cpu_mem_usage": True,
                    **self._hf_kwargs(),
                }
                if model_is_local:
                    load_kwargs["local_files_only"] = True
                if self.device.type == "cuda" and self.enable_cpu_offload:
                    gpu_budget = max(4, int(self.gpu_memory_gib - 1))
                    load_kwargs.update(
                        {
                            "device_map": "auto",
                            "max_memory": {0: f"{gpu_budget}GiB", "cpu": "48GiB"},
                            "offload_folder": str(self.offload_dir),
                        }
                    )
                self._dispose_model(model_to_dispose)

                model = TadaForCausalLM.from_pretrained(model_source, **load_kwargs)
                if "device_map" not in load_kwargs:
                    model = model.to(self.device)

                if self.enable_torch_compile:
                    try:
                        import torch._dynamo

                        torch._dynamo.config.suppress_errors = True
                        model.prediction_head = torch.compile(model.prediction_head, mode="reduce-overhead")
                        print("TADA prediction head compiled with torch.compile.", flush=True)
                    except Exception as exc:
                        print(f"Warning: torch.compile warmup failed: {exc}", flush=True)

                model.decoder.to(self.decoder_device)
                model._decode_wav = types.MethodType(patched_decode_wav, model)
                model.eval()
                self._model = model
                self._model_load_error = None
            except Exception as exc:
                self._model_load_error = self._friendly_error(exc)
                raise RuntimeError(self._model_load_error) from exc
        return self._model

    def _load_encoder(self, language: str) -> Any:
        self._ensure_runtime_ready()
        normalized_language = language if language != "en" else ""
        cache_key = normalized_language or "en"
        if cache_key in self._encoders:
            return self._encoders[cache_key]

        with self._model_lock:
            if cache_key in self._encoders:
                return self._encoders[cache_key]

            if not self._hf_token():
                raise RuntimeError(
                    "Voice cloning needs access to the gated Hugging Face tokenizer `meta-llama/Llama-3.2-1B`. "
                    "Please add an HF token in the admin settings or run `huggingface-cli login`."
                )

            kwargs = self._hf_kwargs()
            encoder_source, encoder_is_local = resolve_pretrained_source(
                self.codec_name,
                self.project_root,
                extra_roots=self._cache_roots(),
            )
            if encoder_is_local:
                kwargs["local_files_only"] = True
            if normalized_language:
                kwargs["language"] = normalized_language
            try:
                encoder = Encoder.from_pretrained(encoder_source, subfolder="encoder", **kwargs)
                encoder = encoder.to(self.encoder_device)
                encoder.eval()
            except Exception as exc:
                raise RuntimeError(self._friendly_encoder_error(exc, normalized_language or "en")) from exc
            self._encoders[cache_key] = encoder
            return encoder

    def create_voice(
        self,
        *,
        name: str,
        language: str,
        transcript: str,
        upload_path: Path,
        trim_start_ms: int | float | None = None,
        trim_end_ms: int | float | None = None,
        overwrite_existing: bool = False,
    ) -> dict[str, Any]:
        normalized_language = language.strip().lower() or "en"
        if normalized_language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language '{normalized_language}'.")
        if normalized_language != "en" and not transcript.strip():
            raise ValueError("For non-English voices you should provide the transcript of the reference audio.")
        cleaned_name = " ".join((name or "").strip().split()) or "Meine Stimme"
        normalized_name = normalize_voice_name(cleaned_name)
        normalized_slug = slugify(cleaned_name)
        existing_voice = None
        for existing in self.list_voices():
            existing_name = str(existing.get("name") or "")
            if normalize_voice_name(existing_name) == normalized_name or slugify(existing_name) == normalized_slug:
                existing_voice = existing
                break

        if existing_voice is not None and not overwrite_existing:
            raise DuplicateVoiceNameError(
                cleaned_name=cleaned_name,
                existing_voice_id=str(existing_voice.get("voice_id") or ""),
                existing_voice_name=str(existing_voice.get("name") or cleaned_name),
            )

        voice_id = str(existing_voice.get("voice_id")) if existing_voice is not None else f"{normalized_slug}-{uuid.uuid4().hex[:8]}"
        voice_dir = self.voices_dir / voice_id
        if existing_voice is not None:
            shutil.rmtree(voice_dir, ignore_errors=True)
        voice_dir.mkdir(parents=True, exist_ok=True)

        reference_path = voice_dir / "reference.wav"
        prompt_cache_path = voice_dir / "prompt_cache.pt"
        try:
            waveform, sample_rate = load_audio_tensor(upload_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.clamp(-1.0, 1.0)
            waveform, trim_metadata = trim_audio_tensor(
                waveform,
                sample_rate=sample_rate,
                trim_start_ms=trim_start_ms,
                trim_end_ms=trim_end_ms,
            )
            waveform, prompt_audio_metadata = prepare_reference_prompt_audio(
                waveform,
                sample_rate=sample_rate,
            )
            save_audio_tensor(reference_path, waveform, sample_rate)

            encoder = self._load_encoder(normalized_language)
            encoder_kwargs: dict[str, Any] = {"sample_rate": sample_rate}
            if transcript.strip():
                encoder_kwargs["text"] = [transcript.strip()]

            prompt = encoder(waveform.to(self.encoder_device), **encoder_kwargs)
            prompt.save(str(prompt_cache_path))
        except Exception:
            shutil.rmtree(voice_dir, ignore_errors=True)
            raise

        duration_seconds = round(float(waveform.shape[-1]) / float(sample_rate), 2)
        record = VoiceRecord(
            voice_id=voice_id,
            name=cleaned_name,
            language=normalized_language,
            transcript=transcript.strip(),
            created_at=now_iso(),
            reference_audio=reference_path.name,
            prompt_cache=prompt_cache_path.name,
            sample_rate=int(sample_rate),
            duration_seconds=duration_seconds,
            trim_start_ms=int(trim_metadata["trim_start_ms"]),
            trim_end_ms=int(trim_metadata["trim_end_ms"]),
            source_duration_seconds=float(trim_metadata["source_duration_seconds"]),
            was_auto_trimmed=bool(trim_metadata["was_auto_trimmed"]),
            tail_silence_ms=int(prompt_audio_metadata["tail_silence_ms"]),
        )
        safe_json_dump(voice_dir / "metadata.json", record.to_dict())
        return self.get_voice(voice_id)

    def transcribe_audio(
        self,
        *,
        upload_path: Path,
        trim_start_ms: int | float | None = None,
        trim_end_ms: int | float | None = None,
    ) -> dict[str, Any]:
        settings = self._settings()
        base_url = settings.whisper_base_url.strip().rstrip("/")
        if not base_url:
            raise RuntimeError("Whisper base URL is not configured in the admin settings.")

        waveform, sample_rate = load_audio_tensor(upload_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.clamp(-1.0, 1.0)
        waveform, trim_metadata = trim_audio_tensor(
            waveform,
            sample_rate=sample_rate,
            trim_start_ms=trim_start_ms,
            trim_end_ms=trim_end_ms,
        )

        temp_path = self.project_root / ".tmp" / f"whisper-trim-{uuid.uuid4().hex}.wav"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        save_audio_tensor(temp_path, waveform, sample_rate)
        try:
            payload = None
            final_error: RuntimeError | None = None
            for endpoint, form_fields in self._whisper_request_candidates(base_url):
                form_body, boundary = _multipart_form_data(form_fields, "file", temp_path)
                headers = {
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "Accept": "application/json",
                }
                whisper_api_key = self.config_store.get_whisper_api_key()
                if whisper_api_key:
                    headers["Authorization"] = f"Bearer {whisper_api_key}"

                request = urllib_request.Request(endpoint, data=form_body, headers=headers, method="POST")
                try:
                    with urllib_request.urlopen(request, timeout=120) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                    break
                except urllib_error.HTTPError as exc:
                    detail = _safe_http_error_detail(exc)
                    final_error = RuntimeError(
                        f"Whisper transcription failed at {endpoint} ({exc.code}): {detail or exc.reason}"
                    )
                    if exc.code in {404, 405}:
                        continue
                    raise final_error from exc
                except urllib_error.URLError as exc:
                    raise RuntimeError(f"Whisper transcription failed at {endpoint}: {exc.reason}") from exc

            if payload is None:
                raise final_error or RuntimeError("Whisper transcription failed: no compatible endpoint responded.")
        finally:
            temp_path.unlink(missing_ok=True)

        transcript = str(payload.get("text") or payload.get("transcription") or "").strip()
        if not transcript:
            raise RuntimeError("Whisper returned no transcript text.")
        return {
            "text": transcript,
            "language": str(payload.get("language") or "").strip() or None,
            "base_url": base_url,
            "trim_start_ms": int(trim_metadata["trim_start_ms"]),
            "trim_end_ms": int(trim_metadata["trim_end_ms"]),
            "duration_seconds": float(trim_metadata["duration_seconds"]),
            "source_duration_seconds": float(trim_metadata["source_duration_seconds"]),
            "was_auto_trimmed": bool(trim_metadata["was_auto_trimmed"]),
        }

    @staticmethod
    def _whisper_request_candidates(base_url: str) -> list[tuple[str, dict[str, str]]]:
        normalized = str(base_url or "").strip().rstrip("/")
        base_without_v1 = normalized[:-3].rstrip("/") if normalized.endswith("/v1") else normalized
        candidates: list[tuple[str, dict[str, str]]] = []
        seen: set[str] = set()

        def add(endpoint: str, fields: dict[str, str]) -> None:
            if endpoint and endpoint not in seen:
                candidates.append((endpoint, fields))
                seen.add(endpoint)

        openai_fields = {"model": "whisper-1"}
        genesis_fields = {"engine": "local", "voice_ident": "false"}

        add(f"{normalized}/audio/transcriptions", openai_fields)
        add(f"{base_without_v1}/audio/transcriptions", openai_fields)
        add(f"{base_without_v1}/v1/audio/transcriptions", openai_fields)
        add(f"{normalized}/transcribe/", genesis_fields)
        add(f"{base_without_v1}/transcribe/", genesis_fields)
        return candidates

    @staticmethod
    def _normalize_seed_text(text: str) -> str:
        return " ".join((text or "").split())

    def _derive_batch_item_seed(self, item: BatchGenerationItem, settings: Any) -> int | None:
        if settings.deterministic_seed is None:
            return None
        material = "\x1f".join(
            [
                str(settings.deterministic_seed),
                self.normalize_model_name(settings.active_model),
                str(settings.model_precision),
                str(item.voice_id),
                self._normalize_seed_text(item.text),
            ]
        )
        digest = hashlib.sha256(material.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    def _prepare_batch_items(self, items: list[BatchGenerationItem], settings: Any) -> list[BatchGenerationItem]:
        prepared_items: list[BatchGenerationItem] = []
        for item in items:
            prepared_items.append(
                replace(
                    item,
                    derived_seed=item.derived_seed
                    if item.derived_seed is not None
                    else self._derive_batch_item_seed(item, settings),
                )
            )
        return prepared_items

    def build_benchmark_corpus(
        self,
        *,
        source_dir: Path | None = None,
        output_dir: Path | None = None,
        mode: str = "full",
    ) -> dict[str, Any]:
        normalized_mode = str(mode or "full").strip().lower() or "full"
        if normalized_mode not in {"smoke", "full"}:
            raise ValueError("Benchmark corpus mode must be 'smoke' or 'full'.")

        source_root = (source_dir or (self.project_root / "voices")).resolve(strict=False)
        if not source_root.exists():
            raise FileNotFoundError(f"Benchmark source directory '{source_root}' was not found.")

        fixtures_root = (output_dir or (self.project_root / BENCHMARK_FIXTURES_DIR)).resolve(strict=False)
        fixtures_root.mkdir(parents=True, exist_ok=True)
        manifest_path = fixtures_root / "manifest.json"
        cached_manifest = safe_json_load(manifest_path) if manifest_path.exists() else {}
        cached_by_hash: dict[str, dict[str, Any]] = {}
        for entry in cached_manifest.get("entries", []):
            source_hash = str(entry.get("source_sha256") or "")
            if source_hash:
                cached_by_hash[source_hash] = entry

        discovered: list[dict[str, Any]] = []
        selected_candidates: list[dict[str, Any]] = []
        for source_path in sorted(source_root.iterdir(), key=lambda item: item.name.lower()):
            if not source_path.is_file():
                continue
            source_hash = audio_file_sha256(source_path)
            try:
                waveform, sample_rate = load_audio_tensor(source_path)
            except Exception as exc:
                discovered.append(
                    BenchmarkFixtureRecord(
                        fixture_id=source_path.stem,
                        source_name=source_path.name,
                        source_path=str(source_path.relative_to(self.project_root)),
                        source_sha256=source_hash,
                        source_duration_seconds=0.0,
                        prepared_duration_seconds=None,
                        sample_rate=None,
                        channels=None,
                        transcript=None,
                        language=None,
                        output_path=None,
                        skipped=True,
                        skip_reason=f"decode_error: {exc}",
                    ).to_dict()
                )
                continue

            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            source_duration_seconds = round(float(waveform.shape[-1]) / float(sample_rate), 4)
            channels = int(waveform.shape[0])
            base_record = {
                "fixture_id": source_path.stem,
                "source_name": source_path.name,
                "source_path": str(source_path.relative_to(self.project_root)),
                "source_sha256": source_hash,
                "source_duration_seconds": source_duration_seconds,
                "sample_rate": int(sample_rate),
                "channels": channels,
                "waveform": waveform,
            }
            if source_duration_seconds < MIN_REFERENCE_AUDIO_SECONDS:
                discovered.append(
                    BenchmarkFixtureRecord(
                        prepared_duration_seconds=None,
                        transcript=None,
                        language=None,
                        output_path=None,
                        skipped=True,
                        skip_reason="duration_below_minimum",
                        **{key: base_record[key] for key in ("fixture_id", "source_name", "source_path", "source_sha256", "source_duration_seconds", "sample_rate", "channels")},
                    ).to_dict()
                )
                continue
            if source_duration_seconds >= BENCHMARK_MAX_SOURCE_SECONDS:
                discovered.append(
                    BenchmarkFixtureRecord(
                        prepared_duration_seconds=None,
                        transcript=None,
                        language=None,
                        output_path=None,
                        skipped=True,
                        skip_reason="duration_at_or_above_limit",
                        **{key: base_record[key] for key in ("fixture_id", "source_name", "source_path", "source_sha256", "source_duration_seconds", "sample_rate", "channels")},
                    ).to_dict()
                )
                continue
            selected_candidates.append(base_record)

        if normalized_mode == "smoke":
            selected_candidates = selected_candidates[:10]

        selected_hashes = {candidate["source_sha256"] for candidate in selected_candidates}
        entries: list[dict[str, Any]] = [entry for entry in discovered if entry["source_sha256"] not in selected_hashes]

        for candidate in selected_candidates:
            source_path = source_root / candidate["source_name"]
            waveform = candidate["waveform"]
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.clamp(-1.0, 1.0)
            trimmed, _ = trim_audio_tensor(
                waveform,
                sample_rate=int(candidate["sample_rate"]),
                min_duration_seconds=MIN_REFERENCE_AUDIO_SECONDS,
                max_duration_seconds=BENCHMARK_MAX_SOURCE_SECONDS,
            )
            resampled = resample_audio_tensor(trimmed, sample_rate=int(candidate["sample_rate"]), target_sample_rate=24000)
            prepared, _ = prepare_reference_prompt_audio(resampled, sample_rate=24000)
            prepared_duration_seconds = round(float(prepared.shape[-1]) / 24000.0, 4)
            fixture_file_name = f"{slugify(Path(candidate['fixture_id']).stem)}-{candidate['source_sha256'][:8]}.wav"
            fixture_path = fixtures_root / fixture_file_name
            save_audio_tensor(fixture_path, prepared, 24000)

            cached_entry = cached_by_hash.get(candidate["source_sha256"], {})
            transcript = str(cached_entry.get("transcript") or "").strip() or None
            language = str(cached_entry.get("language") or "").strip() or None
            skipped = False
            skip_reason = None

            if not transcript:
                try:
                    transcript_payload = self.transcribe_audio(upload_path=source_path)
                    transcript = str(transcript_payload.get("text") or "").strip() or None
                    language = str(transcript_payload.get("language") or "").strip() or None
                except Exception as exc:
                    skipped = True
                    skip_reason = f"transcription_failed: {exc}"

            if not transcript:
                skipped = True
                skip_reason = skip_reason or "empty_transcript"

            entries.append(
                BenchmarkFixtureRecord(
                    fixture_id=candidate["fixture_id"],
                    source_name=candidate["source_name"],
                    source_path=candidate["source_path"],
                    source_sha256=candidate["source_sha256"],
                    source_duration_seconds=float(candidate["source_duration_seconds"]),
                    prepared_duration_seconds=prepared_duration_seconds,
                    sample_rate=24000,
                    channels=1,
                    transcript=transcript,
                    language=language,
                    output_path=str(fixture_path.relative_to(self.project_root)),
                    skipped=skipped,
                    skip_reason=skip_reason,
                ).to_dict()
            )

        entries.sort(key=lambda item: item["source_name"].lower())
        manifest = {
            "generated_at": now_iso(),
            "mode": normalized_mode,
            "source_dir": str(source_root.relative_to(self.project_root)),
            "output_dir": str(fixtures_root.relative_to(self.project_root)),
            "entries": entries,
        }
        safe_json_dump(manifest_path, manifest)
        prepared_entries = [entry for entry in entries if not entry["skipped"]]
        skipped_entries = [entry for entry in entries if entry["skipped"]]
        return {
            "manifest_path": str(manifest_path),
            "entries": entries,
            "prepared_count": len(prepared_entries),
            "skipped_count": len(skipped_entries),
            "mode": normalized_mode,
        }

    def _prompt_path_for_voice(self, voice_id: str) -> Path:
        voice = self.get_voice(voice_id)
        return self.voices_dir / voice_id / voice["prompt_cache"]

    def generate_batch(self, items: list[BatchGenerationItem]) -> list[BatchGenerationResult]:
        return self.generate_batch_progressive(items, on_preview=None)

    def generate_batch_progressive(
        self,
        items: list[BatchGenerationItem],
        *,
        on_preview: Any | None = None,
    ) -> list[BatchGenerationResult]:
        if not items:
            return []

        settings = self._settings()
        voice_ids = {item.voice_id for item in items}
        if len(voice_ids) != 1:
            raise RuntimeError(
                "Mixed-voice batching is disabled in the hardened runtime. Please batch only a single voice_id at a time."
            )

        prepared_items = self._prepare_batch_items(items, settings)
        prompts = []
        preview_states = {
            (item.request_id, item.sentence_index): _PreviewState(item=item) for item in prepared_items
        }
        with self._generation_lock:
            model = self._load_model()
            for item in prepared_items:
                prompt_path = self._prompt_path_for_voice(item.voice_id)
                prompts.append(EncoderOutput.load(str(prompt_path), device=str(self.prompt_device)))
            merged_prompt = merge_encoder_outputs(prompts)
            options = InferenceOptions(
                num_flow_matching_steps=int(settings.steps),
                item_seeds=[item.derived_seed for item in prepared_items],
                vad_trimming=bool(settings.vad_trimming),
                prompt_start_trim_steps=int(settings.prompt_start_trim_steps),
                vad_threshold_pct=float(settings.vad_threshold_pct),
                vad_padding_ms=int(settings.vad_padding_ms),
                vad_fade_ms=int(settings.vad_fade_ms),
            )
            sample_rate = self._output_sample_rate(model)
            buffer_samples = max(0, int(sample_rate * settings.stream_start_buffer_ms / 1000))
            min_emit_samples = max(1, int(sample_rate * 0.2))

            def progress_callback(update: dict[str, Any]) -> None:
                if on_preview is None or bool(update.get("is_final")):
                    return
                encoded_batch = update.get("encoded")
                time_before_batch = update.get("time_before")
                input_lengths = update.get("input_lengths")
                stream_offset = int(update.get("stream_offset", 0) or 0)
                if not isinstance(encoded_batch, torch.Tensor) or not isinstance(time_before_batch, torch.Tensor):
                    return
                if encoded_batch.ndim < 3 or time_before_batch.ndim < 2:
                    return

                batch_size = min(int(encoded_batch.shape[0]), int(time_before_batch.shape[0]), len(prepared_items))
                if isinstance(input_lengths, torch.Tensor) and input_lengths.ndim >= 1:
                    batch_size = min(batch_size, int(input_lengths.shape[0]))
                progress_step = int(update.get("step", -1))
                for batch_index in range(batch_size):
                    preview_item = prepared_items[batch_index]
                    preview_state = preview_states[(preview_item.request_id, preview_item.sentence_index)]
                    input_length = None
                    if isinstance(input_lengths, torch.Tensor):
                        input_length = int(input_lengths[batch_index].item())
                    delta = self._decode_preview_delta(
                        model=model,
                        encoded=encoded_batch[batch_index],
                        time_before=time_before_batch[batch_index],
                        sample_rate=sample_rate,
                        emitted_samples=preview_state.emitted_samples,
                        buffer_samples=buffer_samples,
                        min_emit_samples=min_emit_samples,
                        input_length=input_length,
                        stream_offset=stream_offset,
                    )
                    if delta is None or int(delta.numel()) == 0:
                        continue
                    preview_state.emitted_samples += int(delta.numel())
                    on_preview(
                        BatchPreviewUpdate(
                            request_id=preview_state.item.request_id,
                            voice_id=preview_state.item.voice_id,
                            text=preview_state.item.text,
                            sentence_index=preview_state.item.sentence_index,
                            waveform_delta=delta,
                            sample_rate=sample_rate,
                            progress_step=progress_step,
                        )
                    )

            with torch.inference_mode():
                output = model.generate(
                    prompt=merged_prompt,
                    text=[item.text for item in prepared_items],
                    inference_options=options,
                    progress_callback=progress_callback if on_preview is not None else None,
                    progress_interval_steps=DEFAULT_PROGRESS_INTERVAL_STEPS,
                )

        if not output.audio:
            raise RuntimeError("TADA returned no waveform batch.")

        results: list[BatchGenerationResult] = []
        for index, item in enumerate(prepared_items):
            audio = output.audio[index]
            if audio is None:
                raise RuntimeError(f"TADA returned no waveform for request {item.request_id}.")
            clip = audio.detach().float().cpu()
            if clip.ndim == 1:
                clip = clip.unsqueeze(0)
            duration_seconds = round(float(clip.shape[-1]) / float(sample_rate), 4)
            results.append(
                BatchGenerationResult(
                    request_id=item.request_id,
                    voice_id=item.voice_id,
                    text=item.text,
                    sentence_index=item.sentence_index,
                    waveform=clip,
                    sample_rate=sample_rate,
                    duration_seconds=duration_seconds,
                )
            )
        return results

    @staticmethod
    def _decode_preview_delta(
        *,
        model: Any,
        encoded: torch.Tensor,
        time_before: torch.Tensor,
        sample_rate: int,
        emitted_samples: int,
        buffer_samples: int,
        min_emit_samples: int,
        input_length: int | None,
        stream_offset: int,
    ) -> torch.Tensor | None:
        if encoded.ndim != 2 or encoded.shape[0] <= 0 or time_before.ndim != 1 or time_before.shape[0] <= 1:
            return None
        try:
            # Progress updates already contain only the currently generated window.
            # Re-applying an input-length-based cap here can prematurely cut previews
            # and leave the sentence tail missing until the final chunk lands.
            wav = model._decode_wav(encoded, time_before=time_before).squeeze(0, 1)
        except Exception:
            return None

        lead_samples = int(sample_rate * float(time_before[0].item()) / 50.0) if int(time_before.shape[0]) > 0 else 0
        if lead_samples > 0:
            wav = wav[..., lead_samples:]
        audio = wav.detach().float().cpu().reshape(-1)
        fade_len = min(1200, int(audio.numel()))
        if fade_len > 0:
            fade_curve = torch.linspace(0.0, 1.0, fade_len, dtype=audio.dtype)
            audio[:fade_len] = audio[:fade_len] * fade_curve
        total_samples = int(audio.numel())
        if total_samples <= emitted_samples:
            return None

        safe_end = max(emitted_samples, total_samples - buffer_samples)
        if safe_end <= emitted_samples:
            return None
        if safe_end - emitted_samples < min_emit_samples:
            return None
        return audio[emitted_samples:safe_end]

    def save_request_audio(
        self,
        *,
        audio: torch.Tensor,
        sample_rate: int,
        request_text: str,
        voice_id: str,
        ttft_ms: float | None,
        total_wall_ms: float,
        sentence_count: int,
        batch_count: int,
    ) -> dict[str, Any]:
        settings = self._settings()
        generation_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        asset_name = f"{generation_id}.wav"
        audio_path = self.generated_dir / asset_name
        audio_storage = "disk" if settings.persist_generated_wavs else "memory"
        if settings.persist_generated_wavs:
            save_audio_tensor(audio_path, audio, sample_rate)
        else:
            self._store_generated_audio_in_memory(asset_name, audio, sample_rate)
        duration_seconds = round(float(audio.shape[-1]) / float(sample_rate), 2)
        total_wall_seconds = round(total_wall_ms / 1000.0, 2)
        rtf = round(total_wall_seconds / duration_seconds, 2) if duration_seconds > 0 else 0.0
        voice_name = voice_id
        try:
            voice_name = str(self.get_voice(voice_id).get("name") or voice_id)
        except FileNotFoundError:
            pass

        result = {
            "generation_id": generation_id,
            "text": request_text,
            "voice_id": voice_id,
            "voice_name": voice_name,
            "audio_url": f"/api/assets/generated/{asset_name}",
            "audio_file_name": asset_name,
            "audio_storage": audio_storage,
            "sample_rate": sample_rate,
            "duration_seconds": duration_seconds,
            "processing_time": total_wall_seconds,
            "rtf": rtf,
            "created_at": now_iso(),
            "model_name": self._model_name or self._settings().active_model,
            "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
            "total_wall_ms": round(total_wall_ms, 2),
            "audio_duration_ms": round(duration_seconds * 1000.0, 2),
            "sentence_count": sentence_count,
            "batch_count": batch_count,
        }
        self._append_generation_history(result)
        return result

    def generated_audio_asset(self, file_name: str) -> dict[str, Any]:
        path = self.generated_dir / file_name
        if path.exists():
            return {"kind": "file", "path": path}
        with self._generated_audio_cache_lock:
            payload = self._generated_audio_cache.get(file_name)
            if payload is not None:
                self._generated_audio_cache.move_to_end(file_name)
                return {"kind": "memory", "content": payload}
        raise FileNotFoundError(f"Generated audio '{file_name}' was not found.")

    def generated_audio_path(self, file_name: str) -> Path:
        path = self.generated_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Generated audio '{file_name}' was not found.")
        return path

    def model_statuses(self, storage_path: str | None = None) -> list[dict[str, Any]]:
        settings = self._settings()
        storage_root = _resolve_storage_root(self.project_root, storage_path or settings.model_storage_path)
        storage_key = os.path.normcase(str(storage_root))
        with self._download_lock:
            jobs = dict(self._download_jobs)

        statuses: list[dict[str, Any]] = []
        for model_id, metadata in MODEL_DOWNLOAD_TARGETS.items():
            repo_root = resolve_cached_repo_root(model_id, self.project_root, extra_roots=[storage_root])
            snapshot = resolve_cached_snapshot(model_id, self.project_root, extra_roots=[storage_root])
            job = jobs.get((model_id, storage_key))

            if job and job.get("status") == "downloading":
                status = "downloading"
            elif snapshot:
                status = "ready"
            elif job and job.get("status") == "error":
                status = "error"
            elif repo_root:
                status = "partial"
            else:
                status = "missing"

            size_source = snapshot or repo_root
            statuses.append(
                {
                    "id": model_id,
                    "label": metadata["label"],
                    "kind": metadata["kind"],
                    "status": status,
                    "local_path": str(snapshot) if snapshot else None,
                    "cache_path": str(repo_root) if repo_root else None,
                    "error": job.get("error") if job and job.get("status") == "error" else None,
                    "updated_at": job.get("updated_at") if job else None,
                    "storage_root": str(storage_root),
                    "approx_size_gb": metadata.get("approx_size_gb"),
                    "size_on_disk_gb": _directory_size_gb(size_source),
                }
            )
        return statuses

    def queue_model_download(self, model_id: str, storage_path: str | None = None) -> dict[str, Any]:
        if model_id not in MODEL_DOWNLOAD_TARGETS:
            raise ValueError(f"Unsupported model id '{model_id}'.")

        settings = self._settings()
        storage_root = _resolve_storage_root(self.project_root, storage_path or settings.model_storage_path)
        storage_key = os.path.normcase(str(storage_root))
        storage_root.mkdir(parents=True, exist_ok=True)
        with self._download_lock:
            existing = self._download_jobs.get((model_id, storage_key))
            if existing and existing["status"] == "downloading":
                return dict(existing)
            self._download_jobs[(model_id, storage_key)] = {
                "model_id": model_id,
                "storage_root": str(storage_root),
                "status": "downloading",
                "error": None,
                "updated_at": now_iso(),
            }

        def worker() -> None:
            try:
                snapshot_download(model_id, cache_dir=str(storage_root), resume_download=True, **self._hf_kwargs())
            except Exception as exc:
                with self._download_lock:
                    self._download_jobs[(model_id, storage_key)] = {
                        "model_id": model_id,
                        "storage_root": str(storage_root),
                        "status": "error",
                        "error": str(exc),
                        "updated_at": now_iso(),
                    }
                return

            with self._download_lock:
                self._download_jobs[(model_id, storage_key)] = {
                    "model_id": model_id,
                    "storage_root": str(storage_root),
                    "status": "ready",
                    "error": None,
                    "updated_at": now_iso(),
                }

        threading.Thread(target=worker, daemon=True).start()
        with self._download_lock:
            return dict(self._download_jobs[(model_id, storage_key)])

    def delete_model_cache(self, model_id: str, storage_path: str | None = None) -> dict[str, Any]:
        if model_id not in MODEL_DOWNLOAD_TARGETS:
            raise ValueError(f"Unsupported model id '{model_id}'.")

        settings = self._settings()
        storage_root = _resolve_storage_root(self.project_root, storage_path or settings.model_storage_path)
        storage_key = os.path.normcase(str(storage_root))
        with self._download_lock:
            existing = self._download_jobs.get((model_id, storage_key))
            if existing and existing["status"] == "downloading":
                raise ValueError(f"Model '{model_id}' is still downloading.")

        repo_root = resolve_cached_repo_root(model_id, self.project_root, extra_roots=[storage_root])
        removed = False
        removed_path = None
        if repo_root and repo_root.exists():
            shutil.rmtree(repo_root)
            removed = True
            removed_path = str(repo_root)

        with self._download_lock:
            self._download_jobs.pop((model_id, storage_key), None)

        return {
            "ok": True,
            "removed": removed,
            "removed_path": removed_path,
            "storage_root": str(storage_root),
        }

    @staticmethod
    def _output_sample_rate(model: Any) -> int:
        decoder = getattr(model, "decoder", None)
        if decoder is None:
            return 24000
        sample_rate = getattr(decoder, "sample_rate", None)
        if isinstance(sample_rate, int):
            return sample_rate
        config = getattr(decoder, "config", None)
        config_rate = getattr(config, "sample_rate", None)
        if isinstance(config_rate, int):
            return config_rate
        return 24000

    def _friendly_encoder_error(self, exc: Exception, language: str) -> str:
        message = str(exc)
        lowered = message.lower()
        if (
            "meta-llama/llama-3.2-1b" in lowered
            or "couldn't connect to 'https://huggingface.co'" in lowered
            or "localentrynotfounderror" in lowered
            or "401 client error" in lowered
            or "403 client error" in lowered
            or "gated" in lowered
        ):
            return (
                "Voice cloning needs access to the gated Hugging Face tokenizer `meta-llama/Llama-3.2-1B`. "
                "Please accept the license on Hugging Face and add a valid HF token in the admin settings. "
                f"Current language: {language}."
            )
        return message

    def _friendly_error(self, exc: Exception) -> str:
        message = str(exc)
        lowered = message.lower()
        if "gated" in lowered or "access" in lowered or "meta-llama" in lowered:
            return (
                "Model access was denied. Accept the Meta Llama 3.2 license on Hugging Face and make sure "
                "your HF token is configured before starting downloads or loading the model."
            )
        if "out of memory" in lowered or "cuda error" in lowered:
            return (
                "The model ran out of GPU memory. Keep CPU offload enabled or switch to HumeAI/tada-1b if needed."
            )
        if "no module named" in lowered:
            return "Python dependencies are incomplete. Re-run install.bat so the local vendor packages are installed."
        return message
