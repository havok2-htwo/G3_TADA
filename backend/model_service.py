from __future__ import annotations

import base64
import gc
import importlib.util
import json
import os
import queue
import re
import shutil
import time
import threading
import types
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import get_token

from backend.hf_cache import resolve_pretrained_source
try:
    from tada.modules.encoder import Encoder, EncoderOutput
    from tada.modules.tada import TadaForCausalLM, InferenceOptions

    TADA_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - runtime dependency availability varies.
    Encoder = None  # type: ignore[assignment]
    EncoderOutput = None  # type: ignore[assignment]
    TadaForCausalLM = None  # type: ignore[assignment]
    InferenceOptions = None # type: ignore[assignment]
    TADA_IMPORT_ERROR = exc


DEFAULT_FLOW_STEPS = max(1, int(os.getenv("TADA_DEFAULT_STEPS", "10")))
DEFAULT_MODEL_NAME = "HumeAI/tada-3b-ml"

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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return slug.strip("-") or "voice"


def safe_json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def load_audio_tensor(path: Path) -> tuple[torch.Tensor, int]:
    """Load audio without relying on torchaudio's torchcodec path."""
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


def encode_pcm16_base64(waveform: torch.Tensor) -> str:
    mono = waveform.detach().float().cpu().reshape(-1).clamp(-1.0, 1.0).numpy()
    pcm16 = np.clip(np.round(mono * 32767.0), -32768, 32767).astype("<i2", copy=False)
    return base64.b64encode(pcm16.tobytes()).decode("ascii")


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



@dataclass(slots=True)
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

    @classmethod
    def from_file(cls, metadata_path: Path) -> "VoiceRecord":
        payload = safe_json_load(metadata_path)
        return cls(
            voice_id=payload["voice_id"],
            name=payload["name"],
            language=payload["language"],
            transcript=payload["transcript"],
            created_at=payload["created_at"],
            reference_audio=payload["reference_audio"],
            prompt_cache=payload["prompt_cache"],
            sample_rate=int(payload["sample_rate"]),
            duration_seconds=float(payload["duration_seconds"]),
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
        }


class TadaService:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.backend_dir = project_root / "backend"
        self.voices_dir = self.backend_dir / "data" / "voices"
        self.generated_dir = self.backend_dir / "generated"
        self.offload_dir = self.backend_dir / "offload"
        self.model_name = self._normalize_model_name(
            os.getenv("TADA_MODEL_NAME", DEFAULT_MODEL_NAME),
            allow_unknown=True,
        )
        self.codec_name = os.getenv("TADA_CODEC_NAME", "HumeAI/tada-codec")
        self.hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or get_token()
        self._encoders: dict[str, Any] = {}
        self._model: Any | None = None
        self._model_lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self._model_load_error: str | None = None

        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.offload_dir.mkdir(parents=True, exist_ok=True)

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
        self.model_dtype = (
            torch.bfloat16
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        self.enable_cpu_offload = os.getenv(
            "TADA_ENABLE_CPU_OFFLOAD",
            "0"  # Explizit deaktiviert fÃ¼r volle Performance, sofern VRAM reicht
        ).lower() in {"1", "true", "yes"}
        self.decoder_device = torch.device(
            os.getenv(
                "TADA_DECODER_DEVICE",
                "cpu" if self.enable_cpu_offload else str(self.device),
            )
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

    def status(self) -> dict[str, Any]:
        voices = self.list_voices()
        return {
            "ok": TADA_IMPORT_ERROR is None,
            "cuda_available": torch.cuda.is_available(),
            "device": str(self.device),
            "encoder_device": str(self.encoder_device),
            "decoder_device": str(self.decoder_device),
            "gpu_name": self.gpu_name,
            "gpu_memory_gib": self.gpu_memory_gib,
            "model_name": self.model_name,
            "available_models": self.available_models(),
            "model_loaded": self._model is not None,
            "cpu_offload": self.enable_cpu_offload,
            "dtype": str(self.model_dtype).replace("torch.", ""),
            "default_flow_steps": DEFAULT_FLOW_STEPS,
            "torch_compile_enabled": self.enable_torch_compile,
            "voices_count": len(voices),
            "languages": SUPPORTED_LANGUAGES,
            "hf_token_present": bool(self.hf_token),
            "import_error": str(TADA_IMPORT_ERROR) if TADA_IMPORT_ERROR else None,
            "model_error": self._model_load_error,
        }

    def list_voices(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for metadata_path in self.voices_dir.glob("*/metadata.json"):
            record = VoiceRecord.from_file(metadata_path)
            payload = record.to_dict()
            payload["reference_url"] = f"/api/voices/{record.voice_id}/reference"
            records.append(payload)
        return sorted(records, key=lambda item: item["created_at"], reverse=True)

    def get_voice(self, voice_id: str) -> dict[str, Any]:
        metadata_path = self.voices_dir / voice_id / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Voice '{voice_id}' was not found.")
        record = VoiceRecord.from_file(metadata_path)
        payload = record.to_dict()
        payload["reference_url"] = f"/api/voices/{record.voice_id}/reference"
        return payload

    def reference_audio_path(self, voice_id: str) -> Path:
        voice = self.get_voice(voice_id)
        return self.voices_dir / voice_id / voice["reference_audio"]

    def _ensure_runtime_ready(self) -> None:
        if TADA_IMPORT_ERROR is not None:
            raise RuntimeError(
                "TADA is not installed yet. Run install.bat first so the local vendor packages are available."
            ) from TADA_IMPORT_ERROR

    def available_models(self) -> list[dict[str, str]]:
        return [
            {
                "id": model_name,
                "label": metadata["label"],
                "description": metadata["description"],
            }
            for model_name, metadata in AVAILABLE_MODELS.items()
        ]

    @staticmethod
    def _normalize_model_name(model_name: str, *, allow_unknown: bool = False) -> str:
        candidate = model_name.strip()
        if not candidate:
            if allow_unknown:
                return DEFAULT_MODEL_NAME
            raise ValueError("Model name must not be empty.")

        normalized = MODEL_NAME_ALIASES.get(candidate.lower(), candidate)
        if allow_unknown or normalized in AVAILABLE_MODELS:
            return normalized

        supported = ", ".join(AVAILABLE_MODELS.keys())
        raise ValueError(f"Unsupported model '{model_name}'. Available: {supported}.")

    def _hf_kwargs(self) -> dict[str, Any]:
        return {"token": self.hf_token} if self.hf_token else {}

    def _dispose_model(self, model: Any | None) -> None:
        if model is not None:
            del model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    def set_model(self, model_name: str) -> dict[str, Any]:
        normalized_model_name = self._normalize_model_name(model_name)
        model_to_dispose: Any | None = None

        with self._generation_lock:
            with self._model_lock:
                if normalized_model_name != self.model_name:
                    model_to_dispose = self._model
                    self._model = None
                    self.model_name = normalized_model_name
                    os.environ["TADA_MODEL_NAME"] = normalized_model_name
                self._model_load_error = None

            self._dispose_model(model_to_dispose)

        return self.status()

    def _load_model(self) -> Any:
        self._ensure_runtime_ready()
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model
            try:
                model_source, model_is_local = resolve_pretrained_source(self.model_name, self.project_root)
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

                model = TadaForCausalLM.from_pretrained(model_source, **load_kwargs)
                if "device_map" not in load_kwargs:
                    model = model.to(self.device)

                # VITAL FÃœR PERFORMANCE: Tausende Python-Schleifen-Aufrufe des kleinen MLP Ã¼berlasten die CPU.
                # Das Compilieren des Prediction-Heads fÃ¤ngt das Kernel-Launch-Overhead auf.
                if self.enable_torch_compile:
                    try:
                        import torch._dynamo
                        torch._dynamo.config.suppress_errors = True
                        model.prediction_head = torch.compile(model.prediction_head, mode="reduce-overhead")
                        print("TADA prediction head compiled with torch.compile for better CUDA throughput.", flush=True)
                    except Exception as exc:
                        print(f"Warning: torch.compile warmup failed: {exc}", flush=True)

                model.decoder.to(self.decoder_device)
                model._decode_wav = types.MethodType(patched_decode_wav, model)
                model.eval()
                self._model = model
                self._model_load_error = None
            except Exception as exc:  # pragma: no cover - depends on runtime auth/hardware.
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

            if not self.hf_token:
                raise RuntimeError(
                    "Voice cloning needs access to the gated Hugging Face tokenizer `meta-llama/Llama-3.2-1B`. "
                    "Please run `huggingface-cli login` or set `HF_TOKEN` before starting the app."
                )
            kwargs = self._hf_kwargs()
            encoder_source, encoder_is_local = resolve_pretrained_source(self.codec_name, self.project_root)
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

    def create_voice(self, *, name: str, language: str, transcript: str, upload_path: Path) -> dict[str, Any]:
        normalized_language = language.strip().lower() or "en"
        if normalized_language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language '{normalized_language}'.")
        if normalized_language != "en" and not transcript.strip():
            raise ValueError("For non-English voices you should provide the transcript of the reference audio.")

        voice_id = f"{slugify(name)}-{uuid.uuid4().hex[:8]}"
        voice_dir = self.voices_dir / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)

        reference_path = voice_dir / "reference.wav"
        prompt_cache_path = voice_dir / "prompt_cache.pt"
        try:
            waveform, sample_rate = load_audio_tensor(upload_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.clamp(-1.0, 1.0)
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
            name=name.strip() or "Meine Stimme",
            language=normalized_language,
            transcript=transcript.strip(),
            created_at=now_iso(),
            reference_audio=reference_path.name,
            prompt_cache=prompt_cache_path.name,
            sample_rate=int(sample_rate),
            duration_seconds=duration_seconds,
        )
        safe_json_dump(voice_dir / "metadata.json", record.to_dict())
        return self.get_voice(voice_id)

    def generate(self, *, text: str, voice_id: str, steps: int = DEFAULT_FLOW_STEPS) -> dict[str, Any]:
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("Text must not be empty.")

        voice = self.get_voice(voice_id)
        prompt_path = self.voices_dir / voice_id / voice["prompt_cache"]

        t0 = time.time()
        with self._generation_lock:
            model = self._load_model()
            prompt = EncoderOutput.load(str(prompt_path), device=str(self.prompt_device))
            
            # HÃ¶here Anzahl an Steps fÃ¼r bessere AudioqualitÃ¤t
            options = InferenceOptions(
                num_flow_matching_steps=steps,  # Nutzerdefiniert oder 32 als Standard
            )
            
            with torch.inference_mode():
                output = model.generate(prompt=prompt, text=cleaned_text, inference_options=options)
        
        t1 = time.time()
        processing_time = t1 - t0

        if not output.audio or output.audio[0] is None:
            raise RuntimeError(
                "TADA returned no waveform. The most likely cause is a decoder runtime error during waveform reconstruction."
            )

        audio = output.audio[0].detach().float().cpu()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        sample_rate = self._output_sample_rate(model)
        generation_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        audio_path = self.generated_dir / f"{generation_id}.wav"
        save_audio_tensor(audio_path, audio, sample_rate)
        
        duration_seconds = round(float(audio.shape[-1]) / float(sample_rate), 2)
        rtf = round(processing_time / duration_seconds, 2) if duration_seconds > 0 else 0.0

        return {
            "generation_id": generation_id,
            "text": cleaned_text,
            "voice_id": voice_id,
            "audio_url": f"/api/generated/{audio_path.name}",
            "sample_rate": sample_rate,
            "duration_seconds": duration_seconds,
            "processing_time": round(processing_time, 2),
            "rtf": rtf,
            "created_at": now_iso(),
            "model_name": self.model_name,
        }

    def generate_stream(
        self,
        *,
        text: str,
        voice_id: str,
        steps: int = DEFAULT_FLOW_STEPS,
        stream_buffer_ms: int = 500,
        progress_interval_steps: int = 24,
    ):
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("Text must not be empty.")

        voice = self.get_voice(voice_id)
        prompt_path = self.voices_dir / voice_id / voice["prompt_cache"]
        event_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()

        def worker() -> None:
            try:
                t0 = time.time()
                with self._generation_lock:
                    model = self._load_model()
                    prompt = EncoderOutput.load(str(prompt_path), device=str(self.prompt_device))
                    sample_rate = self._output_sample_rate(model)
                    buffer_samples = max(0, int(sample_rate * stream_buffer_ms / 1000))
                    min_emit_samples = max(1, int(sample_rate * 0.2))
                    emitted_samples = 0

                    event_queue.put(
                        {
                            "type": "start",
                            "sample_rate": sample_rate,
                            "buffer_ms": stream_buffer_ms,
                        }
                    )

                    def on_progress(update: dict[str, Any]) -> None:
                        nonlocal emitted_samples

                        encoded = update["encoded"]
                        time_before = update["time_before"]
                        step_index = int(update["step"])
                        is_final = bool(update["is_final"])
                        if encoded.shape[0] == 0 or encoded.shape[1] == 0 or time_before.shape[1] <= 1:
                            return

                        try:
                            wav = model._decode_wav(encoded[0], time_before=time_before[0]).squeeze(0, 1)
                        except Exception:
                            return

                        lead_samples = int(sample_rate * float(time_before[0][0].item()) / 50.0)
                        if lead_samples > 0:
                            wav = wav[..., lead_samples:]

                        audio = wav.detach().float().cpu().reshape(-1)
                        total_samples = int(audio.numel())
                        event_queue.put(
                            {
                                "type": "progress",
                                "step": step_index,
                                "decoded_seconds": round(total_samples / sample_rate, 2),
                                "emitted_seconds": round(emitted_samples / sample_rate, 2),
                            }
                        )
                        if total_samples <= emitted_samples:
                            return

                        safe_end = total_samples if is_final else max(emitted_samples, total_samples - buffer_samples)
                        if safe_end <= emitted_samples:
                            return
                        if not is_final and safe_end - emitted_samples < min_emit_samples:
                            return

                        chunk = audio[emitted_samples:safe_end]
                        if chunk.numel() == 0:
                            return

                        emitted_samples = safe_end
                        event_queue.put(
                            {
                                "type": "chunk",
                                "step": step_index,
                                "sample_rate": sample_rate,
                                "samples": int(chunk.numel()),
                                "emitted_seconds": round(emitted_samples / sample_rate, 2),
                                "pcm16_b64": encode_pcm16_base64(chunk),
                            }
                        )

                    options = InferenceOptions(num_flow_matching_steps=steps)
                    with torch.inference_mode():
                        output = model.generate(
                            prompt=prompt,
                            text=cleaned_text,
                            inference_options=options,
                            progress_callback=on_progress,
                            progress_interval_steps=progress_interval_steps,
                        )

                    if not output.audio or output.audio[0] is None:
                        raise RuntimeError(
                            "TADA returned no waveform. The most likely cause is a decoder runtime error during waveform reconstruction."
                        )

                    audio = output.audio[0].detach().float().cpu()
                    if audio.ndim == 1:
                        audio = audio.unsqueeze(0)

                    final_mono = audio.reshape(-1)
                    if emitted_samples < int(final_mono.numel()):
                        tail = final_mono[emitted_samples:]
                        if tail.numel() > 0:
                            event_queue.put(
                                {
                                    "type": "chunk",
                                    "step": -1,
                                    "sample_rate": sample_rate,
                                    "samples": int(tail.numel()),
                                    "emitted_seconds": round(float(final_mono.numel()) / sample_rate, 2),
                                    "pcm16_b64": encode_pcm16_base64(tail),
                                }
                            )

                    generation_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
                    audio_path = self.generated_dir / f"{generation_id}.wav"
                    save_audio_tensor(audio_path, audio, sample_rate)

                    processing_time = time.time() - t0
                    duration_seconds = round(float(audio.shape[-1]) / float(sample_rate), 2)
                    rtf = round(processing_time / duration_seconds, 2) if duration_seconds > 0 else 0.0
                    event_queue.put(
                        {
                            "type": "done",
                            "result": {
                                "generation_id": generation_id,
                                "text": cleaned_text,
                                "voice_id": voice_id,
                                "audio_url": f"/api/generated/{audio_path.name}",
                                "sample_rate": sample_rate,
                                "duration_seconds": duration_seconds,
                                "processing_time": round(processing_time, 2),
                                "rtf": rtf,
                                "created_at": now_iso(),
                                "model_name": self.model_name,
                            },
                        }
                    )
            except FileNotFoundError as exc:
                event_queue.put({"type": "error", "message": str(exc)})
            except ValueError as exc:
                event_queue.put({"type": "error", "message": str(exc)})
            except RuntimeError as exc:
                event_queue.put({"type": "error", "message": str(exc)})
            except Exception as exc:
                event_queue.put({"type": "error", "message": str(exc)})
            finally:
                event_queue.put(None)

        threading.Thread(target=worker, daemon=True).start()

        def iter_events():
            while True:
                item = event_queue.get()
                if item is None:
                    break
                yield item

        return iter_events()

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
                "Please accept the license on Hugging Face and set `HF_TOKEN` before starting the app. "
                f"Current language: {language}."
            )

        return message

    def _friendly_error(self, exc: Exception) -> str:
        message = str(exc)
        lowered = message.lower()

        if "gated" in lowered or "access" in lowered or "meta-llama" in lowered:
            return (
                "Model access was denied. Accept the Meta Llama 3.2 license on Hugging Face and make sure "
                "your HF token is available before starting the app."
            )

        if "out of memory" in lowered or "cuda error" in lowered:
            return (
                "The model ran out of GPU memory. On an 8 GB RTX 5070 laptop GPU, TADA-3B-ml can be tight. "
                "Keep CPU offload enabled or switch TADA_MODEL_NAME to HumeAI/tada-1b if needed."
            )

        if "no module named" in lowered:
            return (
                "Python dependencies are incomplete. Re-run install.bat so the local vendor packages are installed."
            )

        return message
























