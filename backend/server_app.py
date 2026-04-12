from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.batch_scheduler import BatchScheduler
from backend.config_store import ConfigStore
from backend.runtime_service import DuplicateVoiceNameError, SUPPORTED_LANGUAGES, TadaRuntimeService

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

config_store = ConfigStore(PROJECT_ROOT)
config_store.apply_runtime_environment()
runtime_service = TadaRuntimeService(PROJECT_ROOT, config_store)
scheduler = BatchScheduler(runtime_service, config_store)


class AdminSettingsUpdateRequest(BaseModel):
    active_model: Optional[str] = None
    steps: Optional[int] = Field(default=None, ge=1, le=128)
    sentence_chunking: Optional[bool] = None
    short_sentence_merge_max_chars: Optional[int] = Field(default=None, ge=0, le=200)
    following_sentence_merge_min_chars: Optional[int] = Field(default=None, ge=0, le=500)
    allow_lan_access: Optional[bool] = None
    stream_start_buffer_ms: Optional[int] = Field(default=None, ge=0, le=5000)
    stream_chunk_ms: Optional[int] = Field(default=None, ge=50, le=5000)
    batch_wait_ms: Optional[int] = Field(default=None, ge=0, le=5000)
    max_batch_size: Optional[int] = Field(default=None, ge=1, le=128)
    max_parallel_requests: Optional[int] = Field(default=None, ge=1, le=128)
    max_queue_size: Optional[int] = Field(default=None, ge=1, le=2048)
    model_storage_path: Optional[str] = None
    whisper_base_url: Optional[str] = None
    hf_token: Optional[str] = None
    whisper_api_key: Optional[str] = None
    
    # VAD & Audio Settings
    vad_trimming: Optional[bool] = None
    duration_multiplier: Optional[float] = Field(default=None, ge=0.5, le=5.0)
    vad_threshold_pct: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    vad_padding_ms: Optional[int] = Field(default=None, ge=0, le=5000)
    vad_fade_ms: Optional[int] = Field(default=None, ge=0, le=2000)


class ModelDownloadRequest(BaseModel):
    model_id: str


class PublicSynthesizeRequest(BaseModel):
    text: str
    voice_id: str


def _unauthorized(message: str) -> HTTPException:
    return HTTPException(status_code=401, detail=message)


def require_admin_key(x_admin_key: Optional[str] = Header(None)) -> dict[str, str]:
    if not config_store.verify_admin_key((x_admin_key or "").strip()):
        raise _unauthorized("A valid X-Admin-Key header is required.")
    return {"role": "admin"}


app = FastAPI(title="TADA Batch TTS Server", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def warmup_on_startup() -> None:
    if os.getenv("TADA_DISABLE_WARMUP", "0").lower() in {"1", "true", "yes"}:
        return

    def _warmup() -> None:
        print("Starting TADA runtime warmup...", flush=True)
        try:
            runtime_service.sync_runtime_settings()
            model = runtime_service._load_model()
            runtime_service._load_encoder("de")
            if runtime_service.enable_torch_compile:
                import torch

                head = model.prediction_head
                head_param = next(head.parameters())
                head_device = head_param.device
                head_dtype = head_param.dtype
                latent_dim = int(head.noisy_images_proj.in_features)
                cond_dim = int(head.cond_proj.in_features)
                dummy_speech = torch.randn(1, latent_dim, device=head_device, dtype=head_dtype)
                dummy_t = torch.ones(1, device=head_device, dtype=head_dtype)
                dummy_cond = torch.randn(1, cond_dim, device=head_device, dtype=head_dtype)
                try:
                    head(dummy_speech, dummy_t, condition=dummy_cond)
                except Exception as exc:
                    print(f"Compile warmup failed, eager inference remains active: {exc}", flush=True)
            print("TADA runtime warmup finished.", flush=True)
        except Exception as exc:
            print(f"Warning: warmup failed: {exc}", flush=True)

    threading.Thread(target=_warmup, daemon=True).start()


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "settings": config_store.export_settings_snapshot(),
        "runtime": runtime_service.status(),
        "dashboard": scheduler.dashboard_snapshot(),
    }


@app.get("/api/admin/settings")
def admin_settings(_: dict[str, str] = Depends(require_admin_key)) -> dict[str, Any]:
    return {
        "settings": config_store.export_settings_snapshot(),
        "runtime": runtime_service.status(),
        "models": runtime_service.model_statuses(),
    }


@app.put("/api/admin/settings")
def update_admin_settings(
    request: AdminSettingsUpdateRequest,
    _: dict[str, str] = Depends(require_admin_key),
) -> dict[str, Any]:
    payload = request.model_dump(exclude_none=True)
    updated = config_store.update_settings(payload)
    runtime_service.sync_runtime_settings()
    return {
        "settings": updated,
        "runtime": runtime_service.status(),
        "models": runtime_service.model_statuses(),
    }


@app.get("/api/admin/models")
def admin_models(_: dict[str, str] = Depends(require_admin_key)) -> dict[str, Any]:
    return {"models": runtime_service.model_statuses()}


@app.post("/api/admin/models/download")
def admin_model_download(
    request: ModelDownloadRequest,
    _: dict[str, str] = Depends(require_admin_key),
) -> dict[str, Any]:
    try:
        job = runtime_service.queue_model_download(request.model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"job": job, "models": runtime_service.model_statuses()}


@app.get("/api/admin/voices")
def admin_voices(_: dict[str, str] = Depends(require_admin_key)) -> dict[str, Any]:
    return {"voices": runtime_service.list_voices(), "languages": SUPPORTED_LANGUAGES}


@app.post("/api/admin/voices")
async def admin_create_voice(
    name: str = Form(...),
    language: str = Form("en"),
    transcript: str = Form(""),
    trim_start_ms: Optional[int] = Form(None),
    trim_end_ms: Optional[int] = Form(None),
    overwrite_existing: bool = Form(False),
    audio: UploadFile = File(...),
    _: dict[str, str] = Depends(require_admin_key),
) -> dict[str, Any]:
    suffix = Path(audio.filename or "reference.wav").suffix or ".wav"
    temp_path = Path(tempfile.gettempdir()) / f"tada-upload-{uuid.uuid4().hex}{suffix}"
    try:
        with temp_path.open("wb") as handle:
            shutil.copyfileobj(audio.file, handle)
        voice = runtime_service.create_voice(
            name=name,
            language=language,
            transcript=transcript,
            upload_path=temp_path,
            trim_start_ms=trim_start_ms,
            trim_end_ms=trim_end_ms,
            overwrite_existing=overwrite_existing,
        )
        return {"voice": voice}
    except DuplicateVoiceNameError as exc:
        return JSONResponse(
            status_code=409,
            content={
                "message": str(exc),
                "code": "voice_name_exists",
                "existing_voice_id": exc.existing_voice_id,
                "existing_voice_name": exc.existing_voice_name,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/api/admin/voices/transcribe")
async def admin_transcribe_voice(
    trim_start_ms: Optional[int] = Form(None),
    trim_end_ms: Optional[int] = Form(None),
    audio: UploadFile = File(...),
    _: dict[str, str] = Depends(require_admin_key),
) -> dict[str, Any]:
    suffix = Path(audio.filename or "reference.wav").suffix or ".wav"
    temp_path = Path(tempfile.gettempdir()) / f"tada-whisper-{uuid.uuid4().hex}{suffix}"
    try:
        with temp_path.open("wb") as handle:
            shutil.copyfileobj(audio.file, handle)
        return runtime_service.transcribe_audio(
            upload_path=temp_path,
            trim_start_ms=trim_start_ms,
            trim_end_ms=trim_end_ms,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)


@app.delete("/api/admin/voices/{voice_id}")
def admin_delete_voice(
    voice_id: str,
    _: dict[str, str] = Depends(require_admin_key),
) -> dict[str, Any]:
    try:
        runtime_service.delete_voice(voice_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, "voices": runtime_service.list_voices(), "languages": SUPPORTED_LANGUAGES}


@app.get("/api/admin/keys")
def admin_keys(_: dict[str, str] = Depends(require_admin_key)) -> dict[str, Any]:
    return config_store.list_keys()


@app.post("/api/admin/keys")
def admin_rotate_key(_: dict[str, str] = Depends(require_admin_key)) -> dict[str, Any]:
    return {"key": config_store.rotate_admin_key(), "keys": config_store.list_keys()}


@app.get("/api/admin/generations")
def admin_generations(_: dict[str, str] = Depends(require_admin_key)) -> dict[str, Any]:
    return {"generations": runtime_service.list_recent_generations()}


@app.get("/api/admin/dashboard/stream")
def admin_dashboard_stream(_: dict[str, str] = Depends(require_admin_key)) -> StreamingResponse:
    def iter_lines():
        for event in scheduler.iter_dashboard_events():
            yield json.dumps(event, ensure_ascii=True) + "\n"

    return StreamingResponse(iter_lines(), media_type="application/x-ndjson")


@app.get("/api/v1/voices")
def public_voices() -> dict[str, Any]:
    return {"voices": runtime_service.list_voices()}


@app.post("/api/v1/synthesize")
def public_synthesize(
    request: PublicSynthesizeRequest,
) -> dict[str, Any]:
    try:
        runtime_service.get_voice(request.voice_id)
        state = scheduler.submit_request(text=request.text, voice_id=request.voice_id)
        return scheduler.wait_for_result(state)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=429 if "queue is full" in str(exc).lower() else 500, detail=str(exc)) from exc


@app.post("/api/v1/synthesize/stream")
def public_synthesize_stream(
    request: PublicSynthesizeRequest,
) -> StreamingResponse:
    try:
        runtime_service.get_voice(request.voice_id)
        state = scheduler.submit_request(text=request.text, voice_id=request.voice_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=429 if "queue is full" in str(exc).lower() else 500, detail=str(exc)) from exc

    def iter_lines():
        for event in scheduler.iter_request_events(state):
            yield json.dumps(event, ensure_ascii=True) + "\n"

    return StreamingResponse(iter_lines(), media_type="application/x-ndjson")


@app.get("/api/assets/generated/{file_name}")
def generated_audio(
    file_name: str,
) -> FileResponse:
    try:
        path = runtime_service.generated_audio_path(file_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type="audio/wav", filename=file_name)


@app.get("/api/assets/voices/{voice_id}/reference")
def voice_reference(
    voice_id: str,
    _: dict[str, str] = Depends(require_admin_key),
) -> FileResponse:
    try:
        path = runtime_service.reference_audio_path(voice_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@app.get("/{full_path:path}", include_in_schema=False)
def frontend(full_path: str):
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")

    if not FRONTEND_DIST.exists():
        return HTMLResponse(
            """
            <html>
              <body style="font-family: sans-serif; padding: 24px;">
                <h1>TADA frontend not built yet</h1>
                <p>Run <code>install.bat</code> once, then start the app again.</p>
              </body>
            </html>
            """,
            status_code=503,
        )

    route_map = {
        "": "index.html",
        "admin": "admin.html",
        "admin/": "admin.html",
        "demo": "demo.html",
        "demo/": "demo.html",
    }
    target_name = route_map.get(full_path)
    if target_name:
        target = FRONTEND_DIST / target_name
        if target.exists():
            return FileResponse(target)

    candidate = FRONTEND_DIST / full_path
    if full_path and candidate.exists() and candidate.is_file():
        return FileResponse(candidate)

    index_path = FRONTEND_DIST / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend asset not found.")
