# TADA Batch TTS API Documentation

This file is the API-focused companion to [README.md](README.md). It documents the HTTP routes, auth headers, request payloads, streaming event format, and the most important response fields for integrations.

Base URL examples:

- local only: `http://127.0.0.1:7878`
- LAN mode: `http://<server-ip>:7878`

Machine-readable schema:

- `GET /openapi.json`
- `GET /docs`

## Authentication

Admin routes require:

- header: `X-Admin-Key: <admin token>`
- used for all `/api/admin/...` routes
- also required for `GET /api/assets/voices/{voice_id}/reference`

Public routes are open:

- `GET /api/v1/voices`
- `POST /api/v1/synthesize`
- `POST /api/v1/synthesize/stream`
- `GET /v1/models`
- `GET /v1/voices`
- `GET /v1/audio/voices`
- `POST /v1/audio/speech`
- `GET /api/assets/generated/{file_name}`

Startup note:

- `start.bat` and `start.sh` print a temporary startup admin key before the server comes up
- `GET /api/admin/keys` shows metadata for the persistent admin key
- `POST /api/admin/keys` rotates the persistent admin key and returns the new plaintext token once

## Common Response Rules

- `200` on success
- `400` for bad input
- `422` for request-schema validation failures
- `401` for missing or invalid auth
- `404` if the requested voice, file, or key does not exist
- `409` for voice-name conflicts during voice creation
- `429` if the synthesis queue is full
- `500` for runtime/server errors

Error shape:

```json
{
  "detail": "Human-readable error message"
}
```

## Health

### `GET /api/health`

Returns the currently effective settings, runtime status, and scheduler dashboard snapshot.

Example response shape:

```json
{
  "settings": {
    "active_model": "HumeAI/tada-3b-ml",
    "model_precision": "fp16",
    "deterministic_seed": null,
    "persist_generated_wavs": false,
    "steps": 10,
    "sentence_chunking": true,
    "short_sentence_merge_max_chars": 30,
    "following_sentence_merge_min_chars": 20,
    "allow_lan_access": false,
    "stream_start_buffer_ms": 500,
    "stream_chunk_ms": 500,
    "batch_wait_ms": 500,
    "max_batch_size": 8,
    "max_parallel_requests": 8,
    "max_queue_size": 256,
    "model_storage_path": "X:\\dev\\G3_TADA3B\\.hf_cache\\hub",
    "whisper_base_url": "",
    "vad_trimming": true,
    "prompt_start_trim_steps": 0,
    "vad_threshold_pct": 0.015,
    "vad_padding_ms": 150,
    "vad_fade_ms": 50,
    "hf_token_present": true,
    "whisper_api_key_present": false,
    "restart_required": false
  },
  "runtime": {},
  "dashboard": {}
}
```

Notes:

- `runtime` includes precision/runtime details such as `configured_model_precision`, `effective_model_precision`, `available_models`, `voices_count`, and `model_error`
- `dashboard` includes the same live queue/current-batch snapshot that is also emitted by `GET /api/admin/dashboard/stream`

## Admin API

## `GET /api/admin/settings`

Returns:

- `settings`: persisted server settings plus secret-presence flags
- `presets`: saved settings presets from `backend/data/presets/*.json`
- `runtime`: current runtime status
- `models`: model download/status snapshot

Notes:

- `settings` contains only `hf_token_present` / `whisper_api_key_present` booleans, never the raw secret values
- `runtime.available_models` is the authoritative model picker for the admin UI

## `PUT /api/admin/settings`

Request body:

```json
{
  "active_model": "HumeAI/tada-3b-ml",
  "model_precision": "fp16",
  "deterministic_seed": null,
  "persist_generated_wavs": false,
  "steps": 10,
  "sentence_chunking": true,
  "short_sentence_merge_max_chars": 30,
  "following_sentence_merge_min_chars": 20,
  "allow_lan_access": false,
  "stream_start_buffer_ms": 500,
  "stream_chunk_ms": 500,
  "batch_wait_ms": 500,
  "max_batch_size": 8,
  "max_parallel_requests": 8,
  "max_queue_size": 256,
  "model_storage_path": "X:\\dev\\G3_TADA3B\\.hf_cache\\hub",
  "whisper_base_url": "http://127.0.0.1:8001/v1",
  "vad_trimming": true,
  "prompt_start_trim_steps": 0,
  "vad_threshold_pct": 0.015,
  "vad_padding_ms": 150,
  "vad_fade_ms": 50,
  "hf_token": "",
  "whisper_api_key": ""
}
```

Important notes:

- `model_precision` accepts `fp16`, `bf16`, `fp32`, `bnb8`, or `fp8`
- `bnb8` uses bitsandbytes 8-bit quantization and requires the optional bitsandbytes runtime package
- `fp8` uses Transformers fine-grained FP8 quantization and requires a CUDA GPU with compute capability >= 9
- quantized modes keep TADA's audio/diffusion head in regular precision and only quantize compatible Llama-side linear layers
- quantized modes stream final stable audio chunks only; progressive preview stays enabled for `fp16`, `bf16`, and `fp32`
- `deterministic_seed` may be `null` or omitted to disable deterministic seeding
- `persist_generated_wavs=false` keeps recent generated audio in memory instead of writing every WAV to disk
- `steps` is clamped to `1..128`
- `max_parallel_requests` is clamped to `max_batch_size`
- `short_sentence_merge_max_chars` is clamped to `0..200`
- `following_sentence_merge_min_chars` is clamped to `0..500`
- `prompt_start_trim_steps` is clamped to `0..12`
- `vad_threshold_pct` is clamped to `0.0..10.0`
- `vad_padding_ms` is clamped to `0..5000`
- `vad_fade_ms` is clamped to `0..2000`
- changing `model_storage_path` or `allow_lan_access` sets `restart_required=true`
- `hf_token` and `whisper_api_key` are write-only convenience fields

## `GET /api/admin/settings/presets`

Returns all preset files currently available under `backend/data/presets`.

Example response:

```json
{
  "presets": [
    {
      "name": "fast-german",
      "label": "Fast German",
      "file_name": "fast-german.json",
      "updated_at": "2026-04-13T09:14:55+00:00",
      "settings": {
        "active_model": "HumeAI/tada-1b",
        "model_precision": "fp16",
        "steps": 8,
        "sentence_chunking": true
      }
    }
  ]
}
```

Notes:

- each preset entry contains the full saved `settings` object, not just the preview fields shown above
- invalid or unreadable preset files are skipped automatically

## `POST /api/admin/settings/presets/save`

Request:

```json
{
  "name": "Fast German"
}
```

Response:

```json
{
  "preset": {
    "name": "fast-german",
    "label": "Fast German",
    "file_name": "fast-german.json",
    "updated_at": "2026-04-13T09:14:55+00:00"
  },
  "presets": [],
  "settings": {}
}
```

Notes:

- the file name is slugified from the provided label
- the saved preset always contains the server's current exported settings snapshot

## `POST /api/admin/settings/presets/apply`

Request:

```json
{
  "name": "fast-german"
}
```

Response:

```json
{
  "settings": {
    "active_model": "HumeAI/tada-1b"
  },
  "presets": [],
  "runtime": {},
  "models": []
}
```

Notes:

- `name` may be the preset label, slug, or file name with `.json`
- returns `404` if the preset does not exist
- applying a preset updates persisted settings and immediately re-syncs the runtime

## `GET /api/admin/models`

Returns:

```json
{
  "models": [
    {
      "id": "HumeAI/tada-3b-ml",
      "label": "TADA 3B ML",
      "kind": "model",
      "status": "ready",
      "local_path": "X:\\...\\snapshot",
      "error": null,
      "updated_at": null,
      "storage_root": "X:\\dev\\G3_TADA3B\\.hf_cache\\hub"
    }
  ]
}
```

Status values commonly seen:

- `missing`
- `ready`
- `downloading`
- `error`

## `POST /api/admin/models/download`

Request:

```json
{
  "model_id": "HumeAI/tada-1b"
}
```

Response:

```json
{
  "job": {
    "model_id": "HumeAI/tada-1b",
    "status": "downloading"
  },
  "models": []
}
```

## `GET /api/admin/voices`

Response:

```json
{
  "voices": [
    {
      "voice_id": "sonya-a665be06",
      "name": "SONYA",
      "language": "de",
      "transcript": "Hallo Welt",
      "created_at": "2026-04-05T15:01:58+00:00",
      "reference_audio": "reference.wav",
      "prompt_cache": "prompt_cache.pt",
      "sample_rate": 24000,
      "duration_seconds": 3.27,
      "trim_start_ms": 500,
      "trim_end_ms": 3770,
      "source_duration_seconds": 9.84,
      "was_auto_trimmed": false,
      "tail_silence_ms": 500,
      "reference_url": "/api/assets/voices/sonya-a665be06/reference"
    }
  ],
  "languages": {
    "de": "German"
  }
}
```

## `POST /api/admin/voices`

Multipart form-data fields:

- `name`
- `language`
- `transcript`
- `trim_start_ms`
- `trim_end_ms`
- `overwrite_existing`
- `audio`

Behavior:

- duplicate names are rejected case-insensitively
- duplicate names return `409` with `code="voice_name_exists"` unless `overwrite_existing=true`
- for non-English voices, a transcript is required
- the selected reference window is saved as `reference.wav`
- spoken reference audio must be between `3 s` and `14.5 s`
- the saved prompt appends `0.5 s` of tail silence automatically

Response:

```json
{
  "voice": {
    "voice_id": "new-voice-1234abcd",
    "name": "New Voice",
    "language": "de",
    "transcript": "Hallo Welt",
    "created_at": "2026-04-13T09:32:00+00:00",
    "reference_audio": "reference.wav",
    "prompt_cache": "prompt_cache.pt",
    "sample_rate": 24000,
    "duration_seconds": 10.5,
    "trim_start_ms": 500,
    "trim_end_ms": 10000,
    "source_duration_seconds": 21.37,
    "was_auto_trimmed": false,
    "tail_silence_ms": 500,
    "reference_url": "/api/assets/voices/new-voice-1234abcd/reference"
  }
}
```

Conflict response example:

```json
{
  "message": "A voice named 'New Voice' already exists. Please choose a different name.",
  "code": "voice_name_exists",
  "existing_voice_id": "new-voice-1234abcd",
  "existing_voice_name": "New Voice"
}
```

## `POST /api/admin/voices/transcribe`

Multipart form-data fields:

- `trim_start_ms`
- `trim_end_ms`
- `audio`

Response:

```json
{
  "text": "Recognized transcript text",
  "language": "de",
  "base_url": "http://127.0.0.1:8001/v1",
  "trim_start_ms": 0,
  "trim_end_ms": 10000,
  "duration_seconds": 10.0,
  "source_duration_seconds": 21.37,
  "was_auto_trimmed": false
}
```

Notes:

- the runtime tries multiple Whisper-compatible endpoints, including OpenAI-style `/audio/transcriptions` and Genesis-style `/transcribe/`
- the response accepts either `text` or `transcription` from the upstream STT service and normalizes that to `text`

## `DELETE /api/admin/voices/{voice_id}`

Response:

```json
{
  "ok": true,
  "voices": [],
  "languages": {}
}
```

## `GET /api/admin/keys`

Response:

```json
{
  "admin_key": {
    "id": "admin",
    "label": "Master Admin Key",
    "created_at": "2026-04-05T12:00:00+00:00",
    "last_used_at": null
  }
}
```

## `POST /api/admin/keys`

Response:

```json
{
  "key": {
    "id": "admin",
    "label": "Master Admin Key",
    "token": "tada_admin_xxx",
    "created_at": "2026-04-05T12:01:00+00:00"
  },
  "keys": {
    "admin_key": {
      "id": "admin",
      "label": "Master Admin Key",
      "created_at": "2026-04-05T12:01:00+00:00",
      "last_used_at": null
    }
  }
}
```

Important:

- `token` is only returned once

## `GET /api/admin/generations`

Response:

```json
{
  "generations": [
    {
      "generation_id": "20260405-180500-a1b2c3",
      "text": "Hallo Welt. Das ist ein Test.",
      "voice_id": "sonya-a665be06",
      "voice_name": "SONYA",
      "audio_url": "/api/assets/generated/20260405-180500-a1b2c3.wav",
      "audio_file_name": "20260405-180500-a1b2c3.wav",
      "audio_storage": "memory",
      "sample_rate": 24000,
      "duration_seconds": 2.31,
      "processing_time": 1.94,
      "rtf": 0.84,
      "created_at": "2026-04-05T18:05:00+00:00",
      "model_name": "HumeAI/tada-3b-ml",
      "ttft_ms": 942.17,
      "total_wall_ms": 1942.11,
      "audio_duration_ms": 2310.0,
      "sentence_count": 2,
      "batch_count": 2
    }
  ]
}
```

Note:

- `audio_url` may be `null` for older memory-backed generations after a server restart or after the in-memory generated-audio cache was evicted

## `GET /api/admin/dashboard/stream`

Streaming response type:

- `application/x-ndjson`

One JSON object per line, roughly every `0.5 s`.

Snapshot fields:

- `timestamp`
- `queue_length`
- `queued_sentence_count`
- `active_request_count`
- `waiting_requests`
- `active_requests`
- `current_batch`
- `mean_ttft_ms`
- `throughput_audio_sps`
- `completed_requests_total`
- `failed_requests_total`
- `completed_requests_delta`
- `failed_requests_delta`
- `last_batch_wall_s`
- `last_batch_rtf`
- `history`

Notes:

- snapshots are emitted roughly every `0.5 s`
- `current_batch` contains `batch_id`, `voice_id`, `request_ids`, and sentence-scoped `items`

## Public API

There are now two parallel public integration surfaces:

- the native workbench API under `/api/v1/...`
- an OpenAI-compatible TTS shim under `/v1/...`

The old `/api/v1/...` endpoints remain fully supported for existing local tools.

## `GET /api/v1/voices`

Returns the public voice list.

Example:

```json
{
  "voices": [
    {
      "voice_id": "sonya-a665be06",
      "name": "SONYA",
      "language": "de",
      "duration_seconds": 3.27,
      "trim_start_ms": 500,
      "trim_end_ms": 3770,
      "source_duration_seconds": 9.84,
      "tail_silence_ms": 500,
      "reference_url": "/api/assets/voices/sonya-a665be06/reference"
    }
  ]
}
```

Note:

- the public route currently mirrors the stored voice metadata used by the admin route; dereferencing `reference_url` still requires `X-Admin-Key`

## `POST /api/v1/synthesize`

Request:

```json
{
  "text": "Hallo Welt. Das ist ein Test.",
  "voice_id": "sonya-a665be06"
}
```

Response:

```json
{
  "generation_id": "20260405-180500-a1b2c3",
  "text": "Hallo Welt. Das ist ein Test.",
  "voice_id": "sonya-a665be06",
  "voice_name": "SONYA",
  "audio_url": "/api/assets/generated/20260405-180500-a1b2c3.wav",
  "audio_file_name": "20260405-180500-a1b2c3.wav",
  "audio_storage": "memory",
  "sample_rate": 24000,
  "duration_seconds": 2.31,
  "processing_time": 1.94,
  "rtf": 0.84,
  "created_at": "2026-04-05T18:05:00+00:00",
  "model_name": "HumeAI/tada-3b-ml",
  "ttft_ms": 942.17,
  "total_wall_ms": 1942.11,
  "audio_duration_ms": 2310.0,
  "sentence_count": 2,
  "batch_count": 2
}
```

## `POST /api/v1/synthesize/stream`

Request:

```json
{
  "text": "Hallo Welt. Das ist ein Test.",
  "voice_id": "sonya-a665be06"
}
```

Response type:

- `application/x-ndjson`

Event order:

- `start`
- `queue`
- zero or more `batch`
- zero or more `chunk`
- `done`

If something fails, the stream ends with:

- `error`

### Streaming event: `start`

```json
{
  "type": "start",
  "request_id": "c2c9...",
  "created_at": "2026-04-05T18:05:00+00:00",
  "sentence_count": 2,
  "sentence_chunking": true
}
```

### Streaming event: `queue`

```json
{
  "type": "queue",
  "request_id": "c2c9...",
  "queue_position": 0,
  "active_requests": 3,
  "queued_sentences": 11
}
```

Field meaning:

- `queue_position`: current request position in the waiting queue; `0` means the request was admitted into the active pool
- `active_requests`: currently admitted requests in the active pool
- `queued_sentences`: total sentence work items still waiting across queued and active requests

### Streaming event: `batch`

```json
{
  "type": "batch",
  "request_id": "c2c9...",
  "batch_id": "8e91ad4b",
  "sentence_index": 0,
  "batch_size": 4,
  "batch_voice_id": "sonya-a665be06"
}
```

Note:

- batch events are sentence-scoped: one request can receive multiple `batch` events for the same GPU batch if several of its sentences were admitted during the same round-robin fill

### Streaming event: `chunk`

```json
{
  "type": "chunk",
  "request_id": "c2c9...",
  "sentence_index": 0,
  "chunk_index": 0,
  "sample_rate": 24000,
  "pcm16_b64": "....",
  "final_chunk_of_sentence": false,
  "emitted_audio_ms": 500.0,
  "progress_step": 59,
  "preview": true
}
```

Field meaning:

- `pcm16_b64`: base64-encoded little-endian mono PCM16 audio
- `sentence_index`: zero-based sentence order within the request
- `chunk_index`: zero-based chunk order within that sentence
- `preview=true`: progressive prefix chunk emitted before sentence completion
- `preview=false`: final tail chunk emitted after sentence completion
- `final_chunk_of_sentence=true`: marks the last chunk for that sentence
- `emitted_audio_ms`: cumulative emitted audio duration across the whole request, not just the current sentence

### Streaming event: `done`

```json
{
  "type": "done",
  "request_id": "c2c9...",
  "result": {
    "generation_id": "20260405-180500-a1b2c3",
    "audio_url": "/api/assets/generated/20260405-180500-a1b2c3.wav"
  }
}
```

Note:

- `result` uses the same payload shape as `POST /api/v1/synthesize`, including timing fields and `audio_storage`
- if `audio_storage="memory"`, `audio_url` remains usable only while that in-memory WAV is still cached by the current server process

### Streaming event: `error`

```json
{
  "type": "error",
  "request_id": "c2c9...",
  "message": "The synthesis queue is full. Please retry in a moment."
}
```

## OpenAI-Compatible TTS API

These routes exist so external tools such as Open WebUI can talk to this server in an OpenAI-style TTS mode without replacing the native `/api/v1/...` API.

Compatibility notes:

- `voice` may be either a stored `voice_id` or the saved voice name
- `model` is accepted for compatibility, but the server still synthesizes with the currently active TADA model from admin settings
- common client aliases such as `tts-1`, `tts-1-hd`, and `gpt-4o-mini-tts` are accepted by `POST /v1/audio/speech`
- `speed` and `instructions` are accepted for client compatibility, but are currently ignored by the shim
- unknown extra JSON fields are accepted and ignored
- unsupported `response_format` values fall back to WAV and return `X-TADA-Actual-Format: wav`

## `GET /v1/models`

Response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "tada-tts",
      "object": "model",
      "created": 0,
      "owned_by": "tada-local"
    },
    {
      "id": "tada-3b-ml-tts",
      "object": "model",
      "created": 0,
      "owned_by": "tada-local"
    },
    {
      "id": "tada-1b-tts",
      "object": "model",
      "created": 0,
      "owned_by": "tada-local"
    },
    {
      "id": "HumeAI/tada-3b-ml",
      "object": "model",
      "created": 0,
      "owned_by": "tada-local"
    }
  ]
}
```

Note:

- `POST /v1/audio/speech` also accepts compatibility aliases such as `tts-1`, `tts-1-hd`, and `gpt-4o-mini-tts` even though those aliases are not listed in this response

## `GET /v1/voices`

Response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "sonya-a665be06",
      "object": "voice",
      "name": "SONYA",
      "language": "de",
      "preview_url": "/api/assets/voices/sonya-a665be06/reference"
    }
  ]
}
```

Note:

- `preview_url` points to the stored reference WAV, but that asset route is still admin-protected

## `GET /v1/audio/voices`

Same payload shape as `GET /v1/voices`.

## `POST /v1/audio/speech`

Request:

```json
{
  "model": "tts-1",
  "input": "Hallo Welt. Das ist ein Test.",
  "voice": "SONYA",
  "response_format": "wav",
  "speed": 1.0,
  "seed": 88205,
  "temperature": 0.1
}
```

Important notes:

- `input` is mapped to the native `text` field
- `voice` is resolved to a saved TADA voice
- `model` is advisory for compatibility only; synthesis still runs on the admin-selected active model
- `speed` and `instructions` are currently ignored
- `seed`, `temperature`, `top_p`, `cfg_scale`, and similar extra fields are currently ignored by this compatibility layer
- the response is binary audio, not JSON

Typical response:

- status: `200`
- content-type: `audio/wav`
- header: `X-TADA-Active-Model: HumeAI/tada-3b-ml`

Fallback response-format behavior:

- if `response_format` is `pcm` or `s16le`, the response is raw mono PCM16 with `content-type: audio/pcm`
- if `response_format` is unsupported, the response falls back to WAV and includes `X-TADA-Actual-Format: wav`

## Asset Routes

## `GET /api/assets/generated/{file_name}`

Returns the generated WAV file.

Note:

- if `persist_generated_wavs=false`, the file may be served from the in-memory cache of recent generations instead of disk
- the in-memory cache keeps only the latest `100` generated WAVs for the current server process; older memory-backed assets can disappear even though their metadata remains in history

## `GET /api/assets/voices/{voice_id}/reference`

Returns the stored reference WAV for a voice.

Requires:

- `X-Admin-Key`

## Minimal cURL Examples

Create a blocking synthesis request:

```bash
curl -X POST "http://127.0.0.1:7878/api/v1/synthesize" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"Hallo Welt.\",\"voice_id\":\"sonya-a665be06\"}"
```

Start an NDJSON stream:

```bash
curl -N -X POST "http://127.0.0.1:7878/api/v1/synthesize/stream" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"Hallo Welt.\",\"voice_id\":\"sonya-a665be06\"}"
```

OpenAI-compatible TTS request:

```bash
curl -X POST "http://127.0.0.1:7878/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"tts-1\",\"input\":\"Hallo Welt.\",\"voice\":\"sonya-a665be06\",\"response_format\":\"wav\"}" \
  --output tada.wav
```

Query admin settings:

```bash
curl "http://127.0.0.1:7878/api/admin/settings" \
  -H "X-Admin-Key: tada_admin_xxx"
```
