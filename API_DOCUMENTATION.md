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

Public client routes require:

- header: `X-API-Key: <client token>`

Protected asset routes accept either:

- `X-Admin-Key`
- `X-API-Key`

## Common Response Rules

- `200` on success
- `400` for bad input
- `401` for missing or invalid auth
- `404` if the requested voice, file, or key does not exist
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
    "hf_token_present": true,
    "whisper_api_key_present": false,
    "restart_required": false
  },
  "runtime": {},
  "dashboard": {}
}
```

## Admin API

## `GET /api/admin/settings`

Returns:

- `settings`: persisted server settings plus secret-presence flags
- `runtime`: current runtime status
- `models`: model download/status snapshot

## `PUT /api/admin/settings`

Request body:

```json
{
  "active_model": "HumeAI/tada-3b-ml",
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
  "hf_token": "",
  "whisper_api_key": ""
}
```

Important notes:

- `steps` is clamped to `1..128`
- `max_parallel_requests` is clamped to `max_batch_size`
- `short_sentence_merge_max_chars` is clamped to `0..200`
- `following_sentence_merge_min_chars` is clamped to `0..500`
- changing `model_storage_path` or `allow_lan_access` sets `restart_required=true`
- `hf_token` and `whisper_api_key` are write-only convenience fields

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
    "voice_id": "new-voice-1234abcd"
  }
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
  "base_url": "http://127.0.0.1:8001/v1",
  "trim_start_ms": 0,
  "trim_end_ms": 10000,
  "duration_seconds": 10.0,
  "source_duration_seconds": 21.37,
  "was_auto_trimmed": false
}
```

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
  },
  "client_keys": [
    {
      "id": "0f7c1f4a9d0d8e6b",
      "label": "Demo Client",
      "created_at": "2026-04-05T12:01:00+00:00",
      "last_used_at": null
    }
  ]
}
```

## `POST /api/admin/keys`

Request:

```json
{
  "label": "Demo Client",
  "kind": "client"
}
```

Response:

```json
{
  "key": {
    "kind": "client",
    "id": "0f7c1f4a9d0d8e6b",
    "label": "Demo Client",
    "token": "tada_client_xxx",
    "created_at": "2026-04-05T12:01:00+00:00"
  },
  "keys": {}
}
```

Important:

- `token` is only returned once

## `DELETE /api/admin/keys/{key_id}`

Response:

```json
{
  "ok": true,
  "keys": {}
}
```

## `GET /api/admin/dashboard/stream`

Streaming response type:

- `application/x-ndjson`

One JSON object per line, roughly once per second.

Snapshot fields:

- `timestamp`
- `queue_length`
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

## Public API

## `GET /api/v1/voices`

Returns the public voice list for authenticated client callers.

Example:

```json
{
  "voices": [
    {
      "voice_id": "sonya-a665be06",
      "name": "SONYA",
      "language": "de",
      "duration_seconds": 3.27,
      "reference_url": "/api/assets/voices/sonya-a665be06/reference"
    }
  ]
}
```

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
  "audio_url": "/api/assets/generated/20260405-180500-a1b2c3.wav",
  "audio_file_name": "20260405-180500-a1b2c3.wav",
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
  "active_requests": 3
}
```

### Streaming event: `batch`

```json
{
  "type": "batch",
  "request_id": "c2c9...",
  "batch_id": "8e91ad4b",
  "sentence_index": 0,
  "batch_size": 4
}
```

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

### Streaming event: `error`

```json
{
  "type": "error",
  "request_id": "c2c9...",
  "message": "The synthesis queue is full. Please retry in a moment."
}
```

## Asset Routes

## `GET /api/assets/generated/{file_name}`

Returns the generated WAV file.

Requires:

- `X-Admin-Key`, or
- `X-API-Key`

## `GET /api/assets/voices/{voice_id}/reference`

Returns the stored reference WAV for a voice.

Requires:

- `X-Admin-Key`, or
- `X-API-Key`

## Minimal cURL Examples

Create a blocking synthesis request:

```bash
curl -X POST "http://127.0.0.1:7878/api/v1/synthesize" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: tada_client_xxx" \
  -d "{\"text\":\"Hallo Welt.\",\"voice_id\":\"sonya-a665be06\"}"
```

Start an NDJSON stream:

```bash
curl -N -X POST "http://127.0.0.1:7878/api/v1/synthesize/stream" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: tada_client_xxx" \
  -d "{\"text\":\"Hallo Welt.\",\"voice_id\":\"sonya-a665be06\"}"
```

Query admin settings:

```bash
curl "http://127.0.0.1:7878/api/admin/settings" \
  -H "X-Admin-Key: tada_admin_xxx"
```
