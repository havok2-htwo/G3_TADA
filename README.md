# TADA Batch TTS Workbench

Windows-first workbench around HumeAI TADA with:

- FastAPI backend with separate admin and public APIs
- round-robin sentence batching across concurrent requests
- React admin frontend and separate React demo client
- model download/status management for TADA, codec and tokenizer assets
- admin-key protected dashboard plus open public synthesis API
- voice management with optional Whisper transcription
- public streaming API used directly by the demo client

This README is the source of truth for humans and coding agents working in this repo.

## 1. What This Repo Is

This is not a clean upstream TADA checkout. It is a Windows-oriented application wrapper plus local runtime patches around the upstream `hume-tada` package and a server/frontend product layer on top.

Important consequences:

- core TADA code is still vendored under `backend/vendor/tada/...`
- batching behavior depends on local patches in this repo, not only on upstream defaults
- the production server entrypoint is now `backend/server_app.py`
- the old single-page prototype files may still exist, but the supported runtime path is the new admin/public architecture described here

## 2. Current Feature Set

Implemented now:

- persistent server settings in `backend/data/server_settings.json`
- secret storage in `backend/data/server_secrets.json`
- saved settings presets in `backend/data/presets/*.json`
- single admin key workflow with persistent hashed storage and startup recovery key
- admin endpoints under `/api/admin/...`
- public versioned endpoints under `/api/v1/...`
- OpenAI-compatible TTS endpoints under `/v1/...` for tools like Open WebUI
- sentence-level fair-share batching with queueing and concurrency limits
- voice-only batching inside one TADA batch
- progressive preview streaming while the first batch is still running
- dashboard snapshots and a live dashboard stream endpoint
- persistent generation history for the admin dashboard
- optional RAM-only generated audio storage with recent in-memory WAV asset serving
- model download jobs and local model readiness checks
- runtime tuning for model precision, deterministic seeding, sentence merge heuristics, and VAD/prompt trimming
- separate `Admin` and `Demo` React frontends built from the same Vite project
- admin-side preset save/apply plus local JSON export/import of runtime settings
- CLI helpers for benchmark corpus preparation and HTTP benchmarking/load testing
- portable install/start scripts for Windows and Linux source installs
- a Windows fat-bundle packager that creates a copy-and-deploy ZIP

Important behavioral note:

- streaming now emits progressive preview audio while a batch is still in flight
- preview chunks are stable audio prefixes with a safety buffer, not arbitrary rewrites of unfinished waveform tails
- final sentence audio is still stitched and saved from the completed model output, so public clients get low-latency playback without pretending TADA is a fully native token-audio streaming model

## 3. Architecture Overview

The runtime has four layers:

1. Config and secrets
   - `backend/config_store.py`
   - persists runtime settings and secrets
   - creates or rotates the admin key and validates temporary startup admin keys

2. Runtime/model layer
   - `backend/runtime_service.py`
   - loads models and encoders
   - creates voices
   - merges cached prompts into real voice-isolated TADA batches
   - exposes model download and Whisper helper functions

3. Batch scheduler
   - `backend/batch_scheduler.py`
   - owns waiting queue, active request pool and batch assembly
   - enforces round-robin fairness per request/sentence
   - emits NDJSON events for streaming and dashboard updates

4. API and frontends
   - `backend/server_app.py`
   - admin API, public API, asset routes and static frontend serving
   - `frontend/admin.html` + `frontend/src/admin/...`
   - `frontend/demo.html` + `frontend/src/demo/...`

## 4. Batch Scheduler Behavior

The scheduler is intentionally simple and deterministic.

### Request admission

- a public synthesis request becomes one scheduler request
- if sentence chunking is enabled, the text is split into deterministic sentence units
- sentence order within one request is always preserved
- waiting requests are kept in FIFO order

### Active pool

- only `max_parallel_requests` requests are active at once
- active requests first contribute one sentence each to the next batch
- if slots remain, the batch is filled round-robin with additional sentences from those same active requests
- while all active requests are still busy, additional requests wait in the queue

### Round-robin fairness

Example with `max_parallel_requests=16`:

- batch 1 first takes sentence 1 from up to 16 active requests
- if that batch still has free slots after the first fairness pass, it keeps filling with sentence 2, sentence 3, and so on in round-robin order from the same active requests
- batch 2 continues with the remaining queued sentence work items
- if one request finishes early, the next waiting request is promoted and can join the following batch

### Streaming behavior

For each active sentence result:

- preview audio prefixes are emitted during batch inference as soon as a stable prefix exceeds the configured safety buffer
- the remaining tail is emitted when the final sentence waveform is available
- chunk events are still emitted in sentence order for that request
- the client receives `start`, `queue`, `batch`, `chunk`, `done`, and `error` events

## 5. Auth Model

The dashboard is protected by a single admin key, while synthesis stays public.

### Admin

- header: `X-Admin-Key`
- scope: all `/api/admin/...` routes plus `GET /api/assets/voices/{voice_id}/reference`
- used for settings, model downloads, voice management, admin key rotation, dashboard access, and generation history
- stored hashed in `backend/data/server_secrets.json`
- a bootstrap admin key is created automatically if none exists and is printed to the server console once
- `start.bat` and `start.sh` also show a short-lived temporary startup admin key for local recovery right before the server comes up

### Public

- `GET /api/v1/voices` is open
- `POST /api/v1/synthesize` is open
- `POST /api/v1/synthesize/stream` is open
- `GET /v1/models` is open
- `GET /v1/voices` is open
- `GET /v1/audio/voices` is open
- `POST /v1/audio/speech` is open
- `GET /api/assets/generated/{file_name}` is open so public responses remain directly usable

Reference voice audio stays admin-only because it exposes the underlying stored prompt material.

## 6. Settings and Persistence

Persistent settings live in `backend/data/server_settings.json`.

Current settings include:

- `active_model`
- `model_precision`
- `deterministic_seed`
- `persist_generated_wavs`
- `steps`
- `sentence_chunking`
- `short_sentence_merge_max_chars`
- `following_sentence_merge_min_chars`
- `allow_lan_access`
- `stream_start_buffer_ms`
- `stream_chunk_ms`
- `batch_wait_ms`
- `max_batch_size`
- `max_parallel_requests`
- `max_queue_size`
- `model_storage_path`
- `whisper_base_url`
- `vad_trimming`
- `prompt_start_trim_steps`
- `vad_threshold_pct`
- `vad_padding_ms`
- `vad_fade_ms`

Notes:

- `max_parallel_requests` is clamped to `max_batch_size`
- changing `model_storage_path` or `allow_lan_access` is treated as restart-required
- `model_precision` supports `fp16`, `bf16`, `fp32`, `bnb8`, and `fp8`; `bnb8` uses bitsandbytes 8-bit quantization, while `fp8` uses Transformers fine-grained FP8 and requires a CUDA GPU with compute capability >= 9; TADA's audio/diffusion head stays unquantized for compatibility, and quantized modes stream only final stable audio chunks
- `deterministic_seed` is expanded into stable per-sentence seeds based on model, voice and normalized text, so repeated requests can be made reproducible without forcing every sentence to reuse the exact same raw seed
- `persist_generated_wavs=false` keeps recent generated WAVs only in memory for the current server process; generation metadata stays in history either way
- `HF_TOKEN` and other environment values are used to seed first-run defaults, but the persisted config is the supported runtime source afterwards
- `config_store.apply_runtime_environment()` applies the persisted active model, steps and model cache path on startup

## 7. Voice Management

Voices are stored under `backend/data/voices/<voice_id>/`.

Each stored voice contains:

- `reference.wav`
- `prompt_cache.pt`
- `metadata.json`

Admin voice creation flow:

1. upload reference audio
2. trim the usable in/out window in the admin UI before saving
3. optionally transcribe exactly that trimmed window via `/api/admin/voices/transcribe`
4. edit the transcript if needed
5. save the voice via `/api/admin/voices`

For non-English voices the transcript is required.

Important prompt rule:

- spoken reference content is limited to `14.5 s`
- the saved prompt appends `0.5 s` of tail silence to create a cleaner hand-off into generated speech
- short prompts around `6-10 s` are strongly preferred
- the saved `reference.wav` and prompt cache are built from the trimmed window, not from the full upload

## 8. Model Downloads and Hugging Face Access

The admin API reports status and can queue downloads for:

- `HumeAI/tada-3b-ml`
- `HumeAI/tada-1b`
- `HumeAI/tada-codec`
- `meta-llama/Llama-3.2-1B`

Download target:

- the configured `model_storage_path`

The Hugging Face token can be stored through the admin settings UI/API.

## 9. Public API Surface

For a dedicated endpoint-by-endpoint integration reference, see [API_DOCUMENTATION.md](API_DOCUMENTATION.md).

### Headers

- admin routes: `X-Admin-Key`
- public routes: none

### Diagnostic route

- `GET /api/health`

### Admin routes

- `GET /api/admin/settings`
- `PUT /api/admin/settings`
- `GET /api/admin/settings/presets`
- `POST /api/admin/settings/presets/save`
- `POST /api/admin/settings/presets/apply`
- `GET /api/admin/models`
- `POST /api/admin/models/download`
- `GET /api/admin/voices`
- `POST /api/admin/voices`
- `DELETE /api/admin/voices/{voice_id}`
- `POST /api/admin/voices/transcribe`
- `GET /api/admin/keys`
- `POST /api/admin/keys`
- `GET /api/admin/generations`
- `GET /api/admin/dashboard/stream`

### Public routes

- legacy public routes used by the built-in demo and local tools:
- `GET /api/v1/voices`
- `POST /api/v1/synthesize`
- `POST /api/v1/synthesize/stream`
- `GET /api/assets/generated/{file_name}`

- OpenAI-compatible TTS routes used by external tools such as Open WebUI:
- `GET /v1/models`
- `GET /v1/voices`
- `GET /v1/audio/voices`
- `POST /v1/audio/speech`

OpenAI compatibility notes:

- the old `/api/v1/...` routes remain supported and are not deprecated
- `POST /v1/audio/speech` accepts the usual OpenAI-style fields `input`, `voice`, `model`, `response_format`, and `speed`
- additional unknown JSON fields are accepted and ignored for compatibility
- `voice` may be either a stored `voice_id` or the saved voice name
- `model` is accepted for compatibility, but synthesis still uses the server's currently active TADA model from the admin settings
- unsupported response formats fall back to WAV and return `X-TADA-Actual-Format: wav`

### Admin-only asset routes

- `GET /api/assets/voices/{voice_id}/reference`

### Streaming event types

The public stream is NDJSON and currently emits:

- `start`
- `queue`
- `batch`
- `chunk`
- `done`
- `error`

Chunk events contain at least:

- `sentence_index`
- `chunk_index`
- `sample_rate`
- `pcm16_b64`
- `final_chunk_of_sentence`
- `emitted_audio_ms`
- `progress_step`
- `preview`

## 10. Frontends

### Admin frontend

URL:

- `/admin`

Responsibilities:

- enter or rotate the admin key
- manage settings and secrets
- save/apply named presets from `backend/data/presets`
- export/import runtime settings as local JSON files
- watch queue/dashboard metrics
- inspect the latest generation history
- trigger model downloads
- create voices

### Demo frontend

URL:

- `/demo`

Responsibilities:

- run a live single-request stream against the public API
- run multi-request benchmarks against the same public API
- choose a local output folder via the File System Access API
- save `request-XXXX.wav` files and `metrics.txt` client-side

Browser note:

- local folder saving requires a Chromium-class browser with File System Access API support

### CLI benchmark helpers

Useful repo-local scripts outside the browser UI:

- `prepare_benchmark_corpus.py`: scans raw audio from `voices/`, keeps only usable reference clips, resamples them to `24 kHz`, adds prompt-tail silence, optionally transcribes them, and writes a manifest under `tests/fixtures/benchmark_voices/`
- `run_server_benchmark.py`: launches a fixed-size parallel HTTP benchmark against `/api/v1/synthesize/stream` and writes JSON results to `benchmark_results/server_http/`
- `run_server_loadtest.py`: generates a deterministic multi-user arrival pattern for a longer HTTP load test and writes per-request, per-user and health snapshots to `benchmark_results/server_loadtest/`

The older `run_benchmark_1b.py` and `run_benchmark_3b.py` scripts still exist for direct raw-model experiments, but the supported product path is the server/runtime stack above.

## 11. Important Files

Most relevant files after the refactor:

- `backend/server_app.py`: FastAPI app, route layer and static frontend serving
- `backend/runtime_service.py`: runtime service, voice creation, batch generation, model downloads
- `backend/batch_scheduler.py`: request queue, fairness, streaming and dashboard metrics
- `backend/config_store.py`: persistent settings, secrets and admin-key management
- `backend/prompt_batch.py`: sentence splitting, waveform chunking and prompt merging
- `backend/vendor/tada/modules/tada.py`: patched to support true heterogeneous prompt batching
- `frontend/admin.html`: admin entry page
- `frontend/demo.html`: demo client entry page
- `frontend/src/admin/AdminApp.jsx`: admin React app
- `frontend/src/demo/DemoApp.jsx`: demo React app
- `install.bat` / `start.bat`: Windows source and bundle workflow
- `install.sh` / `start.sh`: Linux source workflow
- `package_release.bat`: builds the Windows fat release ZIP

## 12. Development and Verification

Validated in this repo now:

- `python -m unittest discover -s tests`
- `cd frontend && npm run build`

Current automated tests cover:

- config persistence and admin-key behavior
- temporary startup admin-key acceptance and expiry behavior
- settings presets plus admin settings precision/seed/persistence fields
- prompt batch merging
- short-sentence merging rules including short `.` sentences and cascading short tails
- round-robin scheduler ordering with voice-only batch isolation
- progressive preview chunk emission without duplicated samples
- in-memory versus disk-backed generated audio history/assets
- Whisper endpoint fallback handling for OpenAI-style and Genesis-style STT servers
- benchmark corpus preparation and transcript cache reuse
- basic admin/public API auth wiring, preset routes, OpenAI-compatible TTS routes, and generation history
- release bundle model-cache copying from the configured storage root

## 13. Quick Start

### Windows source install

1. Run:

```bat
install.bat
start.bat
```

2. Open one of these URLs:

```text
http://127.0.0.1:7878/admin
http://127.0.0.1:7878/demo
```

Notes:

- `install.bat` is conda-first and creates a project-local `.conda-env/` by default
- on Windows the installer prefers a pinned NVIDIA CUDA PyTorch stack from `https://download.pytorch.org/whl/cu128`
- if direct `pip` writes into the managed environment fail, the installer falls back automatically to project-local `.python_packages/` and records that in `.runtime_package_mode`
- the shared installer runs a runtime smokecheck at the end and retries the pinned core runtime once if that import validation still fails
- if `frontend/dist` already exists, npm is not required on the target machine
- if `wheelhouse/` exists, the installer uses it as an offline package source
- `TADA_CONDA_EXE`, `TADA_CONDA_ENV_DIR`, `TADA_CONDA_PYTHON_VERSION`, `TADA_TORCH_INDEX_URL` and `TADA_PYTHON` can override the default install behavior
- `start.bat` prints a temporary startup admin key before launch; the dashboard itself is then opened at `/admin`
- the Admin settings page can enable `LAN Access`; after saving, restart the server and then use `http://<server-ip>:7878/admin` or `http://<server-ip>:7878/demo` from other devices in the same network

### Linux source install

```bash
./install.sh
./start.sh
```

Linux notes:

- `install.sh` prefers a local `.venv/`, reuses `env/` if present, and otherwise falls back to a suitable system `python3`/`python`
- it uses the same shared `tools/install_runtime.py` workflow as Windows, including wheelhouse/offline support, frontend validation/build, vendor checks, and the final runtime smokecheck

### Windows fat bundle deploy

1. Build the bundle on the source machine:

```bat
package_release.bat
```

2. Copy the generated ZIP from `releases/` to the target Windows PC.
3. Extract it.
4. Run:

```bat
install.bat
start.bat
```

Bundle notes:

- the bundle contains a `wheelhouse/`, built frontend assets, vendored backend code, voices and cached model snapshots
- the bundle manifest validates Windows platform, architecture and Python minor version before install
- by default the installer only forces the bundle wheel set if modules are actually missing or a final smokecheck fails
- set `TADA_STRICT_BUNDLE_RUNTIME=1` if you explicitly want to force the bundle's exact torch package versions onto the selected interpreter

## 14. Known Limitations

Current v1 limitations you should keep in mind:

- sentence chunking is regex-based, not full linguistic segmentation
- preview streaming is stable-prefix based; it is not a full waveform rewrite protocol for every unfinished tail
- model downloads are background jobs but there is no pause/cancel management yet
- the Windows fat bundle is platform-specific and should be built on a compatible Windows x64 machine
- the current release bundle workflow assumes Miniconda/Anaconda is available on the target Windows machine; it does not ship an embedded Python runtime
- the demo benchmark uses browser-side parallel fetches and client-side WAV saving, so browser limits can still affect very high request counts
- the old prototype files may still exist in the repo, but the supported product path is the new admin/public stack

## 15. External References

- https://github.com/havok2-htwo/G3_TADA
- https://huggingface.co/HumeAI/tada-3b-ml
- https://huggingface.co/HumeAI/tada-1b
- https://huggingface.co/HumeAI/tada-codec
- https://huggingface.co/meta-llama/Llama-3.2-1B
- https://github.com/HumeAI/tada
