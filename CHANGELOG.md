# Changelog

## 2026-04-13

### Added

- settings presets stored under `backend/data/presets/` plus admin routes for list/save/apply
- admin UI controls for preset save/apply and local JSON export/import of runtime settings
- runtime settings for `model_precision`, `deterministic_seed`, `persist_generated_wavs`, `prompt_start_trim_steps`, and VAD trimming parameters
- temporary startup admin key support with TTL-based validation alongside the persistent admin key
- OpenAI-compatible TTS routes for external tools:
  - `GET /v1/models`
  - `GET /v1/voices`
  - `GET /v1/audio/voices`
  - `POST /v1/audio/speech`
- in-memory generated audio asset serving for RAM-only generations plus history hydration for memory-backed entries
- benchmark corpus preparation via `prepare_benchmark_corpus.py`
- Python-side HTTP benchmark and load-test helpers:
  - `run_server_benchmark.py`
  - `run_server_loadtest.py`

### Changed

- the scheduler now builds sentence-level voice-only batches, performs fair round-robin multi-sentence fill, preserves per-request sentence emit order, and exposes richer queue/current-batch metrics
- `Short Merge Max Chars` now also applies to short sentences ending with `.`, and the splitter can cascade/back-merge consecutive short tails more aggressively
- Whisper transcription now probes both OpenAI-style `/audio/transcriptions` and Genesis-style `/transcribe/` endpoints and accepts either `text` or `transcription` in the upstream response
- admin/public payloads now include richer voice/runtime metadata such as trim windows, source duration, tail silence, configured model precision, and effective runtime precision
- public generated audio can stay RAM-only when `persist_generated_wavs=false`; generation metadata remains visible in history even when the WAV itself is no longer persisted to disk
- OpenAI-style TTS requests accept extra compatibility fields and ignore unknown parameters instead of failing
- unsupported OpenAI-style `response_format` values now fall back to WAV for compatibility clients

### Tests

- expanded coverage for presets, startup admin-key recovery, OpenAI-compatible routes, benchmark corpus preparation, memory-vs-disk generation storage, Whisper fallback handling, scheduler fairness/order, and configured release-bundle cache roots

### Documentation

- updated `README.md` to describe the current settings surface, preset workflow, benchmark helpers, and shared installer/smokecheck behavior
- updated `API_DOCUMENTATION.md` with the current preset/admin/public/OpenAI routes, richer voice/generation payloads, and the expanded NDJSON streaming semantics
