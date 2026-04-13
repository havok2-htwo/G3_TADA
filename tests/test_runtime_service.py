import os
import json
import unittest
import tempfile
from pathlib import Path
from unittest import mock
from urllib import error as urllib_error

import torch

from backend.config_store import ConfigStore
from backend.runtime_service import (
    BatchGenerationItem,
    BENCHMARK_FIXTURES_DIR,
    MAX_REFERENCE_AUDIO_SECONDS,
    MIN_REFERENCE_AUDIO_SECONDS,
    REFERENCE_TAIL_SILENCE_SECONDS,
    TadaRuntimeService,
    load_audio_tensor,
    prepare_reference_prompt_audio,
    resample_audio_tensor,
    save_audio_tensor,
    trim_audio_tensor,
)


class FakePreviewModel:
    def __init__(self, value: float = 1.0):
        self.value = value

    def _decode_wav(self, encoded, time_before):
        sample_count = encoded.shape[0] * 10
        wav = torch.full((1, 1, sample_count), self.value, dtype=torch.float32)
        return wav


class RuntimePreviewTests(unittest.TestCase):
    def test_decode_preview_delta_uses_current_preview_window(self):
        delta = TadaRuntimeService._decode_preview_delta(
            model=FakePreviewModel(),
            encoded=torch.ones(6, 2, dtype=torch.float32),
            time_before=torch.zeros(6, dtype=torch.long),
            sample_rate=1000,
            emitted_samples=0,
            buffer_samples=5,
            min_emit_samples=1,
            input_length=5,
            stream_offset=4,
        )

        self.assertIsNotNone(delta)
        self.assertEqual(int(delta.numel()), 55)

    def test_decode_preview_delta_applies_preview_fade_in(self):
        delta = TadaRuntimeService._decode_preview_delta(
            model=FakePreviewModel(),
            encoded=torch.ones(3, 2, dtype=torch.float32),
            time_before=torch.zeros(3, dtype=torch.long),
            sample_rate=1000,
            emitted_samples=0,
            buffer_samples=0,
            min_emit_samples=1,
            input_length=3,
            stream_offset=2,
        )

        self.assertIsNotNone(delta)
        self.assertAlmostEqual(float(delta[0].item()), 0.0, places=6)
        self.assertGreater(float(delta[-1].item()), 0.9)

    def test_trim_audio_tensor_clamps_to_max_reference_length(self):
        waveform = torch.ones(1, 20_000, dtype=torch.float32)
        trimmed, metadata = trim_audio_tensor(waveform, sample_rate=1000)

        self.assertEqual(int(trimmed.shape[-1]), int(MAX_REFERENCE_AUDIO_SECONDS * 1000))
        self.assertTrue(metadata["was_auto_trimmed"])
        self.assertEqual(metadata["trim_start_ms"], 0)
        self.assertEqual(metadata["trim_end_ms"], int(MAX_REFERENCE_AUDIO_SECONDS * 1000))

    def test_trim_audio_tensor_respects_manual_window(self):
        waveform = torch.arange(0, 10_000, dtype=torch.float32).reshape(1, -1)
        trimmed, metadata = trim_audio_tensor(
            waveform,
            sample_rate=1000,
            trim_start_ms=500,
            trim_end_ms=3750,
        )

        self.assertEqual(int(trimmed.shape[-1]), 3250)
        self.assertEqual(metadata["trim_start_ms"], 500)
        self.assertEqual(metadata["trim_end_ms"], 3750)
        self.assertFalse(metadata["was_auto_trimmed"])

    def test_prepare_reference_prompt_audio_adds_tail_silence(self):
        waveform = torch.ones(1, int(MAX_REFERENCE_AUDIO_SECONDS * 1000), dtype=torch.float32)
        prepared, metadata = prepare_reference_prompt_audio(waveform, sample_rate=1000)

        self.assertEqual(
            int(prepared.shape[-1]),
            int((MAX_REFERENCE_AUDIO_SECONDS + REFERENCE_TAIL_SILENCE_SECONDS) * 1000),
        )
        self.assertEqual(metadata["tail_silence_ms"], int(REFERENCE_TAIL_SILENCE_SECONDS * 1000))
        self.assertTrue(torch.allclose(prepared[:, -500:], torch.zeros(1, 500, dtype=torch.float32)))
        self.assertLess(float(prepared[0, -501].item()), 0.1)

    def test_trim_audio_tensor_rejects_too_short_selection(self):
        waveform = torch.arange(0, 10_000, dtype=torch.float32).reshape(1, -1)
        with self.assertRaises(ValueError):
            trim_audio_tensor(
                waveform,
                sample_rate=1000,
                trim_start_ms=0,
                trim_end_ms=int((MIN_REFERENCE_AUDIO_SECONDS - 0.5) * 1000),
            )

    def test_resample_audio_tensor_updates_length(self):
        waveform = torch.ones(1, 8_000, dtype=torch.float32)
        resampled = resample_audio_tensor(waveform, sample_rate=8000, target_sample_rate=24000)

        self.assertEqual(tuple(resampled.shape), (1, 24_000))

    def test_dispose_model_ignores_cuda_cleanup_runtime_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            runtime = TadaRuntimeService(project_root, ConfigStore(project_root))
            runtime.device = torch.device("cuda")

            with mock.patch("backend.runtime_service.gc.collect"), mock.patch(
                "backend.runtime_service.torch.cuda.empty_cache",
                side_effect=RuntimeError("device-side assert triggered"),
            ), mock.patch("backend.runtime_service.torch.cuda.ipc_collect"):
                runtime._dispose_model(object())

    def test_whisper_request_candidates_cover_openai_and_genesis_shapes(self):
        candidates = TadaRuntimeService._whisper_request_candidates("http://localhost:7861/v1")
        endpoints = [endpoint for endpoint, _ in candidates]

        self.assertIn("http://localhost:7861/v1/audio/transcriptions", endpoints)
        self.assertIn("http://localhost:7861/transcribe/", endpoints)

    def test_transcribe_audio_falls_back_to_genesis_endpoint(self):
        class FakeHttpResponse:
            def __init__(self, payload: dict[str, str]):
                self._payload = payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(self._payload).encode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            config_store = ConfigStore(project_root)
            config_store.update_settings({"whisper_base_url": "http://localhost:7861/v1"})
            runtime = TadaRuntimeService(project_root, config_store)
            upload_path = project_root / "sample.wav"
            save_audio_tensor(upload_path, torch.ones(1, 4000, dtype=torch.float32), 1000)
            attempted_urls: list[str] = []

            def fake_urlopen(request, timeout=120):
                attempted_urls.append(request.full_url)
                if request.full_url.endswith("/v1/audio/transcriptions"):
                    raise urllib_error.HTTPError(
                        request.full_url,
                        405,
                        "Method Not Allowed",
                        hdrs=None,
                        fp=None,
                    )
                if request.full_url.endswith("/audio/transcriptions") or request.full_url.endswith("/v1/transcribe/"):
                    raise urllib_error.HTTPError(
                        request.full_url,
                        404,
                        "Not Found",
                        hdrs=None,
                        fp=None,
                    )
                if request.full_url.endswith("/transcribe/"):
                    return FakeHttpResponse({"transcription": "Hallo Welt"})
                raise AssertionError(f"Unexpected endpoint tried: {request.full_url}")

            with mock.patch("backend.runtime_service.urllib_request.urlopen", side_effect=fake_urlopen):
                payload = runtime.transcribe_audio(upload_path=upload_path)

            self.assertEqual(payload["text"], "Hallo Welt")
            self.assertEqual(attempted_urls[-1], "http://localhost:7861/transcribe/")


class FakeBenchmarkRuntimeService(TadaRuntimeService):
    def __init__(self, project_root: Path, config_store: ConfigStore):
        super().__init__(project_root, config_store)
        self.transcribe_calls: list[str] = []

    def transcribe_audio(self, *, upload_path: Path, trim_start_ms=None, trim_end_ms=None):
        self.transcribe_calls.append(upload_path.name)
        return {
            "text": f"Transcript for {upload_path.stem}",
            "language": "de",
            "trim_start_ms": 0,
            "trim_end_ms": 0,
            "duration_seconds": 0.0,
            "source_duration_seconds": 0.0,
            "was_auto_trimmed": False,
        }


class BenchmarkCorpusTests(unittest.TestCase):
    def setUp(self):
        self._previous_admin = os.environ.get("TADA_ADMIN_KEY")
        os.environ["TADA_ADMIN_KEY"] = "admin-test-key"

    def tearDown(self):
        if self._previous_admin is None:
            os.environ.pop("TADA_ADMIN_KEY", None)
        else:
            os.environ["TADA_ADMIN_KEY"] = self._previous_admin

    def test_build_benchmark_corpus_creates_prepared_fixture_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            source_dir = project_root / "voices"
            source_dir.mkdir(parents=True, exist_ok=True)
            fixture_dir = project_root / BENCHMARK_FIXTURES_DIR
            fixture_dir.mkdir(parents=True, exist_ok=True)

            save_audio_tensor(source_dir / "valid.wav", torch.ones(1, 4000), 1000)
            save_audio_tensor(source_dir / "too_short.wav", torch.ones(1, 2000), 1000)
            save_audio_tensor(source_dir / "too_long.wav", torch.ones(1, 14500), 1000)

            runtime = FakeBenchmarkRuntimeService(project_root, ConfigStore(project_root))
            result = runtime.build_benchmark_corpus(source_dir=source_dir, output_dir=fixture_dir, mode="full")

            self.assertEqual(result["prepared_count"], 1)
            self.assertEqual(result["skipped_count"], 2)
            manifest_path = Path(result["manifest_path"])
            self.assertTrue(manifest_path.exists())
            manifest = manifest_path.read_text(encoding="utf-8")
            self.assertIn("valid.wav", manifest)
            self.assertIn("duration_below_minimum", manifest)
            self.assertIn("duration_at_or_above_limit", manifest)

            valid_entry = next(entry for entry in result["entries"] if entry["source_name"] == "valid.wav")
            prepared_path = project_root / valid_entry["output_path"]
            waveform, sample_rate = load_audio_tensor(prepared_path)
            self.assertEqual(sample_rate, 24000)
            self.assertEqual(int(waveform.shape[-1]), 108000)
            self.assertEqual(runtime.transcribe_calls, ["valid.wav"])

    def test_build_benchmark_corpus_reuses_cached_transcript(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            source_dir = project_root / "voices"
            source_dir.mkdir(parents=True, exist_ok=True)
            fixture_dir = project_root / BENCHMARK_FIXTURES_DIR
            fixture_dir.mkdir(parents=True, exist_ok=True)
            save_audio_tensor(source_dir / "cached.wav", torch.ones(1, 4000), 1000)

            runtime = FakeBenchmarkRuntimeService(project_root, ConfigStore(project_root))
            first = runtime.build_benchmark_corpus(source_dir=source_dir, output_dir=fixture_dir, mode="full")
            self.assertEqual(first["prepared_count"], 1)
            self.assertEqual(runtime.transcribe_calls, ["cached.wav"])

            runtime.transcribe_calls.clear()

            def fail_transcribe(*, upload_path: Path, trim_start_ms=None, trim_end_ms=None):
                raise AssertionError("transcribe_audio should not be called when cache is warm")

            runtime.transcribe_audio = fail_transcribe  # type: ignore[method-assign]
            second = runtime.build_benchmark_corpus(source_dir=source_dir, output_dir=fixture_dir, mode="full")
            self.assertEqual(second["prepared_count"], 1)
            self.assertEqual(runtime.transcribe_calls, [])

    def test_prepare_batch_items_derives_stable_seed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            runtime = FakeBenchmarkRuntimeService(project_root, ConfigStore(project_root))
            runtime.config_store.update_settings({"deterministic_seed": 1234, "model_precision": "fp16"})
            settings = runtime.config_store.get_settings()
            items = [
                BatchGenerationItem(request_id="a", voice_id="voice-a", text="Hallo   Welt", sentence_index=0),
                BatchGenerationItem(request_id="b", voice_id="voice-a", text="Hallo Welt", sentence_index=1),
            ]

            prepared = runtime._prepare_batch_items(items, settings)

            self.assertIsNotNone(prepared[0].derived_seed)
            self.assertEqual(prepared[0].derived_seed, prepared[1].derived_seed)

    def test_save_request_audio_can_keep_wavs_only_in_memory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            config_store = ConfigStore(project_root)
            config_store.update_settings({"persist_generated_wavs": False})
            runtime = FakeBenchmarkRuntimeService(project_root, config_store)

            result = runtime.save_request_audio(
                audio=torch.ones(1, 2400, dtype=torch.float32),
                sample_rate=24000,
                request_text="Hallo Welt",
                voice_id="missing-voice",
                ttft_ms=123.0,
                total_wall_ms=456.0,
                sentence_count=1,
                batch_count=1,
            )

            self.assertEqual(result["audio_storage"], "memory")
            self.assertFalse((project_root / "backend" / "generated" / result["audio_file_name"]).exists())
            asset = runtime.generated_audio_asset(result["audio_file_name"])
            self.assertEqual(asset["kind"], "memory")

            reloaded = FakeBenchmarkRuntimeService(project_root, config_store)
            history = reloaded.list_recent_generations(limit=1)
            self.assertEqual(len(history), 1)
            self.assertIsNone(history[0]["audio_url"])

    def test_save_request_audio_can_persist_wavs_to_disk(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            config_store = ConfigStore(project_root)
            config_store.update_settings({"persist_generated_wavs": True})
            runtime = FakeBenchmarkRuntimeService(project_root, config_store)

            result = runtime.save_request_audio(
                audio=torch.ones(1, 2400, dtype=torch.float32),
                sample_rate=24000,
                request_text="Hallo Welt",
                voice_id="missing-voice",
                ttft_ms=123.0,
                total_wall_ms=456.0,
                sentence_count=1,
                batch_count=1,
            )

            self.assertEqual(result["audio_storage"], "disk")
            self.assertTrue((project_root / "backend" / "generated" / result["audio_file_name"]).exists())
            history = runtime.list_recent_generations(limit=1)
            self.assertEqual(history[0]["audio_url"], result["audio_url"])


if __name__ == "__main__":
    unittest.main()
