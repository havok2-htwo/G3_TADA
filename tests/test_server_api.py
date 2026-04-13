from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.config_store import ConfigStore
from backend.runtime_service import DuplicateVoiceNameError


class FakeRuntimeService:
    def __init__(self):
        temp_root = Path(tempfile.gettempdir())
        self.settings_saved = None
        self.voices = {
            "demo-voice": {
                "voice_id": "demo-voice",
                "name": "Demo",
                "language": "de",
                "reference_url": "/api/assets/voices/demo-voice/reference",
            }
        }
        self.generated_file = temp_root / "tada-generated-test.wav"
        self.reference_file = temp_root / "tada-reference-test.wav"
        self.generated_file.write_bytes(b"RIFFdemo")
        self.reference_file.write_bytes(b"RIFFdemo")
        self.generations = [
            {
                "generation_id": "gen-001",
                "created_at": "2026-04-10T10:00:00+00:00",
                "voice_id": "demo-voice",
                "voice_name": "Demo",
                "text": "Hallo Welt",
                "duration_seconds": 1.5,
                "ttft_ms": 120.0,
                "total_wall_ms": 420.0,
                "batch_count": 1,
                "audio_url": "/api/assets/generated/tada-generated-test.wav",
            }
        ]

    def status(self):
        return {
            "loaded_model_name": "HumeAI/tada-3b-ml",
            "available_models": [{"id": "HumeAI/tada-3b-ml", "label": "3B", "description": "x"}],
            "voices_count": len(self.voices),
        }

    def model_statuses(self):
        return [{"id": "HumeAI/tada-3b-ml", "label": "3B", "status": "ready", "storage_root": "x"}]

    def sync_runtime_settings(self):
        self.settings_saved = True

    def list_voices(self):
        return list(self.voices.values())

    def get_voice(self, voice_id):
        if voice_id not in self.voices:
            raise FileNotFoundError("missing")
        return self.voices[voice_id]

    def create_voice(self, *, name, language, transcript, upload_path, trim_start_ms=None, trim_end_ms=None, overwrite_existing=False):
        cleaned = " ".join((name or "").strip().split()) or "Meine Stimme"
        for existing in self.voices.values():
            if existing["name"].casefold() == cleaned.casefold():
                if not overwrite_existing:
                    raise DuplicateVoiceNameError(
                        cleaned_name=cleaned,
                        existing_voice_id=existing["voice_id"],
                        existing_voice_name=existing["name"],
                    )
                voice_id = existing["voice_id"]
                break
        else:
            voice_id = f"{cleaned.lower().replace(' ', '-')}-001"
        voice = {
            "voice_id": voice_id,
            "name": cleaned,
            "language": language,
            "transcript": transcript,
            "reference_url": f"/api/assets/voices/{voice_id}/reference",
            "trim_start_ms": trim_start_ms or 0,
            "trim_end_ms": trim_end_ms or 10000,
            "duration_seconds": 10.0,
        }
        self.voices[voice_id] = voice
        return voice

    def delete_voice(self, voice_id):
        if voice_id not in self.voices:
            raise FileNotFoundError("missing")
        del self.voices[voice_id]

    def transcribe_audio(self, *, upload_path, trim_start_ms=None, trim_end_ms=None):
        return {
            "text": "Hallo Welt",
            "trim_start_ms": trim_start_ms or 0,
            "trim_end_ms": trim_end_ms or 10000,
            "duration_seconds": 10.0,
            "source_duration_seconds": 21.0,
            "was_auto_trimmed": False,
        }

    def list_recent_generations(self, *, limit=60):
        return self.generations[:limit]

    def generated_audio_path(self, file_name):
        return self.generated_file

    def generated_audio_asset(self, file_name):
        return {"kind": "file", "path": self.generated_file}

    def reference_audio_path(self, voice_id):
        return self.reference_file


class FakeScheduler:
    def dashboard_snapshot(self):
        return {"queue_length": 0, "history": []}

    def iter_dashboard_events(self):
        yield {"queue_length": 0, "history": []}

    def submit_request(self, *, text, voice_id):
        return {"text": text, "voice_id": voice_id}

    def wait_for_result(self, state, timeout=None):
        return {
            "generation_id": "fake",
            "text": state["text"],
            "voice_id": state["voice_id"],
            "audio_url": "/api/assets/generated/tada-generated-test.wav",
        }

    def iter_request_events(self, state):
        yield {"type": "start", "request_id": "fake", "sentence_count": 1}
        yield {"type": "done", "request_id": "fake", "result": {"generation_id": "fake"}}


class ServerApiTests(unittest.TestCase):
    def test_admin_routes_accept_temporary_startup_admin_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_admin = os.environ.get("TADA_ADMIN_KEY")
            previous_startup = os.environ.get("TADA_STARTUP_ADMIN_KEY")
            previous_ttl = os.environ.get("TADA_STARTUP_ADMIN_KEY_TTL_SECONDS")
            previous_warmup = os.environ.get("TADA_DISABLE_WARMUP")
            os.environ["TADA_ADMIN_KEY"] = "admin-test-key"
            os.environ["TADA_STARTUP_ADMIN_KEY"] = "startup-admin-test-key"
            os.environ["TADA_STARTUP_ADMIN_KEY_TTL_SECONDS"] = "60"
            os.environ["TADA_DISABLE_WARMUP"] = "1"
            try:
                module = importlib.import_module("backend.server_app")
                module.config_store = ConfigStore(Path(temp_dir))
                module.runtime_service = FakeRuntimeService()
                module.scheduler = FakeScheduler()

                with TestClient(module.app) as client:
                    admin_response = client.get(
                        "/api/admin/settings",
                        headers={"X-Admin-Key": "startup-admin-test-key"},
                    )
                    self.assertEqual(admin_response.status_code, 200)
            finally:
                if previous_admin is None:
                    os.environ.pop("TADA_ADMIN_KEY", None)
                else:
                    os.environ["TADA_ADMIN_KEY"] = previous_admin
                if previous_startup is None:
                    os.environ.pop("TADA_STARTUP_ADMIN_KEY", None)
                else:
                    os.environ["TADA_STARTUP_ADMIN_KEY"] = previous_startup
                if previous_ttl is None:
                    os.environ.pop("TADA_STARTUP_ADMIN_KEY_TTL_SECONDS", None)
                else:
                    os.environ["TADA_STARTUP_ADMIN_KEY_TTL_SECONDS"] = previous_ttl
                if previous_warmup is None:
                    os.environ.pop("TADA_DISABLE_WARMUP", None)
                else:
                    os.environ["TADA_DISABLE_WARMUP"] = previous_warmup

    def test_admin_routes_require_key_but_public_routes_are_open(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_admin = os.environ.get("TADA_ADMIN_KEY")
            previous_warmup = os.environ.get("TADA_DISABLE_WARMUP")
            os.environ["TADA_ADMIN_KEY"] = "admin-test-key"
            os.environ["TADA_DISABLE_WARMUP"] = "1"
            try:
                module = importlib.import_module("backend.server_app")
                module.config_store = ConfigStore(Path(temp_dir))
                admin_token = module.config_store.rotate_admin_key(label="Master")["token"]
                module.runtime_service = FakeRuntimeService()
                module.scheduler = FakeScheduler()

                with TestClient(module.app) as client:
                    missing_admin = client.get("/api/admin/settings")
                    self.assertEqual(missing_admin.status_code, 401)
                    admin_response = client.get("/api/admin/settings", headers={"X-Admin-Key": admin_token})
                    self.assertEqual(admin_response.status_code, 200)
                    public_response = client.get("/api/v1/voices")
                    self.assertEqual(public_response.status_code, 200)
                    synth_response = client.post(
                        "/api/v1/synthesize",
                        json={"text": "Hallo", "voice_id": "demo-voice"},
                    )
                    self.assertEqual(synth_response.status_code, 200)
                    self.assertEqual(synth_response.json()["generation_id"], "fake")
                    asset_response = client.get("/api/assets/generated/tada-generated-test.wav")
                    self.assertEqual(asset_response.status_code, 200)
                    reference_denied = client.get("/api/assets/voices/demo-voice/reference")
                    self.assertEqual(reference_denied.status_code, 401)
                    reference_allowed = client.get(
                        "/api/assets/voices/demo-voice/reference",
                        headers={"X-Admin-Key": admin_token},
                    )
                    self.assertEqual(reference_allowed.status_code, 200)
            finally:
                if previous_admin is None:
                    os.environ.pop("TADA_ADMIN_KEY", None)
                else:
                    os.environ["TADA_ADMIN_KEY"] = previous_admin
                if previous_warmup is None:
                    os.environ.pop("TADA_DISABLE_WARMUP", None)
                else:
                    os.environ["TADA_DISABLE_WARMUP"] = previous_warmup

    def test_openai_compatible_tts_routes_are_available(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_admin = os.environ.get("TADA_ADMIN_KEY")
            previous_warmup = os.environ.get("TADA_DISABLE_WARMUP")
            os.environ["TADA_ADMIN_KEY"] = "admin-test-key"
            os.environ["TADA_DISABLE_WARMUP"] = "1"
            try:
                module = importlib.import_module("backend.server_app")
                module.config_store = ConfigStore(Path(temp_dir))
                module.runtime_service = FakeRuntimeService()
                module.scheduler = FakeScheduler()

                with TestClient(module.app) as client:
                    models_response = client.get("/v1/models")
                    self.assertEqual(models_response.status_code, 200)
                    model_ids = {item["id"] for item in models_response.json()["data"]}
                    self.assertIn("tada-tts", model_ids)

                    voices_response = client.get("/v1/audio/voices")
                    self.assertEqual(voices_response.status_code, 200)
                    self.assertEqual(voices_response.json()["data"][0]["id"], "demo-voice")

                    speech_response = client.post(
                        "/v1/audio/speech",
                        json={
                            "input": "Hallo",
                            "voice": "Demo",
                            "model": "tts-1",
                            "response_format": "mp3",
                            "seed": 88205,
                            "temperature": 0.1,
                        },
                    )
                    self.assertEqual(speech_response.status_code, 200)
                    self.assertEqual(speech_response.headers["content-type"], "audio/wav")
                    self.assertEqual(speech_response.headers["x-tada-actual-format"], "wav")
                    self.assertEqual(speech_response.content, b"RIFFdemo")
            finally:
                if previous_admin is None:
                    os.environ.pop("TADA_ADMIN_KEY", None)
                else:
                    os.environ["TADA_ADMIN_KEY"] = previous_admin
                if previous_warmup is None:
                    os.environ.pop("TADA_DISABLE_WARMUP", None)
                else:
                    os.environ["TADA_DISABLE_WARMUP"] = previous_warmup

    def test_admin_keys_rotation_and_generation_history_routes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_admin = os.environ.get("TADA_ADMIN_KEY")
            previous_warmup = os.environ.get("TADA_DISABLE_WARMUP")
            os.environ["TADA_ADMIN_KEY"] = "admin-test-key"
            os.environ["TADA_DISABLE_WARMUP"] = "1"
            try:
                module = importlib.import_module("backend.server_app")
                module.config_store = ConfigStore(Path(temp_dir))
                admin_token = module.config_store.rotate_admin_key(label="Master")["token"]
                module.runtime_service = FakeRuntimeService()
                module.scheduler = FakeScheduler()

                with TestClient(module.app) as client:
                    list_response = client.get("/api/admin/keys", headers={"X-Admin-Key": admin_token})
                    self.assertEqual(list_response.status_code, 200)
                    self.assertEqual(list_response.json()["admin_key"]["label"], "Master")

                    rotate_response = client.post("/api/admin/keys", headers={"X-Admin-Key": admin_token})
                    self.assertEqual(rotate_response.status_code, 200)
                    rotated_token = rotate_response.json()["key"]["token"]
                    self.assertTrue(rotated_token.startswith("tada_admin_"))

                    history_response = client.get("/api/admin/generations", headers={"X-Admin-Key": rotated_token})
                    self.assertEqual(history_response.status_code, 200)
                    self.assertEqual(history_response.json()["generations"][0]["generation_id"], "gen-001")
            finally:
                if previous_admin is None:
                    os.environ.pop("TADA_ADMIN_KEY", None)
                else:
                    os.environ["TADA_ADMIN_KEY"] = previous_admin
                if previous_warmup is None:
                    os.environ.pop("TADA_DISABLE_WARMUP", None)
                else:
                    os.environ["TADA_DISABLE_WARMUP"] = previous_warmup

    def test_admin_voice_routes_reject_duplicate_names_and_allow_delete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_admin = os.environ.get("TADA_ADMIN_KEY")
            previous_warmup = os.environ.get("TADA_DISABLE_WARMUP")
            os.environ["TADA_ADMIN_KEY"] = "admin-test-key"
            os.environ["TADA_DISABLE_WARMUP"] = "1"
            try:
                module = importlib.import_module("backend.server_app")
                module.config_store = ConfigStore(Path(temp_dir))
                admin_token = module.config_store.rotate_admin_key(label="Master")["token"]
                module.runtime_service = FakeRuntimeService()
                module.scheduler = FakeScheduler()

                with TestClient(module.app) as client:
                    first_create = client.post(
                        "/api/admin/voices",
                        headers={"X-Admin-Key": admin_token},
                        files={"audio": ("ref.wav", b"RIFFdemo", "audio/wav")},
                        data={"name": "Fresh Voice", "language": "de", "transcript": "Hallo Welt"},
                    )
                    self.assertEqual(first_create.status_code, 200)
                    created_voice_id = first_create.json()["voice"]["voice_id"]

                    duplicate_create = client.post(
                        "/api/admin/voices",
                        headers={"X-Admin-Key": admin_token},
                        files={"audio": ("ref.wav", b"RIFFdemo", "audio/wav")},
                        data={"name": "fresh   voice", "language": "de", "transcript": "Hallo Welt"},
                    )
                    self.assertEqual(duplicate_create.status_code, 409)
                    self.assertEqual(duplicate_create.json()["code"], "voice_name_exists")
                    self.assertIn("already exists", duplicate_create.json()["message"])

                    overwrite_create = client.post(
                        "/api/admin/voices",
                        headers={"X-Admin-Key": admin_token},
                        files={"audio": ("ref.wav", b"RIFFdemo", "audio/wav")},
                        data={
                            "name": "fresh   voice",
                            "language": "de",
                            "transcript": "Hallo Welt",
                            "overwrite_existing": "true",
                        },
                    )
                    self.assertEqual(overwrite_create.status_code, 200)
                    overwritten_voice_id = overwrite_create.json()["voice"]["voice_id"]
                    self.assertEqual(overwritten_voice_id, created_voice_id)

                    delete_response = client.delete(
                        f"/api/admin/voices/{created_voice_id}",
                        headers={"X-Admin-Key": admin_token},
                    )
                    self.assertEqual(delete_response.status_code, 200)
                    remaining_voice_ids = {voice["voice_id"] for voice in delete_response.json()["voices"]}
                    self.assertNotIn(created_voice_id, remaining_voice_ids)
            finally:
                if previous_admin is None:
                    os.environ.pop("TADA_ADMIN_KEY", None)
                else:
                    os.environ["TADA_ADMIN_KEY"] = previous_admin
                if previous_warmup is None:
                    os.environ.pop("TADA_DISABLE_WARMUP", None)
                else:
                    os.environ["TADA_DISABLE_WARMUP"] = previous_warmup

    def test_admin_voice_routes_forward_trim_window(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_admin = os.environ.get("TADA_ADMIN_KEY")
            previous_warmup = os.environ.get("TADA_DISABLE_WARMUP")
            os.environ["TADA_ADMIN_KEY"] = "admin-test-key"
            os.environ["TADA_DISABLE_WARMUP"] = "1"
            try:
                module = importlib.import_module("backend.server_app")
                module.config_store = ConfigStore(Path(temp_dir))
                admin_token = module.config_store.rotate_admin_key(label="Master")["token"]
                module.runtime_service = FakeRuntimeService()
                module.scheduler = FakeScheduler()

                with TestClient(module.app) as client:
                    create_response = client.post(
                        "/api/admin/voices",
                        headers={"X-Admin-Key": admin_token},
                        files={"audio": ("ref.wav", b"RIFFdemo", "audio/wav")},
                        data={
                            "name": "Trimmed Voice",
                            "language": "de",
                            "transcript": "Hallo Welt",
                            "trim_start_ms": "500",
                            "trim_end_ms": "9500",
                        },
                    )
                    self.assertEqual(create_response.status_code, 200)
                    payload = create_response.json()["voice"]
                    self.assertEqual(payload["trim_start_ms"], 500)
                    self.assertEqual(payload["trim_end_ms"], 9500)

                    transcribe_response = client.post(
                        "/api/admin/voices/transcribe",
                        headers={"X-Admin-Key": admin_token},
                        files={"audio": ("ref.wav", b"RIFFdemo", "audio/wav")},
                        data={"trim_start_ms": "250", "trim_end_ms": "7250"},
                    )
                    self.assertEqual(transcribe_response.status_code, 200)
                    transcribe_payload = transcribe_response.json()
                    self.assertEqual(transcribe_payload["trim_start_ms"], 250)
                    self.assertEqual(transcribe_payload["trim_end_ms"], 7250)
            finally:
                if previous_admin is None:
                    os.environ.pop("TADA_ADMIN_KEY", None)
                else:
                    os.environ["TADA_ADMIN_KEY"] = previous_admin
                if previous_warmup is None:
                    os.environ.pop("TADA_DISABLE_WARMUP", None)
                else:
                    os.environ["TADA_DISABLE_WARMUP"] = previous_warmup

    def test_admin_settings_route_accepts_precision_and_seed_updates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_admin = os.environ.get("TADA_ADMIN_KEY")
            previous_warmup = os.environ.get("TADA_DISABLE_WARMUP")
            os.environ["TADA_ADMIN_KEY"] = "admin-test-key"
            os.environ["TADA_DISABLE_WARMUP"] = "1"
            try:
                module = importlib.import_module("backend.server_app")
                module.config_store = ConfigStore(Path(temp_dir))
                admin_token = module.config_store.rotate_admin_key(label="Master")["token"]
                module.runtime_service = FakeRuntimeService()
                module.scheduler = FakeScheduler()

                with TestClient(module.app) as client:
                    response = client.put(
                        "/api/admin/settings",
                        headers={"X-Admin-Key": admin_token},
                        json={
                            "model_precision": "fp32",
                            "deterministic_seed": 4242,
                            "persist_generated_wavs": True,
                            "prompt_start_trim_steps": 3,
                        },
                    )
                    self.assertEqual(response.status_code, 200)
                    payload = response.json()
                    self.assertEqual(payload["settings"]["model_precision"], "fp32")
                    self.assertEqual(payload["settings"]["deterministic_seed"], 4242)
                    self.assertTrue(payload["settings"]["persist_generated_wavs"])
                    self.assertEqual(payload["settings"]["prompt_start_trim_steps"], 3)
                    self.assertTrue(module.runtime_service.settings_saved)
            finally:
                if previous_admin is None:
                    os.environ.pop("TADA_ADMIN_KEY", None)
                else:
                    os.environ["TADA_ADMIN_KEY"] = previous_admin
                if previous_warmup is None:
                    os.environ.pop("TADA_DISABLE_WARMUP", None)
                else:
                    os.environ["TADA_DISABLE_WARMUP"] = previous_warmup

    def test_admin_settings_presets_can_be_listed_saved_and_applied(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_admin = os.environ.get("TADA_ADMIN_KEY")
            previous_warmup = os.environ.get("TADA_DISABLE_WARMUP")
            os.environ["TADA_ADMIN_KEY"] = "admin-test-key"
            os.environ["TADA_DISABLE_WARMUP"] = "1"
            try:
                module = importlib.import_module("backend.server_app")
                module.config_store = ConfigStore(Path(temp_dir))
                admin_token = module.config_store.rotate_admin_key(label="Master")["token"]
                module.runtime_service = FakeRuntimeService()
                module.scheduler = FakeScheduler()

                with TestClient(module.app) as client:
                    initial_precision = module.config_store.get_settings().model_precision
                    save_response = client.post(
                        "/api/admin/settings/presets/save",
                        headers={"X-Admin-Key": admin_token},
                        json={"name": "Test Preset"},
                    )
                    self.assertEqual(save_response.status_code, 200)
                    self.assertEqual(save_response.json()["preset"]["name"], "test-preset")

                    list_response = client.get(
                        "/api/admin/settings/presets",
                        headers={"X-Admin-Key": admin_token},
                    )
                    self.assertEqual(list_response.status_code, 200)
                    self.assertEqual(len(list_response.json()["presets"]), 1)

                    module.config_store.update_settings({"model_precision": "fp32"})
                    apply_response = client.post(
                        "/api/admin/settings/presets/apply",
                        headers={"X-Admin-Key": admin_token},
                        json={"name": "test-preset"},
                    )
                    self.assertEqual(apply_response.status_code, 200)
                    self.assertEqual(apply_response.json()["settings"]["model_precision"], initial_precision)
                    self.assertTrue(module.runtime_service.settings_saved)
            finally:
                if previous_admin is None:
                    os.environ.pop("TADA_ADMIN_KEY", None)
                else:
                    os.environ["TADA_ADMIN_KEY"] = previous_admin
                if previous_warmup is None:
                    os.environ.pop("TADA_DISABLE_WARMUP", None)
                else:
                    os.environ["TADA_DISABLE_WARMUP"] = previous_warmup


if __name__ == "__main__":
    unittest.main()
