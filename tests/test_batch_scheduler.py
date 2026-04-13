from __future__ import annotations

import base64
import os
import tempfile
import time
import unittest
from pathlib import Path

import torch

from backend.batch_scheduler import BatchScheduler, SynthesisRequestState
from backend.config_store import ConfigStore
from backend.runtime_service import BatchGenerationResult, BatchPreviewUpdate


class FakeRuntimeService:
    def __init__(self):
        self.calls = []

    def generate_batch(self, items):
        self.calls.append([(item.request_id, item.sentence_index, item.text) for item in items])
        results = []
        for item in items:
            samples = torch.zeros(1, 240)
            results.append(
                BatchGenerationResult(
                    request_id=item.request_id,
                    voice_id=item.voice_id,
                    text=item.text,
                    sentence_index=item.sentence_index,
                    waveform=samples,
                    sample_rate=24000,
                    duration_seconds=0.01,
                )
            )
        return results

    def save_request_audio(self, **kwargs):
        return {
            "generation_id": "fake",
            "voice_id": kwargs["voice_id"],
            "text": kwargs["request_text"],
            "audio_url": "/api/assets/generated/fake.wav",
            "sample_rate": kwargs["sample_rate"],
            "duration_seconds": 0.02,
            "ttft_ms": kwargs["ttft_ms"],
            "batch_count": kwargs["batch_count"],
        }


class FakeProgressiveRuntimeService(FakeRuntimeService):
    def generate_batch_progressive(self, items, *, on_preview=None):
        self.calls.append([(item.request_id, item.sentence_index, item.text) for item in items])
        results = []
        for item in items:
            preview = torch.ones(12000, dtype=torch.float32) * 0.1
            if on_preview is not None:
                on_preview(
                    BatchPreviewUpdate(
                        request_id=item.request_id,
                        voice_id=item.voice_id,
                        text=item.text,
                        sentence_index=item.sentence_index,
                        waveform_delta=preview,
                        sample_rate=24000,
                        progress_step=2,
                    )
                )
            samples = torch.ones(1, 24000, dtype=torch.float32) * 0.1
            results.append(
                BatchGenerationResult(
                    request_id=item.request_id,
                    voice_id=item.voice_id,
                    text=item.text,
                    sentence_index=item.sentence_index,
                    waveform=samples,
                    sample_rate=24000,
                    duration_seconds=1.0,
                )
            )
        return results


class BatchSchedulerTests(unittest.TestCase):
    def test_voice_only_batching_keeps_mixed_queue_isolated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous = os.environ.get("TADA_ADMIN_KEY")
            os.environ["TADA_ADMIN_KEY"] = "admin-test-key"
            try:
                store = ConfigStore(Path(temp_dir))
                store.update_settings(
                    {
                        "max_batch_size": 2,
                        "max_parallel_requests": 2,
                        "batch_wait_ms": 50,
                        "short_sentence_merge_max_chars": 0,
                    }
                )
                runtime = FakeRuntimeService()
                scheduler = BatchScheduler(runtime, store)

                first = scheduler.submit_request(text="A1. A2.", voice_id="voice-a")
                second = scheduler.submit_request(text="B1.", voice_id="voice-b")
                third = scheduler.submit_request(text="C1.", voice_id="voice-c")

                scheduler.wait_for_result(first, timeout=5)
                scheduler.wait_for_result(second, timeout=5)
                scheduler.wait_for_result(third, timeout=5)

                self.assertGreaterEqual(len(runtime.calls), 3)
                self.assertEqual([text for _, _, text in runtime.calls[0]], ["A1.", "A2."])
                self.assertEqual([text for _, _, text in runtime.calls[1]], ["B1."])
                self.assertEqual([text for _, _, text in runtime.calls[2]], ["C1."])
            finally:
                if previous is None:
                    os.environ.pop("TADA_ADMIN_KEY", None)
                else:
                    os.environ["TADA_ADMIN_KEY"] = previous
            time.sleep(0.1)

    def test_sentence_level_batching_fairly_round_robins_same_voice_requests(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous = os.environ.get("TADA_ADMIN_KEY")
            os.environ["TADA_ADMIN_KEY"] = "admin-test-key"
            try:
                store = ConfigStore(Path(temp_dir))
                store.update_settings(
                    {
                        "max_batch_size": 4,
                        "max_parallel_requests": 4,
                        "batch_wait_ms": 50,
                        "short_sentence_merge_max_chars": 0,
                    }
                )
                runtime = FakeRuntimeService()
                scheduler = BatchScheduler(runtime, store)

                first = scheduler.submit_request(text="A1. A2. A3.", voice_id="voice-a")
                second = scheduler.submit_request(text="B1. B2. B3.", voice_id="voice-a")

                scheduler.wait_for_result(first, timeout=5)
                scheduler.wait_for_result(second, timeout=5)

                self.assertGreaterEqual(len(runtime.calls), 2)
                self.assertEqual([text for _, _, text in runtime.calls[0]], ["A1.", "B1.", "A2.", "B2."])
                self.assertEqual([text for _, _, text in runtime.calls[1]], ["A3.", "B3."])
            finally:
                if previous is None:
                    os.environ.pop("TADA_ADMIN_KEY", None)
                else:
                    os.environ["TADA_ADMIN_KEY"] = previous
            time.sleep(0.1)

    def test_voice_only_batch_selection_groups_same_voice_requests(self):
        states = [
            SynthesisRequestState(
                request_id="req-a1",
                voice_id="voice-a",
                text="A1.",
                sentences=["A1."],
                created_at="now",
                created_perf=0.0,
            ),
            SynthesisRequestState(
                request_id="req-b1",
                voice_id="voice-b",
                text="B1.",
                sentences=["B1."],
                created_at="now",
                created_perf=0.0,
            ),
            SynthesisRequestState(
                request_id="req-a2",
                voice_id="voice-a",
                text="A2.",
                sentences=["A2."],
                created_at="now",
                created_perf=0.0,
            ),
        ]

        selected = BatchScheduler._select_voice_only_batch(states, 2)
        self.assertEqual([state.request_id for state in selected], ["req-a1", "req-a2"])

    def test_progressive_preview_chunk_arrives_before_done_without_duplication(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous = os.environ.get("TADA_ADMIN_KEY")
            os.environ["TADA_ADMIN_KEY"] = "admin-test-key"
            try:
                store = ConfigStore(Path(temp_dir))
                store.update_settings(
                    {
                        "max_batch_size": 1,
                        "max_parallel_requests": 1,
                        "batch_wait_ms": 0,
                        "stream_chunk_ms": 500,
                        "stream_start_buffer_ms": 500,
                    }
                )
                runtime = FakeProgressiveRuntimeService()
                scheduler = BatchScheduler(runtime, store)

                state = scheduler.submit_request(text="A1.", voice_id="voice-a")
                events = list(scheduler.iter_request_events(state))
                event_types = [event["type"] for event in events]
                self.assertIn("chunk", event_types)
                self.assertEqual(event_types[-1], "done")

                first_chunk_index = event_types.index("chunk")
                done_index = event_types.index("done")
                self.assertLess(first_chunk_index, done_index)

                chunk_events = [event for event in events if event["type"] == "chunk"]
                self.assertTrue(chunk_events[0]["preview"])
                self.assertEqual(chunk_events[-1]["final_chunk_of_sentence"], True)
                self.assertIsNotNone(events[-1]["result"]["ttft_ms"])

                total_samples = 0
                for event in chunk_events:
                    pcm_bytes = base64.b64decode(event["pcm16_b64"])
                    total_samples += len(pcm_bytes) // 2
                self.assertEqual(total_samples, 24000)
            finally:
                if previous is None:
                    os.environ.pop("TADA_ADMIN_KEY", None)
                else:
                    os.environ["TADA_ADMIN_KEY"] = previous
            time.sleep(0.1)

    def test_progressive_multi_sentence_request_keeps_emit_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous = os.environ.get("TADA_ADMIN_KEY")
            os.environ["TADA_ADMIN_KEY"] = "admin-test-key"
            try:
                store = ConfigStore(Path(temp_dir))
                store.update_settings(
                    {
                        "max_batch_size": 2,
                        "max_parallel_requests": 1,
                        "batch_wait_ms": 0,
                        "stream_chunk_ms": 500,
                        "stream_start_buffer_ms": 500,
                        "short_sentence_merge_max_chars": 0,
                    }
                )
                runtime = FakeProgressiveRuntimeService()
                scheduler = BatchScheduler(runtime, store)

                state = scheduler.submit_request(text="A1. A2.", voice_id="voice-a")
                events = list(scheduler.iter_request_events(state))
                chunk_events = [event for event in events if event["type"] == "chunk"]

                self.assertEqual([text for _, _, text in runtime.calls[0]], ["A1.", "A2."])
                self.assertTrue(any(event["preview"] for event in chunk_events))
                self.assertEqual([event["sentence_index"] for event in chunk_events], sorted(event["sentence_index"] for event in chunk_events))

                total_samples = 0
                for event in chunk_events:
                    pcm_bytes = base64.b64decode(event["pcm16_b64"])
                    total_samples += len(pcm_bytes) // 2
                self.assertEqual(total_samples, 48000)
            finally:
                if previous is None:
                    os.environ.pop("TADA_ADMIN_KEY", None)
                else:
                    os.environ["TADA_ADMIN_KEY"] = previous
            time.sleep(0.1)


if __name__ == "__main__":
    unittest.main()
