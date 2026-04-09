import unittest

import torch

from backend.runtime_service import (
    MAX_REFERENCE_AUDIO_SECONDS,
    MIN_REFERENCE_AUDIO_SECONDS,
    REFERENCE_TAIL_SILENCE_SECONDS,
    TadaRuntimeService,
    prepare_reference_prompt_audio,
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
    def test_decode_preview_delta_respects_valid_input_length(self):
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
        self.assertEqual(int(delta.numel()), 25)

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


if __name__ == "__main__":
    unittest.main()
