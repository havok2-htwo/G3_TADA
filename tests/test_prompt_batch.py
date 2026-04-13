from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch

import torch

from backend.prompt_batch import merge_encoder_outputs, split_sentences


@dataclass
class FakeEncoderOutput:
    audio: torch.Tensor
    audio_len: torch.Tensor
    text: list[str]
    token_positions: torch.Tensor
    token_values: torch.Tensor
    sample_rate: int = 24000
    text_tokens: torch.Tensor | None = None
    text_tokens_len: torch.Tensor | None = None
    encoded_expanded: torch.Tensor | None = None
    non_sampled_encoded_expanded: torch.Tensor | None = None
    text_emb_expanded: torch.Tensor | None = None
    token_masks: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    text_emb_reduced: torch.Tensor | None = None


class PromptBatchTests(unittest.TestCase):
    def test_sentence_splitter_preserves_order(self):
        parts = split_sentences("Das ist ein laengerer Einstiegssatz. Das ist gut!\nNoch ein Satz?", enabled=True)
        self.assertEqual(parts, ["Das ist ein laengerer Einstiegssatz.", "Das ist gut!", "Noch ein Satz?"])

    def test_sentence_splitter_merges_short_sentence_with_next(self):
        parts = split_sentences("Hallo Matthias! Schön, Sie wiederzusehen.", enabled=True)
        self.assertEqual(parts, ["Hallo Matthias! Schön, Sie wiederzusehen."])

    def test_sentence_splitter_cascades_consecutive_short_sentences(self):
        parts = split_sentences("Hi! Ja! Dann geht es weiter.", enabled=True)
        self.assertEqual(parts, ["Hi! Ja! Dann geht es weiter."])

    def test_sentence_splitter_merges_short_period_sentence(self):
        parts = split_sentences("Hallo Welt. Das ist gut!", enabled=True)
        self.assertEqual(parts, ["Hallo Welt. Das ist gut!"])

    def test_sentence_splitter_respects_configurable_thresholds(self):
        parts = split_sentences(
            "Hallo Matthias! Schön, Sie wiederzusehen.",
            enabled=True,
            short_sentence_merge_max_chars=5,
            following_sentence_merge_min_chars=20,
        )
        self.assertEqual(parts, ["Hallo Matthias!", "Schön, Sie wiederzusehen."])

    def test_sentence_splitter_backward_merge_orphaned_short_tail(self):
        parts = split_sentences("Er sagte etwas. Ok! Ja!", enabled=True)
        self.assertEqual(parts, ["Er sagte etwas. Ok! Ja!"])

    def test_sentence_splitter_merges_two_short_sentences_only(self):
        parts = split_sentences("Ok! Ja!", enabled=True)
        self.assertEqual(parts, ["Ok! Ja!"])

    def test_sentence_splitter_short_with_short_follower(self):
        parts = split_sentences("Hallo! Wie geht es dir?", enabled=True)
        self.assertEqual(parts, ["Hallo! Wie geht es dir?"])

    def test_sentence_splitter_many_short_cascade(self):
        parts = split_sentences("Wow! Cool! Super! Na dann.", enabled=True)
        self.assertEqual(parts, ["Wow! Cool! Super! Na dann."])

    def test_merge_encoder_outputs_keeps_per_row_prompt_data(self):
        prompt_a = FakeEncoderOutput(
            audio=torch.zeros(1, 8),
            audio_len=torch.tensor([8.0]),
            text=["eins"],
            text_tokens=torch.tensor([[1, 2, 3]], dtype=torch.long),
            text_tokens_len=torch.tensor([3], dtype=torch.long),
            token_positions=torch.tensor([[1, 2, 3]], dtype=torch.long),
            token_values=torch.ones(1, 3, 4),
            sample_rate=24000,
        )
        prompt_b = FakeEncoderOutput(
            audio=torch.zeros(1, 12),
            audio_len=torch.tensor([12.0]),
            text=["zwei"],
            text_tokens=torch.tensor([[4, 5]], dtype=torch.long),
            text_tokens_len=torch.tensor([2], dtype=torch.long),
            token_positions=torch.tensor([[1, 2]], dtype=torch.long),
            token_values=torch.full((1, 2, 4), 2.0),
            sample_rate=24000,
        )
        with patch("backend.prompt_batch.EncoderOutput", FakeEncoderOutput):
            merged = merge_encoder_outputs([prompt_a, prompt_b])
        self.assertEqual(merged.text, ["eins", "zwei"])
        self.assertEqual(tuple(merged.token_values.shape), (2, 3, 4))
        self.assertEqual(tuple(merged.audio.shape), (2, 12))
        self.assertEqual(merged.text_tokens_len.tolist(), [3, 2])
        self.assertEqual(merged.token_values[1, 0, 0].item(), 2.0)


if __name__ == "__main__":
    unittest.main()
