from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch

try:
    from tada.modules.encoder import EncoderOutput
except Exception:  # pragma: no cover - runtime import depends on vendor setup.
    EncoderOutput = Any  # type: ignore[assignment]


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？…])\s+|\n+")
SHORT_SENTENCE_MERGE_MAX_CHARS = 30
SHORT_SENTENCE_MERGE_ENDINGS = ("!", "?", "！", "？")
FOLLOWING_SENTENCE_MERGE_MIN_CHARS = 20


def split_sentences(
    text: str,
    *,
    enabled: bool,
    short_sentence_merge_max_chars: int = SHORT_SENTENCE_MERGE_MAX_CHARS,
    following_sentence_merge_min_chars: int = FOLLOWING_SENTENCE_MERGE_MIN_CHARS,
) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    if not enabled:
        return [cleaned]
    parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(cleaned) if part.strip()]
    if not parts:
        return [cleaned]

    merged_parts: list[str] = []
    index = 0
    while index < len(parts):
        current = parts[index]
        if (
            len(current) <= max(0, int(short_sentence_merge_max_chars))
            and current.endswith(SHORT_SENTENCE_MERGE_ENDINGS)
            and index + 1 < len(parts)
            and len(parts[index + 1]) >= max(0, int(following_sentence_merge_min_chars))
        ):
            merged_parts.append(f"{current} {parts[index + 1]}")
            index += 2
            continue
        merged_parts.append(current)
        index += 1
    return merged_parts


def _pad_and_concat_1d(tensors: list[torch.Tensor], *, pad_value: int | float = 0) -> torch.Tensor:
    max_len = max(tensor.shape[-1] for tensor in tensors)
    padded: list[torch.Tensor] = []
    for tensor in tensors:
        if tensor.shape[-1] == max_len:
            padded.append(tensor)
            continue
        padded.append(torch.nn.functional.pad(tensor, (0, max_len - tensor.shape[-1]), value=pad_value))
    return torch.cat(padded, dim=0)


def _pad_and_concat_2d(tensors: list[torch.Tensor], *, pad_value: int | float = 0) -> torch.Tensor:
    max_len = max(tensor.shape[1] for tensor in tensors)
    padded: list[torch.Tensor] = []
    for tensor in tensors:
        if tensor.shape[1] == max_len:
            padded.append(tensor)
            continue
        padding = (0, 0, 0, max_len - tensor.shape[1])
        padded.append(torch.nn.functional.pad(tensor, padding, value=pad_value))
    return torch.cat(padded, dim=0)


def merge_encoder_outputs(prompts: list[EncoderOutput]) -> EncoderOutput:
    if not prompts:
        raise ValueError("Cannot merge an empty prompt list.")
    if len(prompts) == 1:
        return prompts[0]

    first = prompts[0]
    sample_rate = int(first.sample_rate)
    batch_payload: dict[str, Any] = {}

    batch_payload["text"] = [prompt.text[0] if prompt.text else "" for prompt in prompts]
    batch_payload["sample_rate"] = sample_rate

    direct_cat_fields = {"audio_len", "text_tokens_len", "text_emb_reduced"}
    pad_1d_fields = {"audio", "token_positions", "text_tokens", "token_masks"}
    pad_2d_fields = {
        "token_values",
        "encoded_expanded",
        "non_sampled_encoded_expanded",
        "text_emb_expanded",
        "logits",
    }

    for field_name in first.__dataclass_fields__:
        if field_name in {"text", "sample_rate"}:
            continue
        field_values = [getattr(prompt, field_name) for prompt in prompts]
        if all(value is None for value in field_values):
            batch_payload[field_name] = None
            continue

        tensor_values = [value for value in field_values if isinstance(value, torch.Tensor)]
        if len(tensor_values) != len(field_values):
            batch_payload[field_name] = field_values[0]
            continue

        if field_name in direct_cat_fields:
            batch_payload[field_name] = torch.cat(tensor_values, dim=0)
        elif field_name in pad_1d_fields:
            batch_payload[field_name] = _pad_and_concat_1d(tensor_values)
        elif field_name in pad_2d_fields:
            batch_payload[field_name] = _pad_and_concat_2d(tensor_values)
        else:
            batch_payload[field_name] = torch.cat(tensor_values, dim=0)

    return EncoderOutput(**batch_payload)


def chunk_waveform(waveform: torch.Tensor, *, sample_rate: int, chunk_ms: int) -> list[torch.Tensor]:
    mono = waveform.detach().float().cpu()
    if mono.ndim == 2:
        mono = mono.reshape(-1)
    if mono.ndim == 0:
        mono = mono.unsqueeze(0)
    chunk_size = max(1, int(sample_rate * chunk_ms / 1000))
    return [mono[index : index + chunk_size] for index in range(0, int(mono.numel()), chunk_size)]


def ensure_directory(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path
