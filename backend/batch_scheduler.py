from __future__ import annotations

import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch

from backend.config_store import ConfigStore, now_iso
from backend.prompt_batch import chunk_waveform, split_sentences
from backend.runtime_service import BatchGenerationItem, BatchGenerationResult, BatchPreviewUpdate, TadaRuntimeService

DASHBOARD_REFRESH_SECONDS = 0.5


@dataclass
class SynthesisRequestState:
    request_id: str
    voice_id: str
    text: str
    sentences: list[str]
    created_at: str
    created_perf: float
    event_queue: queue.Queue[dict[str, Any]] = field(default_factory=queue.Queue)
    done_event: threading.Event = field(default_factory=threading.Event)
    pending_sentence_indices: deque[int] = field(init=False)
    inflight_sentence_indices: set[int] = field(default_factory=set)
    next_emit_sentence_index: int = 0
    sample_rate: int | None = None
    sentence_waveforms: dict[int, torch.Tensor] = field(default_factory=dict)
    first_chunk_perf: float | None = None
    admitted_perf: float | None = None
    finished_perf: float | None = None
    batch_count: int = 0
    total_chunks: int = 0
    emitted_total_samples: int = 0
    emitted_samples_by_sentence: dict[int, int] = field(default_factory=dict)
    chunk_index_by_sentence: dict[int, int] = field(default_factory=dict)
    pending_preview_by_sentence: dict[int, torch.Tensor] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        self.pending_sentence_indices = deque(range(len(self.sentences)))

    def has_pending_sentences(self) -> bool:
        return bool(self.pending_sentence_indices)

    def is_complete(self) -> bool:
        return (
            len(self.sentence_waveforms) >= len(self.sentences)
            and not self.pending_sentence_indices
            and not self.inflight_sentence_indices
            and self.next_emit_sentence_index >= len(self.sentences)
        )

    def summary(self, *, now_perf: float) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "voice_id": self.voice_id,
            "sentence_index": self.next_emit_sentence_index,
            "sentence_count": len(self.sentences),
            "pending_sentences": len(self.pending_sentence_indices),
            "inflight_sentences": len(self.inflight_sentence_indices),
            "age_ms": round((now_perf - self.created_perf) * 1000.0, 2),
            "batches": self.batch_count,
        }


class BatchScheduler:
    def __init__(self, runtime_service: TadaRuntimeService, config_store: ConfigStore):
        self.runtime_service = runtime_service
        self.config_store = config_store
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._waiting: deque[SynthesisRequestState] = deque()
        self._active: list[SynthesisRequestState] = []
        self._stop_event = threading.Event()
        self._current_batch: dict[str, Any] | None = None
        self._history: deque[dict[str, Any]] = deque(maxlen=600)
        self._recent_ttfts: deque[tuple[float, float]] = deque(maxlen=2048)
        self._total_audio_seconds = 0.0
        self._completed_requests = 0
        self._failed_requests = 0
        self._last_snapshot_audio_total = 0.0
        self._last_snapshot_perf = time.perf_counter()
        self._last_snapshot_completed = 0
        self._last_snapshot_failed = 0
        self._last_batch_stats: dict[str, Any] = {"batch_wall_s": 0.0, "batch_rtf": 0.0, "batch_size": 0}
        self._latest_snapshot: dict[str, Any] = {
            "timestamp": now_iso(),
            "queue_length": 0,
            "queued_sentence_count": 0,
            "active_request_count": 0,
            "waiting_requests": [],
            "active_requests": [],
            "current_batch": None,
            "mean_ttft_ms": None,
            "throughput_audio_sps": 0.0,
            "completed_requests_total": 0,
            "failed_requests_total": 0,
            "completed_requests_delta": 0,
            "failed_requests_delta": 0,
            "last_batch_wall_s": 0.0,
            "last_batch_rtf": 0.0,
            "history": [],
        }
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._metrics_worker = threading.Thread(target=self._metrics_loop, daemon=True)
        self._worker.start()
        self._metrics_worker.start()

    def submit_request(self, *, text: str, voice_id: str) -> SynthesisRequestState:
        settings = self.config_store.get_settings()
        sentences = split_sentences(
            text,
            enabled=settings.sentence_chunking,
            short_sentence_merge_max_chars=settings.short_sentence_merge_max_chars,
            following_sentence_merge_min_chars=settings.following_sentence_merge_min_chars,
        )
        if not sentences:
            raise ValueError("Text must not be empty.")

        state = SynthesisRequestState(
            request_id=uuid.uuid4().hex,
            voice_id=voice_id,
            text=text.strip(),
            sentences=sentences,
            created_at=now_iso(),
            created_perf=time.perf_counter(),
        )

        with self._condition:
            if len(self._waiting) >= settings.max_queue_size:
                raise RuntimeError("The synthesis queue is full. Please retry in a moment.")
            self._waiting.append(state)
            state.event_queue.put(
                {
                    "type": "start",
                    "request_id": state.request_id,
                    "created_at": state.created_at,
                    "sentence_count": len(state.sentences),
                    "sentence_chunking": settings.sentence_chunking,
                }
            )
            state.event_queue.put(
                {
                    "type": "queue",
                    "request_id": state.request_id,
                    "queue_position": len(self._waiting),
                    "active_requests": len(self._active),
                    "queued_sentences": self._queued_sentence_count_locked(),
                }
            )
            self._condition.notify_all()
        return state

    def wait_for_result(self, state: SynthesisRequestState, timeout: float | None = None) -> dict[str, Any]:
        completed = state.done_event.wait(timeout=timeout)
        if not completed:
            raise TimeoutError("Timed out while waiting for synthesis result.")
        if state.error:
            raise RuntimeError(state.error)
        if state.result is None:
            raise RuntimeError("Synthesis finished without a result.")
        return state.result

    def iter_request_events(self, state: SynthesisRequestState):
        while True:
            event = state.event_queue.get()
            yield event
            if event["type"] in {"done", "error"}:
                break

    def dashboard_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest_snapshot)

    def iter_dashboard_events(self):
        while not self._stop_event.is_set():
            yield self.dashboard_snapshot()
            time.sleep(DASHBOARD_REFRESH_SECONDS)

    def _queued_sentence_count_locked(self) -> int:
        return sum(len(state.pending_sentence_indices) for state in self._waiting) + sum(
            len(state.pending_sentence_indices) for state in self._active
        )

    def _promote_waiting_locked(self, settings) -> None:
        while self._waiting and len(self._active) < settings.max_parallel_requests:
            state = self._waiting.popleft()
            state.admitted_perf = time.perf_counter()
            self._active.append(state)
            state.event_queue.put(
                {
                    "type": "queue",
                    "request_id": state.request_id,
                    "queue_position": 0,
                    "active_requests": len(self._active),
                    "queued_sentences": self._queued_sentence_count_locked(),
                }
            )

    @staticmethod
    def _select_voice_only_batch(active_states: list[SynthesisRequestState], max_batch_size: int) -> list[SynthesisRequestState]:
        if not active_states or max_batch_size <= 0:
            return []
        anchor_voice_id = next((state.voice_id for state in active_states if state.has_pending_sentences()), None)
        if anchor_voice_id is None:
            return []
        selected: list[SynthesisRequestState] = []
        for state in active_states:
            if state.voice_id != anchor_voice_id or not state.has_pending_sentences():
                continue
            selected.append(state)
            if len(selected) >= max_batch_size:
                break
        return selected

    @classmethod
    def _build_sentence_batch_plan(
        cls,
        active_states: list[SynthesisRequestState],
        max_batch_size: int,
    ) -> list[tuple[SynthesisRequestState, int]]:
        voice_states = cls._select_voice_only_batch(active_states, max_batch_size)
        if not voice_states:
            return []

        plan: list[tuple[SynthesisRequestState, int]] = []
        per_request_offsets: dict[str, int] = {state.request_id: 0 for state in voice_states}
        made_progress = True
        while len(plan) < max_batch_size and made_progress:
            made_progress = False
            for state in voice_states:
                local_offset = per_request_offsets[state.request_id]
                if local_offset >= len(state.pending_sentence_indices):
                    continue
                plan.append((state, state.pending_sentence_indices[local_offset]))
                per_request_offsets[state.request_id] = local_offset + 1
                made_progress = True
                if len(plan) >= max_batch_size:
                    break
        return plan

    def _reserve_sentence_batch_locked(
        self,
        batch_plan: list[tuple[SynthesisRequestState, int]],
    ) -> tuple[list[BatchGenerationItem], list[SynthesisRequestState]]:
        selected_by_request: dict[str, list[int]] = {}
        states_by_request: dict[str, SynthesisRequestState] = {}
        request_order: list[str] = []
        for state, sentence_index in batch_plan:
            if state.request_id not in selected_by_request:
                selected_by_request[state.request_id] = []
                states_by_request[state.request_id] = state
                request_order.append(state.request_id)
            selected_by_request[state.request_id].append(sentence_index)

        reserved_by_request: dict[str, deque[int]] = {}
        batch_states = [states_by_request[request_id] for request_id in request_order]
        for request_id in request_order:
            state = states_by_request[request_id]
            expected_indices = selected_by_request[request_id]
            reserved_indices: list[int] = []
            for expected in expected_indices:
                actual = state.pending_sentence_indices.popleft()
                if actual != expected:
                    raise RuntimeError(
                        f"Sentence queue order drifted for request {request_id}: expected {expected}, got {actual}."
                    )
                state.inflight_sentence_indices.add(actual)
                reserved_indices.append(actual)
            state.batch_count += 1
            reserved_by_request[request_id] = deque(reserved_indices)

        batch_items: list[BatchGenerationItem] = []
        for state, _ in batch_plan:
            sentence_index = reserved_by_request[state.request_id].popleft()
            batch_items.append(
                BatchGenerationItem(
                    request_id=state.request_id,
                    voice_id=state.voice_id,
                    text=state.sentences[sentence_index],
                    sentence_index=sentence_index,
                )
            )
        return batch_items, batch_states

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._condition:
                settings = self.config_store.get_settings()
                self._promote_waiting_locked(settings)
                if not self._active:
                    self._condition.wait(timeout=0.25)
                    continue

                should_wait = (
                    settings.batch_wait_ms > 0
                    and len(self._active) < settings.max_parallel_requests
                    and (self._waiting or any(state.batch_count == 0 for state in self._active))
                )
                if should_wait:
                    self._condition.wait(timeout=settings.batch_wait_ms / 1000.0)
                    settings = self.config_store.get_settings()
                    self._promote_waiting_locked(settings)
                    if not self._active:
                        continue

                batch_id = uuid.uuid4().hex[:8]
                batch_plan = self._build_sentence_batch_plan(self._active, settings.max_batch_size)
                if not batch_plan:
                    self._condition.wait(timeout=0.05)
                    continue

                batch_items, batch_states = self._reserve_sentence_batch_locked(batch_plan)
                self._current_batch = {
                    "batch_id": batch_id,
                    "started_at": now_iso(),
                    "size": len(batch_items),
                    "voice_id": batch_items[0].voice_id if batch_items else None,
                    "request_ids": [item.request_id for item in batch_items],
                    "items": [
                        {
                            "request_id": item.request_id,
                            "voice_id": item.voice_id,
                            "sentence_index": item.sentence_index,
                            "text_preview": item.text[:120],
                        }
                        for item in batch_items
                    ],
                }
                state_by_request = {state.request_id: state for state in batch_states}
                selected_indices_by_request: dict[str, list[int]] = {}
                for item in batch_items:
                    state = state_by_request[item.request_id]
                    selected_indices_by_request.setdefault(item.request_id, []).append(item.sentence_index)
                    state.event_queue.put(
                        {
                            "type": "batch",
                            "request_id": state.request_id,
                            "batch_id": batch_id,
                            "sentence_index": item.sentence_index,
                            "batch_size": len(batch_items),
                            "batch_voice_id": item.voice_id,
                        }
                    )
                batch_started_perf = time.perf_counter()

            def on_preview(update: BatchPreviewUpdate) -> None:
                with self._condition:
                    state = state_by_request.get(update.request_id)
                    if state is None or state.error or state.next_emit_sentence_index != update.sentence_index:
                        return
                    self._emit_preview_audio_locked(
                        state,
                        waveform=update.waveform_delta,
                        sample_rate=update.sample_rate,
                        sentence_index=update.sentence_index,
                        progress_step=update.progress_step,
                        preview=True,
                        final_chunk_of_sentence=False,
                    )

            try:
                if hasattr(self.runtime_service, "generate_batch_progressive"):
                    results = self.runtime_service.generate_batch_progressive(batch_items, on_preview=on_preview)
                else:
                    results = self.runtime_service.generate_batch(batch_items)
            except Exception as exc:
                with self._condition:
                    self._current_batch = None
                    for state in batch_states:
                        self._fail_request_locked(state, str(exc))
                        if state in self._active:
                            self._active.remove(state)
                    self._condition.notify_all()
                continue

            batch_wall_seconds = max(0.0, time.perf_counter() - batch_started_perf)
            batch_audio_seconds = sum(result.duration_seconds for result in results)
            results_by_sentence = {(result.request_id, result.sentence_index): result for result in results}

            with self._condition:
                self._current_batch = None
                for state in batch_states:
                    selected_indices = sorted(selected_indices_by_request.get(state.request_id, []))
                    if any((state.request_id, sentence_index) not in results_by_sentence for sentence_index in selected_indices):
                        self._fail_request_locked(state, "Batch result was incomplete.")
                        if state in self._active:
                            self._active.remove(state)
                        continue
                    for sentence_index in selected_indices:
                        result = results_by_sentence[(state.request_id, sentence_index)]
                        self._apply_sentence_result_locked(state, result)
                    if state.is_complete():
                        self._complete_request_locked(state)
                        if state in self._active:
                            self._active.remove(state)
                self._total_audio_seconds += batch_audio_seconds
                self._last_batch_stats = {
                    "batch_wall_s": round(batch_wall_seconds, 3),
                    "batch_rtf": round(batch_wall_seconds / batch_audio_seconds, 3) if batch_audio_seconds else 0.0,
                    "batch_size": len(batch_items),
                }
                self._condition.notify_all()

    def _apply_sentence_result_locked(self, state: SynthesisRequestState, result: BatchGenerationResult) -> None:
        state.sample_rate = result.sample_rate
        state.inflight_sentence_indices.discard(result.sentence_index)
        state.sentence_waveforms[result.sentence_index] = result.waveform
        self._flush_ready_sentences_locked(state)

    def _flush_ready_sentences_locked(self, state: SynthesisRequestState) -> None:
        while state.next_emit_sentence_index in state.sentence_waveforms:
            sentence_index = state.next_emit_sentence_index
            final_mono = state.sentence_waveforms[sentence_index].detach().float().cpu().reshape(-1)
            state.pending_preview_by_sentence.pop(sentence_index, None)
            emitted_samples = state.emitted_samples_by_sentence.get(sentence_index, 0)
            remaining = final_mono[emitted_samples:]
            self._emit_audio_chunks_locked(
                state,
                waveform=remaining,
                sample_rate=state.sample_rate or 24000,
                sentence_index=sentence_index,
                progress_step=-1,
                preview=False,
                final_chunk_of_sentence=True,
            )
            state.emitted_samples_by_sentence.pop(sentence_index, None)
            state.chunk_index_by_sentence.pop(sentence_index, None)
            state.next_emit_sentence_index += 1

    def _emit_preview_audio_locked(
        self,
        state: SynthesisRequestState,
        *,
        waveform: torch.Tensor,
        sample_rate: int,
        sentence_index: int,
        progress_step: int,
        preview: bool,
        final_chunk_of_sentence: bool,
    ) -> None:
        if int(waveform.numel()) == 0:
            return
        settings = self.config_store.get_settings()
        prebuffer_ms = max(0, int(settings.stream_prebuffer_ms))
        if prebuffer_ms <= 0 or state.emitted_samples_by_sentence.get(sentence_index, 0) > 0:
            self._emit_audio_chunks_locked(
                state,
                waveform=waveform,
                sample_rate=sample_rate,
                sentence_index=sentence_index,
                progress_step=progress_step,
                preview=preview,
                final_chunk_of_sentence=final_chunk_of_sentence,
            )
            return

        buffered = waveform.detach().float().cpu().reshape(-1)
        pending = state.pending_preview_by_sentence.get(sentence_index)
        if pending is not None and int(pending.numel()) > 0:
            buffered = torch.cat([pending, buffered], dim=-1)
        buffered_ms = float(buffered.numel()) / float(sample_rate) * 1000.0
        if buffered_ms < float(prebuffer_ms):
            state.pending_preview_by_sentence[sentence_index] = buffered
            return

        state.pending_preview_by_sentence.pop(sentence_index, None)
        self._emit_audio_chunks_locked(
            state,
            waveform=buffered,
            sample_rate=sample_rate,
            sentence_index=sentence_index,
            progress_step=progress_step,
            preview=preview,
            final_chunk_of_sentence=final_chunk_of_sentence,
        )

    def _emit_audio_chunks_locked(
        self,
        state: SynthesisRequestState,
        *,
        waveform: torch.Tensor,
        sample_rate: int,
        sentence_index: int,
        progress_step: int,
        preview: bool,
        final_chunk_of_sentence: bool,
    ) -> None:
        if int(waveform.numel()) == 0:
            return
        settings = self.config_store.get_settings()
        chunks = chunk_waveform(waveform, sample_rate=sample_rate, chunk_ms=settings.stream_chunk_ms)
        emitted_so_far_ms = float(state.emitted_total_samples) / float(sample_rate) * 1000.0
        base_chunk_index = state.chunk_index_by_sentence.get(sentence_index, 0)
        for local_index, chunk in enumerate(chunks):
            state.total_chunks += 1
            if state.first_chunk_perf is None:
                state.first_chunk_perf = time.perf_counter()
                self._recent_ttfts.append((state.first_chunk_perf, (state.first_chunk_perf - state.created_perf) * 1000.0))
            chunk_ms = float(chunk.shape[-1]) / float(sample_rate) * 1000.0
            emitted_so_far_ms += chunk_ms
            chunk_index = base_chunk_index + local_index
            state.event_queue.put(
                {
                    "type": "chunk",
                    "request_id": state.request_id,
                    "sentence_index": sentence_index,
                    "chunk_index": chunk_index,
                    "sample_rate": sample_rate,
                    "pcm16_b64": self._encode_pcm16_base64(chunk),
                    "final_chunk_of_sentence": final_chunk_of_sentence and local_index == len(chunks) - 1,
                    "emitted_audio_ms": round(emitted_so_far_ms, 2),
                    "progress_step": progress_step,
                    "preview": preview,
                }
            )
        emitted_samples = int(waveform.numel())
        state.emitted_total_samples += emitted_samples
        state.emitted_samples_by_sentence[sentence_index] = (
            state.emitted_samples_by_sentence.get(sentence_index, 0) + emitted_samples
        )
        state.chunk_index_by_sentence[sentence_index] = base_chunk_index + len(chunks)

    def _complete_request_locked(self, state: SynthesisRequestState) -> None:
        state.finished_perf = time.perf_counter()
        final_audio = torch.cat([state.sentence_waveforms[index] for index in range(len(state.sentences))], dim=-1)
        ttft_ms = (state.first_chunk_perf - state.created_perf) * 1000.0 if state.first_chunk_perf is not None else None
        total_wall_ms = (state.finished_perf - state.created_perf) * 1000.0
        state.result = self.runtime_service.save_request_audio(
            audio=final_audio,
            sample_rate=state.sample_rate or 24000,
            request_text=state.text,
            voice_id=state.voice_id,
            ttft_ms=ttft_ms,
            total_wall_ms=total_wall_ms,
            sentence_count=len(state.sentences),
            batch_count=state.batch_count,
        )
        self._completed_requests += 1
        state.event_queue.put({"type": "done", "request_id": state.request_id, "result": state.result})
        state.done_event.set()

    def _fail_request_locked(self, state: SynthesisRequestState, message: str) -> None:
        state.finished_perf = time.perf_counter()
        state.error = message
        self._failed_requests += 1
        state.event_queue.put({"type": "error", "request_id": state.request_id, "message": message})
        state.done_event.set()

    def _metrics_loop(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(DASHBOARD_REFRESH_SECONDS)
            with self._lock:
                snapshot = self._build_dashboard_snapshot_locked()
                self._history.append(
                    {
                        "timestamp": snapshot["timestamp"],
                        "queue_length": snapshot["queue_length"],
                        "queued_sentence_count": snapshot["queued_sentence_count"],
                        "active_requests": snapshot["active_request_count"],
                        "batch_size": snapshot["current_batch"]["size"] if snapshot["current_batch"] else 0,
                        "throughput_audio_sps": snapshot["throughput_audio_sps"],
                        "batch_wall_s": snapshot["last_batch_wall_s"],
                        "batch_rtf": snapshot["last_batch_rtf"],
                    }
                )
                snapshot["history"] = list(self._history)
                self._latest_snapshot = snapshot

    def _build_dashboard_snapshot_locked(self) -> dict[str, Any]:
        now_perf = time.perf_counter()
        valid_ttfts = [value for timestamp, value in self._recent_ttfts if now_perf - timestamp <= 600.0]
        mean_ttft_ms = round(sum(valid_ttfts) / len(valid_ttfts), 2) if valid_ttfts else None
        elapsed_s = max(0.001, now_perf - self._last_snapshot_perf)
        throughput_audio_sps = round((self._total_audio_seconds - self._last_snapshot_audio_total) / elapsed_s, 3)
        completed_delta = self._completed_requests - self._last_snapshot_completed
        failed_delta = self._failed_requests - self._last_snapshot_failed
        self._last_snapshot_audio_total = self._total_audio_seconds
        self._last_snapshot_perf = now_perf
        self._last_snapshot_completed = self._completed_requests
        self._last_snapshot_failed = self._failed_requests
        return {
            "timestamp": now_iso(),
            "queue_length": len(self._waiting),
            "queued_sentence_count": self._queued_sentence_count_locked(),
            "active_request_count": len(self._active),
            "waiting_requests": [state.summary(now_perf=now_perf) for state in list(self._waiting)[:32]],
            "active_requests": [state.summary(now_perf=now_perf) for state in self._active],
            "current_batch": self._current_batch,
            "mean_ttft_ms": mean_ttft_ms,
            "throughput_audio_sps": throughput_audio_sps,
            "completed_requests_total": self._completed_requests,
            "failed_requests_total": self._failed_requests,
            "completed_requests_delta": completed_delta,
            "failed_requests_delta": failed_delta,
            "last_batch_wall_s": self._last_batch_stats["batch_wall_s"],
            "last_batch_rtf": self._last_batch_stats["batch_rtf"],
            "history": list(self._history),
        }

    @staticmethod
    def _encode_pcm16_base64(waveform: torch.Tensor) -> str:
        import base64
        import numpy as np

        mono = waveform.detach().float().cpu().reshape(-1).clamp(-1.0, 1.0).numpy()
        pcm16 = np.clip(np.round(mono * 32767.0), -32768, 32767).astype("<i2", copy=False)
        return base64.b64encode(pcm16.tobytes()).decode("ascii")
