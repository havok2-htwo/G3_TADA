from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "benchmark_results" / "server_loadtest"
DEFAULT_SERVER_BASE_URL = "http://127.0.0.1:7878"
DEFAULT_DURATION_SECONDS = 60.0
DEFAULT_USER_COUNT = 24
DEFAULT_ARRIVAL_RATE = 1.8
DEFAULT_RANDOM_SEED = 1337

TEXT_LIBRARY = [
    "Hallo! Das ist ein kurzer Lasttest.",
    "Das Wetter ist heute wirklich hervorragend, finden Sie nicht auch?",
    "Kuenstliche Intelligenz entwickelt sich rasend schnell und eroeffnet neue Moeglichkeiten.",
    "Ich bin ein virtueller Sprachassistent und helfe Ihnen gerne bei Ihren Aufgaben.",
    "Die Batch-Verarbeitung auf Grafikkarten spart massiv Zeit in der Produktion.",
    "Lassen Sie uns gemeinsam herausfinden, wie wir dieses Problem effizient loesen koennen.",
    "Viele Technologieunternehmen setzen vermehrt auf lokal laufende Open-Source-Modelle.",
    "Ein gutes Sprachmodell muss nicht nur intelligent sein, sondern auch natuerlich klingen.",
    "Ich freue mich darauf, Ihnen dieses fantastische Audioergebnis praesentieren zu duerfen.",
    "Haben Sie schon einmal darueber nachgedacht, wie viele Berechnungen hier pro Sekunde stattfinden?",
    "Manchmal ist es besser, tief durchzuatmen und erst dann eine Entscheidung zu treffen.",
    "Geduld ist eine Tugend, vor allem wenn man komplexe neuronale Netze trainiert.",
    "Vergessen Sie nicht, das Projekt vor dem Feierabend ausgiebig zu testen.",
    "Wenn wir zusammenarbeiten, koennen wir selbst die schwierigsten Herausforderungen meistern.",
    "Ich kann es kaum erwarten, dass wir die finale Version der Software veroeffentlichen.",
    "Sprachsynthese auf einer RTX 5090 ist eine Demonstration technischer Brillanz.",
    "Bitte pruefen Sie die Warteschlange, die Batch-Groesse und die Zeit bis zum ersten Audio.",
    "Kurze Eingaben sollten schnell kommen, lange Eingaben muessen trotzdem sauber abgearbeitet werden.",
    "Wir wollen sehen, wie gut sich Batching, Wait-Time und Antwortlatenz gegenseitig austarieren.",
    "Auch unter Last sollen die Antworten stabil bleiben und nicht ploetzlich qualitativ einbrechen.",
]

SIZE_TO_SENTENCE_RANGE = {
    "small": (1, 2),
    "medium": (3, 5),
    "large": (6, 9),
}
SIZE_WEIGHTS = ("small", "medium", "large"), (0.5, 0.35, 0.15)


@dataclass
class RequestPlan:
    request_index: int
    user_id: str
    scheduled_offset_s: float
    size_label: str
    planned_sentence_count: int
    text: str
    text_char_count: int


@dataclass
class RequestResult:
    request_index: int
    user_id: str
    scheduled_offset_s: float
    size_label: str
    text_char_count: int
    planned_sentence_count: int
    success: bool
    request_id: str | None
    ttft_ms: float | None
    total_ms: float | None
    audio_duration_ms: float | None
    sentence_count: int | None
    batch_count: int | None
    observed_batch_sizes: list[int]
    error: str | None = None


def _json_request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib_request.Request(url, data=data, headers=headers, method=method)
    with urllib_request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_voices(server_base_url: str) -> list[dict[str, Any]]:
    payload = _json_request(f"{server_base_url.rstrip('/')}/api/v1/voices", timeout=30.0)
    return list(payload.get("voices") or [])


def _resolve_voice(server_base_url: str, voice_id: str | None) -> dict[str, Any]:
    voices = _fetch_voices(server_base_url)
    if not voices:
        raise RuntimeError("Server returned no voices. Create at least one voice before running the load test.")
    if voice_id:
        for voice in voices:
            if voice.get("voice_id") == voice_id:
                return voice
        available = ", ".join(str(voice.get("voice_id")) for voice in voices)
        raise RuntimeError(f"Voice '{voice_id}' was not found. Available voices: {available}")
    return voices[0]


def _compose_text(rng: random.Random, sentence_count: int) -> str:
    sentence_indices = [rng.randrange(len(TEXT_LIBRARY)) for _ in range(sentence_count)]
    return " ".join(TEXT_LIBRARY[index] for index in sentence_indices)


def _phase_rate(offset_s: float, duration_seconds: float, arrival_rate: float) -> float:
    if duration_seconds <= 0:
        return arrival_rate
    position = offset_s / duration_seconds
    if position < 0.2:
        return arrival_rate * 0.7
    if position < 0.75:
        return arrival_rate * 1.15
    return arrival_rate * 0.85


def _generate_request_plans(
    *,
    duration_seconds: float,
    user_count: int,
    arrival_rate: float,
    seed: int,
) -> list[RequestPlan]:
    rng = random.Random(seed)
    plans: list[RequestPlan] = []
    current_offset = 0.0
    request_index = 1
    user_cursor = 0

    while True:
        effective_rate = max(0.05, _phase_rate(current_offset, duration_seconds, arrival_rate))
        inter_arrival = max(0.05, rng.expovariate(effective_rate))
        current_offset += inter_arrival
        if current_offset > duration_seconds:
            break

        size_label = rng.choices(*SIZE_WEIGHTS, k=1)[0]
        min_sentences, max_sentences = SIZE_TO_SENTENCE_RANGE[size_label]
        planned_sentence_count = rng.randint(min_sentences, max_sentences)
        text = _compose_text(rng, planned_sentence_count)
        plans.append(
            RequestPlan(
                request_index=request_index,
                user_id=f"user-{user_cursor + 1:03d}",
                scheduled_offset_s=round(current_offset, 3),
                size_label=size_label,
                planned_sentence_count=planned_sentence_count,
                text=text,
                text_char_count=len(text),
            )
        )
        request_index += 1
        user_cursor = (user_cursor + 1) % user_count

    return plans


def _sleep_until(target_perf: float) -> None:
    while True:
        remaining = target_perf - time.perf_counter()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.1))


def _run_stream_request(
    server_base_url: str,
    *,
    voice_id: str,
    plan: RequestPlan,
    run_started_perf: float,
    timeout: float,
) -> RequestResult:
    scheduled_perf = run_started_perf + plan.scheduled_offset_s
    _sleep_until(scheduled_perf)

    payload = json.dumps({"text": plan.text, "voice_id": voice_id}).encode("utf-8")
    request = urllib_request.Request(
        f"{server_base_url.rstrip('/')}/api/v1/synthesize/stream",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/x-ndjson",
        },
        method="POST",
    )

    started_at = time.perf_counter()
    request_id = None
    ttft_ms = None
    done_result = None
    last_emitted_audio_ms = None
    observed_batch_sizes: list[int] = []

    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                event = json.loads(line)
                event_type = str(event.get("type") or "")
                request_id = request_id or event.get("request_id")
                if event_type == "chunk":
                    if ttft_ms is None:
                        ttft_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
                    last_emitted_audio_ms = float(event.get("emitted_audio_ms") or 0.0)
                elif event_type == "batch":
                    batch_size = int(event.get("batch_size") or 0)
                    if batch_size > 0:
                        observed_batch_sizes.append(batch_size)
                elif event_type == "done":
                    done_result = event.get("result") or {}
                    break
                elif event_type == "error":
                    raise RuntimeError(str(event.get("message") or "Streaming request failed."))
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        return RequestResult(
            request_index=plan.request_index,
            user_id=plan.user_id,
            scheduled_offset_s=plan.scheduled_offset_s,
            size_label=plan.size_label,
            text_char_count=plan.text_char_count,
            planned_sentence_count=plan.planned_sentence_count,
            success=False,
            request_id=request_id,
            ttft_ms=ttft_ms,
            total_ms=round((time.perf_counter() - started_at) * 1000.0, 2),
            audio_duration_ms=last_emitted_audio_ms,
            sentence_count=None,
            batch_count=None,
            observed_batch_sizes=observed_batch_sizes,
            error=f"HTTP {exc.code}: {detail or exc.reason}",
        )
    except Exception as exc:
        return RequestResult(
            request_index=plan.request_index,
            user_id=plan.user_id,
            scheduled_offset_s=plan.scheduled_offset_s,
            size_label=plan.size_label,
            text_char_count=plan.text_char_count,
            planned_sentence_count=plan.planned_sentence_count,
            success=False,
            request_id=request_id,
            ttft_ms=ttft_ms,
            total_ms=round((time.perf_counter() - started_at) * 1000.0, 2),
            audio_duration_ms=last_emitted_audio_ms,
            sentence_count=None,
            batch_count=None,
            observed_batch_sizes=observed_batch_sizes,
            error=str(exc),
        )

    total_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
    return RequestResult(
        request_index=plan.request_index,
        user_id=plan.user_id,
        scheduled_offset_s=plan.scheduled_offset_s,
        size_label=plan.size_label,
        text_char_count=plan.text_char_count,
        planned_sentence_count=plan.planned_sentence_count,
        success=True,
        request_id=request_id,
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        audio_duration_ms=float((done_result or {}).get("audio_duration_ms") or last_emitted_audio_ms or 0.0),
        sentence_count=(done_result or {}).get("sentence_count"),
        batch_count=(done_result or {}).get("batch_count"),
        observed_batch_sizes=observed_batch_sizes,
        error=None,
    )


def _poll_health(
    server_base_url: str,
    *,
    stop_event: threading.Event,
    poll_interval: float,
    health_samples: list[dict[str, Any]],
) -> None:
    while not stop_event.is_set():
        local_started = time.perf_counter()
        try:
            payload = _json_request(f"{server_base_url.rstrip('/')}/api/health", timeout=max(5.0, poll_interval * 4))
            dashboard = payload.get("dashboard") or {}
            current_batch = dashboard.get("current_batch") or {}
            health_samples.append(
                {
                    "polled_at_perf": local_started,
                    "timestamp": dashboard.get("timestamp"),
                    "queue_length": int(dashboard.get("queue_length") or 0),
                    "active_request_count": int(dashboard.get("active_request_count") or 0),
                    "current_batch_size": int(current_batch.get("size") or 0),
                    "current_batch_voice_id": current_batch.get("voice_id"),
                    "throughput_audio_sps": float(dashboard.get("throughput_audio_sps") or 0.0),
                    "completed_requests_total": int(dashboard.get("completed_requests_total") or 0),
                    "failed_requests_total": int(dashboard.get("failed_requests_total") or 0),
                }
            )
        except Exception as exc:
            health_samples.append(
                {
                    "polled_at_perf": local_started,
                    "error": str(exc),
                }
            )
        stop_event.wait(poll_interval)


def _mean(values: list[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return round(statistics.mean(numeric), 2)


def _min(values: list[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return round(min(numeric), 2)


def _max(values: list[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return round(max(numeric), 2)


def _percentile(values: list[float | None], fraction: float) -> float | None:
    numeric = sorted(float(value) for value in values if value is not None)
    if not numeric:
        return None
    if len(numeric) == 1:
        return round(numeric[0], 2)
    index = max(0, min(len(numeric) - 1, round((len(numeric) - 1) * fraction)))
    return round(numeric[index], 2)


def _size_summary(results: list[RequestResult]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for size_label in ("small", "medium", "large"):
        grouped = [result for result in results if result.size_label == size_label and result.success]
        summary[size_label] = {
            "success_count": len(grouped),
            "mean_ttft_ms": _mean([result.ttft_ms for result in grouped]),
            "max_ttft_ms": _max([result.ttft_ms for result in grouped]),
            "mean_total_ms": _mean([result.total_ms for result in grouped]),
        }
    return summary


def _per_user_summary(results: list[RequestResult]) -> dict[str, Any]:
    grouped: dict[str, list[RequestResult]] = defaultdict(list)
    for result in results:
        grouped[result.user_id].append(result)

    payload: dict[str, Any] = {}
    for user_id, user_results in sorted(grouped.items()):
        successful = [result for result in user_results if result.success]
        payload[user_id] = {
            "request_count": len(user_results),
            "success_count": len(successful),
            "failure_count": len(user_results) - len(successful),
            "min_ttft_ms": _min([result.ttft_ms for result in successful]),
            "mean_ttft_ms": _mean([result.ttft_ms for result in successful]),
            "max_ttft_ms": _max([result.ttft_ms for result in successful]),
            "min_total_ms": _min([result.total_ms for result in successful]),
            "mean_total_ms": _mean([result.total_ms for result in successful]),
            "max_total_ms": _max([result.total_ms for result in successful]),
            "size_counts": dict(Counter(result.size_label for result in user_results)),
        }
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic 60s-style HTTP load test for the running TADA server.")
    parser.add_argument("--server", default=DEFAULT_SERVER_BASE_URL, help="Server base URL, e.g. http://127.0.0.1:7878")
    parser.add_argument("--voice-id", default="", help="Voice ID to use. Defaults to the first public voice.")
    parser.add_argument("--duration-seconds", type=float, default=DEFAULT_DURATION_SECONDS, help="Arrival phase duration in seconds.")
    parser.add_argument("--user-count", type=int, default=DEFAULT_USER_COUNT, help="Number of synthetic users.")
    parser.add_argument("--arrival-rate", type=float, default=DEFAULT_ARRIVAL_RATE, help="Average request arrivals per second.")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Deterministic RNG seed.")
    parser.add_argument("--poll-interval", type=float, default=0.5, help="How often to poll /api/health in seconds.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout in seconds.")
    args = parser.parse_args()

    server_base_url = str(args.server).strip().rstrip("/")
    voice = _resolve_voice(server_base_url, args.voice_id.strip() or None)
    plans = _generate_request_plans(
        duration_seconds=max(1.0, float(args.duration_seconds)),
        user_count=max(1, int(args.user_count)),
        arrival_rate=max(0.1, float(args.arrival_rate)),
        seed=int(args.seed),
    )
    if not plans:
        raise RuntimeError("The deterministic plan generator produced no requests. Increase duration or arrival rate.")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = OUT_DIR / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Server: {server_base_url}")
    print(f"Voice: {voice.get('voice_id')} ({voice.get('name')})")
    print(f"Duration: {float(args.duration_seconds):.1f}s")
    print(f"Users: {int(args.user_count)}")
    print(f"Arrival rate: {float(args.arrival_rate):.2f} req/s")
    print(f"Seed: {int(args.seed)}")
    print(f"Planned requests: {len(plans)}")
    print(f"Size mix: {dict(Counter(plan.size_label for plan in plans))}")
    print("")

    stop_event = threading.Event()
    health_samples: list[dict[str, Any]] = []
    poll_thread = threading.Thread(
        target=_poll_health,
        kwargs={
            "server_base_url": server_base_url,
            "stop_event": stop_event,
            "poll_interval": max(0.1, float(args.poll_interval)),
            "health_samples": health_samples,
        },
        daemon=True,
    )
    poll_thread.start()

    request_results: list[RequestResult] = []
    started_at = time.perf_counter()

    with ThreadPoolExecutor(max_workers=len(plans)) as executor:
        futures = [
            executor.submit(
                _run_stream_request,
                server_base_url,
                voice_id=str(voice.get("voice_id")),
                plan=plan,
                run_started_perf=started_at,
                timeout=float(args.timeout),
            )
            for plan in plans
        ]
        for future in as_completed(futures):
            request_results.append(future.result())

    total_wall_seconds = time.perf_counter() - started_at
    stop_event.set()
    poll_thread.join(timeout=5.0)

    request_results.sort(key=lambda item: item.request_index)
    successful = [result for result in request_results if result.success]
    failed = [result for result in request_results if not result.success]
    batch_sizes_health = [int(sample.get("current_batch_size") or 0) for sample in health_samples if "current_batch_size" in sample]
    batch_sizes_stream = [
        int(batch_size)
        for result in request_results
        for batch_size in result.observed_batch_sizes
        if int(batch_size) > 0
    ]
    active_counts = [int(sample.get("active_request_count") or 0) for sample in health_samples if "active_request_count" in sample]
    queue_lengths = [int(sample.get("queue_length") or 0) for sample in health_samples if "queue_length" in sample]
    throughputs = [float(sample.get("throughput_audio_sps") or 0.0) for sample in health_samples if "throughput_audio_sps" in sample]
    per_user = _per_user_summary(request_results)
    per_user_min_ttfts = [payload.get("min_ttft_ms") for payload in per_user.values()]
    per_user_mean_ttfts = [payload.get("mean_ttft_ms") for payload in per_user.values()]
    per_user_max_ttfts = [payload.get("max_ttft_ms") for payload in per_user.values()]

    summary = {
        "server_base_url": server_base_url,
        "voice_id": voice.get("voice_id"),
        "voice_name": voice.get("name"),
        "duration_seconds": float(args.duration_seconds),
        "user_count": int(args.user_count),
        "arrival_rate": float(args.arrival_rate),
        "seed": int(args.seed),
        "planned_request_count": len(plans),
        "success_count": len(successful),
        "failure_count": len(failed),
        "total_wall_seconds": round(total_wall_seconds, 2),
        "drain_seconds_after_arrivals": round(max(0.0, total_wall_seconds - float(args.duration_seconds)), 2),
        "ttft_min_ms": _min([result.ttft_ms for result in successful]),
        "ttft_mean_ms": _mean([result.ttft_ms for result in successful]),
        "ttft_p95_ms": _percentile([result.ttft_ms for result in successful], 0.95),
        "ttft_max_ms": _max([result.ttft_ms for result in successful]),
        "total_min_ms": _min([result.total_ms for result in successful]),
        "total_mean_ms": _mean([result.total_ms for result in successful]),
        "total_p95_ms": _percentile([result.total_ms for result in successful], 0.95),
        "total_max_ms": _max([result.total_ms for result in successful]),
        "audio_mean_ms": _mean([result.audio_duration_ms for result in successful]),
        "mean_batch_count_per_request": _mean([float(result.batch_count) if result.batch_count is not None else None for result in successful]),
        "max_observed_batch_size": max(batch_sizes_stream + batch_sizes_health) if (batch_sizes_stream or batch_sizes_health) else 0,
        "max_observed_batch_size_stream": max(batch_sizes_stream) if batch_sizes_stream else 0,
        "max_observed_batch_size_health": max(batch_sizes_health) if batch_sizes_health else 0,
        "max_observed_active_requests": max(active_counts) if active_counts else 0,
        "max_observed_queue_length": max(queue_lengths) if queue_lengths else 0,
        "max_observed_realtime_factor": round(max(throughputs), 3) if throughputs else 0.0,
        "user_ttft_min_min_ms": _min(per_user_min_ttfts),
        "user_ttft_min_mean_ms": _mean(per_user_min_ttfts),
        "user_ttft_min_max_ms": _max(per_user_min_ttfts),
        "user_ttft_mean_min_ms": _min(per_user_mean_ttfts),
        "user_ttft_mean_mean_ms": _mean(per_user_mean_ttfts),
        "user_ttft_mean_max_ms": _max(per_user_mean_ttfts),
        "user_ttft_max_min_ms": _min(per_user_max_ttfts),
        "user_ttft_max_mean_ms": _mean(per_user_max_ttfts),
        "user_ttft_max_max_ms": _max(per_user_max_ttfts),
        "size_summary": _size_summary(request_results),
        "size_mix": dict(Counter(plan.size_label for plan in plans)),
    }

    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "request_plan.json", [asdict(plan) for plan in plans])
    _write_json(run_dir / "request_results.json", [asdict(item) for item in request_results])
    _write_json(run_dir / "per_user_summary.json", per_user)
    _write_json(run_dir / "health_samples.json", health_samples)

    print("Summary")
    print(f"  Success: {summary['success_count']} / {summary['planned_request_count']}")
    print(f"  TTFT min/mean/p95/max: {summary['ttft_min_ms']} / {summary['ttft_mean_ms']} / {summary['ttft_p95_ms']} / {summary['ttft_max_ms']} ms")
    print(f"  User min TTFT min/mean/max: {summary['user_ttft_min_min_ms']} / {summary['user_ttft_min_mean_ms']} / {summary['user_ttft_min_max_ms']} ms")
    print(f"  User mean TTFT min/mean/max: {summary['user_ttft_mean_min_ms']} / {summary['user_ttft_mean_mean_ms']} / {summary['user_ttft_mean_max_ms']} ms")
    print(f"  User max TTFT min/mean/max: {summary['user_ttft_max_min_ms']} / {summary['user_ttft_max_mean_ms']} / {summary['user_ttft_max_max_ms']} ms")
    print(f"  Total mean/p95/max: {summary['total_mean_ms']} / {summary['total_p95_ms']} / {summary['total_max_ms']} ms")
    print(f"  Max observed batch size: {summary['max_observed_batch_size']}")
    print(f"  Max observed active requests: {summary['max_observed_active_requests']}")
    print(f"  Max observed queue length: {summary['max_observed_queue_length']}")
    print(f"  Max observed realtime: {summary['max_observed_realtime_factor']}x")
    print(f"  Drain after arrivals: {summary['drain_seconds_after_arrivals']} s")
    print(f"  Output: {run_dir}")


if __name__ == "__main__":
    main()
