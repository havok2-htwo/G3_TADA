from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "benchmark_results" / "server_http"
DEFAULT_SERVER_BASE_URL = "http://127.0.0.1:7878"
DEFAULT_TEXT = "Hallo! Das ist ein echter Python-HTTP-Benchmark fuer den TADA-Server."


@dataclass
class RequestResult:
    request_index: int
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
        raise RuntimeError("Server returned no voices. Create at least one voice before benchmarking.")
    if voice_id:
        for voice in voices:
            if voice.get("voice_id") == voice_id:
                return voice
        available = ", ".join(str(voice.get("voice_id")) for voice in voices)
        raise RuntimeError(f"Voice '{voice_id}' was not found. Available voices: {available}")
    return voices[0]


def _parse_stream_response(
    server_base_url: str,
    *,
    voice_id: str,
    text: str,
    request_index: int,
    barrier: threading.Barrier,
    timeout: float,
) -> RequestResult:
    payload = json.dumps({"text": text, "voice_id": voice_id}).encode("utf-8")
    request = urllib_request.Request(
        f"{server_base_url.rstrip('/')}/api/v1/synthesize/stream",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/x-ndjson",
        },
        method="POST",
    )

    barrier.wait()
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
            request_index=request_index,
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
            request_index=request_index,
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
        request_index=request_index,
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
    started_event: threading.Event,
    poll_interval: float,
    health_samples: list[dict[str, Any]],
) -> None:
    started_event.wait(timeout=10.0)
    while not stop_event.is_set():
        local_started = time.perf_counter()
        try:
            payload = _json_request(f"{server_base_url.rstrip('/')}/api/health", timeout=max(5.0, poll_interval * 4))
            dashboard = payload.get("dashboard") or {}
            current_batch = dashboard.get("current_batch") or {}
            health_samples.append(
                {
                    "polled_at_perf": local_started,
                    "timestamp": payload.get("dashboard", {}).get("timestamp"),
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


def _percentile(values: list[float | None], fraction: float) -> float | None:
    numeric = sorted(float(value) for value in values if value is not None)
    if not numeric:
        return None
    if len(numeric) == 1:
        return round(numeric[0], 2)
    index = max(0, min(len(numeric) - 1, round((len(numeric) - 1) * fraction)))
    return round(numeric[index], 2)


def _summarize(
    *,
    server_base_url: str,
    voice: dict[str, Any],
    text: str,
    request_results: list[RequestResult],
    health_samples: list[dict[str, Any]],
    total_wall_seconds: float,
    count: int,
    concurrency: int,
) -> dict[str, Any]:
    successful = [result for result in request_results if result.success]
    failed = [result for result in request_results if not result.success]
    batch_sizes = [int(sample.get("current_batch_size") or 0) for sample in health_samples if "current_batch_size" in sample]
    streamed_batch_sizes = [
        int(batch_size)
        for result in request_results
        for batch_size in result.observed_batch_sizes
        if int(batch_size) > 0
    ]
    active_counts = [int(sample.get("active_request_count") or 0) for sample in health_samples if "active_request_count" in sample]
    queue_lengths = [int(sample.get("queue_length") or 0) for sample in health_samples if "queue_length" in sample]
    throughputs = [float(sample.get("throughput_audio_sps") or 0.0) for sample in health_samples if "throughput_audio_sps" in sample]
    non_zero_batch_sizes = [size for size in batch_sizes if size > 0]
    all_non_zero_batch_sizes = sorted(set(non_zero_batch_sizes + streamed_batch_sizes))

    return {
        "server_base_url": server_base_url,
        "voice_id": voice.get("voice_id"),
        "voice_name": voice.get("name"),
        "text": text,
        "request_count": count,
        "concurrency": concurrency,
        "total_wall_seconds": round(total_wall_seconds, 2),
        "success_count": len(successful),
        "failure_count": len(failed),
        "mean_ttft_ms": _mean([result.ttft_ms for result in successful]),
        "p95_ttft_ms": _percentile([result.ttft_ms for result in successful], 0.95),
        "mean_total_ms": _mean([result.total_ms for result in successful]),
        "p95_total_ms": _percentile([result.total_ms for result in successful], 0.95),
        "mean_audio_duration_ms": _mean([result.audio_duration_ms for result in successful]),
        "mean_batch_count_per_request": _mean([float(result.batch_count) if result.batch_count is not None else None for result in successful]),
        "max_observed_batch_size": max(all_non_zero_batch_sizes) if all_non_zero_batch_sizes else 0,
        "max_observed_batch_size_health": max(non_zero_batch_sizes) if non_zero_batch_sizes else 0,
        "max_observed_batch_size_stream": max(streamed_batch_sizes) if streamed_batch_sizes else 0,
        "max_observed_active_requests": max(active_counts) if active_counts else 0,
        "max_observed_queue_length": max(queue_lengths) if queue_lengths else 0,
        "max_observed_realtime_factor": round(max(throughputs), 3) if throughputs else 0.0,
        "observed_non_zero_batch_sizes": all_non_zero_batch_sizes,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the running TADA HTTP server outside the browser.")
    parser.add_argument("--server", default=DEFAULT_SERVER_BASE_URL, help="Server base URL, e.g. http://127.0.0.1:7878")
    parser.add_argument("--voice-id", default="", help="Voice ID to use. Defaults to the first public voice.")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to synthesize for every request.")
    parser.add_argument("--count", type=int, default=32, help="Total number of requests to launch.")
    parser.add_argument("--concurrency", type=int, default=32, help="Maximum concurrent request worker threads.")
    parser.add_argument("--poll-interval", type=float, default=0.5, help="How often to poll /api/health in seconds.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout in seconds.")
    args = parser.parse_args()

    server_base_url = str(args.server).strip().rstrip("/")
    count = max(1, int(args.count))
    concurrency = max(1, min(int(args.concurrency), count))
    voice = _resolve_voice(server_base_url, args.voice_id.strip() or None)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = OUT_DIR / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Server: {server_base_url}")
    print(f"Voice: {voice.get('voice_id')} ({voice.get('name')})")
    print(f"Requests: {count} | Concurrency: {concurrency}")
    print(f"Text: {args.text}")
    print("")

    barrier = threading.Barrier(concurrency + 1)
    stop_event = threading.Event()
    started_event = threading.Event()
    health_samples: list[dict[str, Any]] = []

    poll_thread = threading.Thread(
        target=_poll_health,
        kwargs={
            "server_base_url": server_base_url,
            "stop_event": stop_event,
            "started_event": started_event,
            "poll_interval": max(0.1, float(args.poll_interval)),
            "health_samples": health_samples,
        },
        daemon=True,
    )
    poll_thread.start()

    request_results: list[RequestResult] = []
    started_event.set()
    started_at = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _parse_stream_response,
                server_base_url,
                voice_id=str(voice.get("voice_id")),
                text=str(args.text),
                request_index=request_index,
                barrier=barrier,
                timeout=float(args.timeout),
            )
            for request_index in range(1, count + 1)
        ]
        barrier.wait()
        for future in as_completed(futures):
            request_results.append(future.result())

    total_wall_seconds = time.perf_counter() - started_at
    stop_event.set()
    poll_thread.join(timeout=5.0)

    request_results.sort(key=lambda item: item.request_index)
    summary = _summarize(
        server_base_url=server_base_url,
        voice=voice,
        text=str(args.text),
        request_results=request_results,
        health_samples=health_samples,
        total_wall_seconds=total_wall_seconds,
        count=count,
        concurrency=concurrency,
    )

    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "request_results.json", [asdict(item) for item in request_results])
    _write_json(run_dir / "health_samples.json", health_samples)

    print("Summary")
    print(f"  Success: {summary['success_count']} / {summary['request_count']}")
    print(f"  Mean TTFT: {summary['mean_ttft_ms']} ms")
    print(f"  P95 TTFT: {summary['p95_ttft_ms']} ms")
    print(f"  Mean Total: {summary['mean_total_ms']} ms")
    print(f"  Mean Audio: {summary['mean_audio_duration_ms']} ms")
    print(f"  Max Observed Batch Size: {summary['max_observed_batch_size']}")
    print(f"  Max Observed Active Requests: {summary['max_observed_active_requests']}")
    print(f"  Max Observed Queue Length: {summary['max_observed_queue_length']}")
    print(f"  Max Observed Realtime: {summary['max_observed_realtime_factor']}x")
    print(f"  Non-zero Batch Sizes: {summary['observed_non_zero_batch_sizes']}")
    print(f"  Output: {run_dir}")


if __name__ == "__main__":
    main()
