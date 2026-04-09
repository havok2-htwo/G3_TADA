import { useEffect, useRef, useState } from "react";

import { createWavBlobFromInt16Chunks, decodePcm16Base64, saveBlobToFile } from "../shared/audio";
import { apiFetch, formatDate, formatMs, formatSeconds, streamNdjson } from "../shared/api";

const CLIENT_KEY_STORAGE = "tada_client_key";

function nowStamp() {
  const value = new Date();
  const parts = [
    value.getFullYear(),
    String(value.getMonth() + 1).padStart(2, "0"),
    String(value.getDate()).padStart(2, "0"),
    "-",
    String(value.getHours()).padStart(2, "0"),
    String(value.getMinutes()).padStart(2, "0"),
    String(value.getSeconds()).padStart(2, "0"),
  ];
  return parts.join("");
}

export default function DemoApp() {
  const [clientKey, setClientKey] = useState(() => sessionStorage.getItem(CLIENT_KEY_STORAGE) || "");
  const [clientKeyInput, setClientKeyInput] = useState(() => sessionStorage.getItem(CLIENT_KEY_STORAGE) || "");
  const [voices, setVoices] = useState([]);
  const [selectedVoiceId, setSelectedVoiceId] = useState("");
  const [text, setText] = useState("Hallo! Das ist ein echter Batch-Streaming-Test fuer den neuen TADA-Server.");
  const [singleResult, setSingleResult] = useState(null);
  const [singleMetrics, setSingleMetrics] = useState(null);
  const [benchmarkCount, setBenchmarkCount] = useState(8);
  const [directoryHandle, setDirectoryHandle] = useState(null);
  const [directoryLabel, setDirectoryLabel] = useState("");
  const [benchmarkResults, setBenchmarkResults] = useState([]);
  const [benchmarkSummary, setBenchmarkSummary] = useState(null);
  const [runningSingle, setRunningSingle] = useState(false);
  const [runningBenchmark, setRunningBenchmark] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const audioContextRef = useRef(null);
  const nextPlaybackTimeRef = useRef(0);
  const singleAbortRef = useRef(null);

  useEffect(() => {
    if (!clientKey) {
      return;
    }
    apiFetch("/api/v1/voices", { clientKey })
      .then((data) => {
        setVoices(data.voices || []);
        if (!selectedVoiceId && data.voices?.length) {
          setSelectedVoiceId(data.voices[0].voice_id);
        }
      })
      .catch((loadError) => setError(loadError.message));
  }, [clientKey, selectedVoiceId]);

  useEffect(() => () => {
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
    }
  }, []);

  async function ensureAudioContext(sampleRate) {
    const AudioContextImpl = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextImpl) {
      throw new Error("Web Audio is not available in this browser.");
    }
    if (audioContextRef.current && audioContextRef.current.sampleRate !== sampleRate) {
      await audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
      nextPlaybackTimeRef.current = 0;
    }
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContextImpl({ sampleRate });
      nextPlaybackTimeRef.current = 0;
    }
    if (audioContextRef.current.state === "suspended") {
      await audioContextRef.current.resume();
    }
    return audioContextRef.current;
  }

  async function queuePlayback(float32, sampleRate) {
    const audioContext = await ensureAudioContext(sampleRate);
    const buffer = audioContext.createBuffer(1, float32.length, sampleRate);
    buffer.getChannelData(0).set(float32);
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);
    const startAt = Math.max(audioContext.currentTime + 0.08, nextPlaybackTimeRef.current);
    source.start(startAt);
    nextPlaybackTimeRef.current = startAt + buffer.duration;
  }

  function persistClientKey(nextKey) {
    sessionStorage.setItem(CLIENT_KEY_STORAGE, nextKey);
    setClientKey(nextKey);
    setClientKeyInput(nextKey);
  }

  async function runStreamingJob({ playback = false, signal } = {}) {
    if (!selectedVoiceId) {
      throw new Error("Please choose a voice first.");
    }

    const startedAt = performance.now();
    let sampleRate = 24000;
    let ttftMs = null;
    let firstPlaybackMs = null;
    let lastEmittedAudioMs = 0;
    const int16Chunks = [];
    let finalResult = null;

    await streamNdjson("/api/v1/synthesize/stream", {
      clientKey,
      body: { text, voice_id: selectedVoiceId },
      signal,
      onEvent: async (event) => {
        if (event.type === "chunk") {
          sampleRate = event.sample_rate || sampleRate;
          if (ttftMs === null) {
            ttftMs = performance.now() - startedAt;
          }
          const decoded = decodePcm16Base64(event.pcm16_b64);
          int16Chunks.push(decoded.int16);
          lastEmittedAudioMs = event.emitted_audio_ms || lastEmittedAudioMs;
          if (playback) {
            await queuePlayback(decoded.float32, sampleRate);
            if (firstPlaybackMs === null) {
              firstPlaybackMs = performance.now() - startedAt;
            }
          }
          return;
        }
        if (event.type === "done") {
          finalResult = event.result;
          return;
        }
        if (event.type === "error") {
          throw new Error(event.message || "Streaming failed.");
        }
      },
    });

    const totalMs = performance.now() - startedAt;
    return {
      success: true,
      sampleRate,
      int16Chunks,
      result: finalResult,
      ttftMs,
      firstPlaybackMs,
      totalMs,
      audioDurationMs: lastEmittedAudioMs,
    };
  }

  async function resolveResultBlob(outcome) {
    if (outcome?.result?.audio_url) {
      try {
        return await apiFetch(outcome.result.audio_url, { clientKey, responseType: "blob" });
      } catch {
        return null;
      }
    }
    return null;
  }

  async function handleSingleRun() {
    if (runningSingle) {
      singleAbortRef.current?.abort();
      return;
    }
    setMessage("");
    setError("");
    setRunningSingle(true);
    setSingleResult(null);
    setSingleMetrics(null);
    nextPlaybackTimeRef.current = 0;

    const controller = new AbortController();
    singleAbortRef.current = controller;

    try {
      const outcome = await runStreamingJob({ playback: true, signal: controller.signal });
      const blob = await resolveResultBlob(outcome) || createWavBlobFromInt16Chunks(outcome.int16Chunks, outcome.sampleRate);
      const audioUrl = URL.createObjectURL(blob);
      setSingleResult({
        ...(outcome.result || {}),
        local_audio_url: audioUrl,
      });
      setSingleMetrics(outcome);
      setMessage("Single live stream finished.");
    } catch (runError) {
      if (!controller.signal.aborted) {
        setError(runError.message);
      }
    } finally {
      if (singleAbortRef.current === controller) {
        singleAbortRef.current = null;
      }
      setRunningSingle(false);
    }
  }

  async function pickDirectory() {
    if (!window.showDirectoryPicker) {
      setError("This browser does not support the File System Access API.");
      return;
    }
    const handle = await window.showDirectoryPicker();
    setDirectoryHandle(handle);
    setDirectoryLabel(handle.name);
  }

  async function handleBenchmarkRun() {
    if (!directoryHandle) {
      setError("Choose a target directory first.");
      return;
    }
    setRunningBenchmark(true);
    setError("");
    setMessage("");
    setBenchmarkResults([]);
    setBenchmarkSummary(null);
    try {
      const runDirectory = await directoryHandle.getDirectoryHandle(`demo-run-${nowStamp()}`, { create: true });
      const requests = Array.from({ length: Number(benchmarkCount) }, (_, index) => index + 1);
      const results = await Promise.all(
        requests.map(async (requestIndex) => {
          try {
            const outcome = await runStreamingJob({ playback: false });
            const wavBlob = await resolveResultBlob(outcome) || createWavBlobFromInt16Chunks(outcome.int16Chunks, outcome.sampleRate);
            const fileHandle = await runDirectory.getFileHandle(
              `request-${String(requestIndex).padStart(4, "0")}.wav`,
              { create: true },
            );
            await saveBlobToFile(fileHandle, wavBlob);
            return { requestIndex, ...outcome };
          } catch (runError) {
            return {
              requestIndex,
              success: false,
              error: runError.message,
              ttftMs: null,
              totalMs: null,
              audioDurationMs: null,
            };
          }
        }),
      );

      const metricsLines = results.map((item) => [
        `request=${item.requestIndex}`,
        `success=${item.success}`,
        `ttft_ms=${item.ttftMs ?? "-"}`,
        `total_ms=${item.totalMs ?? "-"}`,
        `audio_duration_ms=${item.audioDurationMs ?? "-"}`,
        item.error ? `error=${item.error}` : "",
      ].filter(Boolean).join(" | "));
      const metricsHandle = await runDirectory.getFileHandle("metrics.txt", { create: true });
      await saveBlobToFile(metricsHandle, new Blob([metricsLines.join("\n")], { type: "text/plain" }));

      const successful = results.filter((item) => item.success);
      const meanTtft = successful.length
        ? successful.reduce((sum, item) => sum + (item.ttftMs || 0), 0) / successful.length
        : null;
      setBenchmarkResults(results);
      setBenchmarkSummary({
        started_at: new Date().toISOString(),
        count: results.length,
        success_count: successful.length,
        mean_ttft_ms: meanTtft,
        output_directory: runDirectory.name,
      });
      setMessage(`Benchmark finished. Files were written into ${runDirectory.name}.`);
    } catch (runError) {
      setError(runError.message);
    } finally {
      setRunningBenchmark(false);
    }
  }

  if (!clientKey) {
    return (
      <main className="auth-shell">
        <section className="panel stack">
          <p className="eyebrow">Demo Client Login</p>
          <h1>TADA Demo Client</h1>
          <p className="muted">Enter a client API key created in the admin panel.</p>
          <input value={clientKeyInput} onChange={(event) => setClientKeyInput(event.target.value)} placeholder="tada_client_..." />
          <button type="button" onClick={() => persistClientKey(clientKeyInput.trim())} disabled={!clientKeyInput.trim()}>
            Open Demo Client
          </button>
        </section>
      </main>
    );
  }

  return (
    <main className="page-shell">
      <section className="hero">
        <div>
          <p className="eyebrow">Public Demo Client</p>
          <h1>Streaming and Parallel Load Testing</h1>
          <p className="hero-copy">
            This UI uses the same public client API that other projects will call. Single request mode plays live audio, while benchmark mode stores WAV files and metrics locally in your selected folder.
          </p>
          <div className="button-row" style={{ marginTop: 16 }}>
            <button type="button" className="ghost" onClick={() => {
              sessionStorage.removeItem(CLIENT_KEY_STORAGE);
              setClientKey("");
              setClientKeyInput("");
            }}>
              Logout
            </button>
          </div>
        </div>

        <div className="status-grid">
          <div className="status-pill"><span>Voices</span><strong>{voices.length}</strong></div>
          <div className="status-pill"><span>Selected Voice</span><strong>{selectedVoiceId || "-"}</strong></div>
          <div className="status-pill"><span>Folder</span><strong>{directoryLabel || "not selected"}</strong></div>
          <div className="status-pill"><span>Benchmark Count</span><strong>{benchmarkCount}</strong></div>
        </div>
      </section>

      {message ? <div className="message success">{message}</div> : null}
      {error ? <div className="message error">{error}</div> : null}

      <section className="panel-grid">
        <section className="panel stack">
          <p className="eyebrow">Request</p>
          <h2>Single Live Stream</h2>
          <label>
            Voice
            <select value={selectedVoiceId} onChange={(event) => setSelectedVoiceId(event.target.value)}>
              <option value="">Choose voice</option>
              {voices.map((voice) => (
                <option key={voice.voice_id} value={voice.voice_id}>
                  {voice.name} ({voice.language})
                </option>
              ))}
            </select>
          </label>
          <label>
            Text
            <textarea value={text} onChange={(event) => setText(event.target.value)} />
          </label>
          <div className="button-row">
            <button type="button" onClick={handleSingleRun} disabled={!selectedVoiceId}>
              {runningSingle ? "Stop Stream" : "Start Live Stream"}
            </button>
          </div>
          {singleMetrics ? (
            <div className="card">
              <strong>Single Request Metrics</strong>
              <div className="muted">TTFAudio (first received audio): {formatMs(singleMetrics.ttftMs)}</div>
              <div className="muted">First playback: {formatMs(singleMetrics.firstPlaybackMs)}</div>
              <div className="muted">Total time: {formatMs(singleMetrics.totalMs)}</div>
              <div className="muted">Audio duration: {formatMs(singleMetrics.audioDurationMs)}</div>
              {singleResult?.local_audio_url ? <audio controls src={singleResult.local_audio_url} /> : null}
            </div>
          ) : null}
        </section>

        <section className="panel stack">
          <p className="eyebrow">Benchmark</p>
          <h2>Parallel Multi-Request Run</h2>
          <label>
            Parallel Requests
            <input type="number" min="1" max="100" value={benchmarkCount} onChange={(event) => setBenchmarkCount(Math.max(1, Math.min(100, Number(event.target.value) || 1)))} />
          </label>
          <div className="button-row">
            <button type="button" className="secondary" onClick={pickDirectory}>
              Choose Output Folder
            </button>
            <button type="button" onClick={handleBenchmarkRun} disabled={!directoryHandle || !selectedVoiceId || runningBenchmark}>
              {runningBenchmark ? "Running..." : "Run Benchmark"}
            </button>
          </div>
          <div className="card">
            <div className="muted">Selected folder: {directoryLabel || "none"}</div>
            <div className="muted">A subfolder per run will be created and will contain WAV files plus a metrics.txt report.</div>
          </div>
          {benchmarkSummary ? (
            <div className="card">
              <strong>Last Benchmark</strong>
              <div className="muted">Started: {formatDate(benchmarkSummary.started_at)}</div>
              <div className="muted">Requests: {benchmarkSummary.count}</div>
              <div className="muted">Succeeded: {benchmarkSummary.success_count}</div>
              <div className="muted">Mean TTFAudio: {formatMs(benchmarkSummary.mean_ttft_ms)}</div>
              <div className="muted">Folder: {benchmarkSummary.output_directory}</div>
            </div>
          ) : null}
        </section>
      </section>

      <section className="panel">
        <p className="eyebrow">Results</p>
        <h2>Last Benchmark Requests</h2>
        <table className="table">
          <thead>
            <tr>
              <th>Request</th>
              <th>Success</th>
              <th>TTFAudio</th>
              <th>Total</th>
              <th>Audio</th>
              <th>Error</th>
            </tr>
          </thead>
          <tbody>
            {benchmarkResults.map((item) => (
              <tr key={item.requestIndex}>
                <td>{item.requestIndex}</td>
                <td>{String(item.success)}</td>
                <td>{formatMs(item.ttftMs)}</td>
                <td>{formatMs(item.totalMs)}</td>
                <td>{formatMs(item.audioDurationMs)}</td>
                <td>{item.error || "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </main>
  );
}
