import { useEffect, useRef, useState } from "react";

const initialVoiceForm = {
  name: "",
  language: "de",
  transcript: "",
  file: null,
};

const initialStreamState = {
  active: false,
  chunks: 0,
  step: 0,
  decodedSeconds: 0,
  emittedSeconds: 0,
  sampleRate: 24000,
  bufferMs: 500,
};

const STREAM_BUFFER_MS = 500;
const STREAM_PROGRESS_INTERVAL = 24;

async function apiFetch(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || data.message || "Request failed");
  }
  return data;
}

function formatDate(value) {
  return new Date(value).toLocaleString("de-DE");
}

function formatDuration(value) {
  return `${Number(value).toFixed(2)} s`;
}

function decodePcm16(base64) {
  const binary = window.atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }

  const view = new DataView(bytes.buffer);
  const samples = new Float32Array(bytes.byteLength / 2);
  for (let i = 0; i < samples.length; i += 1) {
    samples[i] = view.getInt16(i * 2, true) / 32768;
  }
  return samples;
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [voices, setVoices] = useState([]);
  const [languages, setLanguages] = useState({});
  const [voiceForm, setVoiceForm] = useState(initialVoiceForm);
  const [selectedVoiceId, setSelectedVoiceId] = useState("");
  const [text, setText] = useState(
    "Hallo! Das ist ein erster Test mit TADA und einer geklonten Stimme.",
  );
  const [steps, setSteps] = useState(10);
  const [generated, setGenerated] = useState([]);
  const [loadingState, setLoadingState] = useState({ voice: false, generate: false, stream: false, model: false });
  const [streamState, setStreamState] = useState(initialStreamState);
  const [error, setError] = useState("");

  const audioContextRef = useRef(null);
  const nextPlaybackTimeRef = useRef(0);
  const streamAbortRef = useRef(null);

  async function refresh() {
    setError("");
    try {
      const [healthData, voiceData] = await Promise.all([
        apiFetch("/api/health"),
        apiFetch("/api/voices"),
      ]);
      setHealth(healthData);
      setVoices(voiceData.voices || []);
      setLanguages(voiceData.languages || {});
      if (!selectedVoiceId && (voiceData.voices || []).length > 0) {
        setSelectedVoiceId(voiceData.voices[0].voice_id);
      }
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function ensureAudioContext(sampleRate) {
    const AudioContextImpl = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextImpl) {
      throw new Error("Dein Browser unterstuetzt Web Audio nicht.");
    }

    if (audioContextRef.current && audioContextRef.current.sampleRate !== sampleRate) {
      try {
        await audioContextRef.current.close();
      } catch {
        // ignore
      }
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

  async function stopStreamPlayback() {
    if (streamAbortRef.current) {
      streamAbortRef.current.abort();
      streamAbortRef.current = null;
    }

    if (audioContextRef.current) {
      try {
        await audioContextRef.current.close();
      } catch {
        // ignore
      }
      audioContextRef.current = null;
    }

    nextPlaybackTimeRef.current = 0;
    setStreamState(initialStreamState);
    setLoadingState((current) => ({ ...current, stream: false }));
  }

  async function queueAudioChunk(base64, sampleRate) {
    const samples = decodePcm16(base64);
    if (samples.length === 0) {
      return;
    }

    const audioContext = await ensureAudioContext(sampleRate);
    const audioBuffer = audioContext.createBuffer(1, samples.length, sampleRate);
    audioBuffer.getChannelData(0).set(samples);

    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);

    const startAt = Math.max(audioContext.currentTime + 0.08, nextPlaybackTimeRef.current);
    source.start(startAt);
    nextPlaybackTimeRef.current = startAt + audioBuffer.duration;
  }

  async function handleCreateVoice(event) {
    event.preventDefault();
    if (!voiceForm.file) {
      setError("Bitte eine Referenzaufnahme hochladen.");
      return;
    }

    const formData = new FormData();
    formData.append("name", voiceForm.name || "Meine Stimme");
    formData.append("language", voiceForm.language);
    formData.append("transcript", voiceForm.transcript);
    formData.append("audio", voiceForm.file);

    setLoadingState((current) => ({ ...current, voice: true }));
    setError("");

    try {
      const data = await apiFetch("/api/voices", {
        method: "POST",
        body: formData,
      });
      setVoiceForm(initialVoiceForm);
      setSelectedVoiceId(data.voice.voice_id);
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingState((current) => ({ ...current, voice: false }));
    }
  }

  async function handleModelChange(event) {
    const modelName = event.target.value;
    if (!modelName || modelName === health?.model_name) {
      return;
    }

    const previousModelName = health?.model_name || "";
    setLoadingState((current) => ({ ...current, model: true }));
    setError("");
    setHealth((current) => (current ? { ...current, model_name: modelName, model_loaded: false } : current));

    try {
      const data = await apiFetch("/api/model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: modelName }),
      });
      setHealth(data.status);
    } catch (err) {
      setHealth((current) =>
        current ? { ...current, model_name: previousModelName || current.model_name } : current,
      );
      setError(err.message);
    } finally {
      setLoadingState((current) => ({ ...current, model: false }));
    }
  }

  async function handleGenerate(event) {
    event.preventDefault();
    if (!selectedVoiceId) {
      setError("Bitte zuerst eine Stimme anlegen oder auswaehlen.");
      return;
    }

    setLoadingState((current) => ({ ...current, generate: true }));
    setError("");

    try {
      const data = await apiFetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          voice_id: selectedVoiceId,
          steps: Number(steps),
        }),
      });
      setGenerated((current) => [data, ...current].slice(0, 8));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingState((current) => ({ ...current, generate: false }));
    }
  }

  async function handleGenerateStream() {
    if (loadingState.stream) {
      await stopStreamPlayback();
      return;
    }

    if (!selectedVoiceId) {
      setError("Bitte zuerst eine Stimme anlegen oder auswaehlen.");
      return;
    }

    await stopStreamPlayback();

    const controller = new AbortController();
    streamAbortRef.current = controller;
    setLoadingState((current) => ({ ...current, stream: true }));
    setStreamState({ ...initialStreamState, active: true, bufferMs: STREAM_BUFFER_MS });
    setError("");

    try {
      const response = await fetch("/api/generate/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          voice_id: selectedVoiceId,
          steps: Number(steps),
          buffer_ms: STREAM_BUFFER_MS,
          progress_interval_steps: STREAM_PROGRESS_INTERVAL,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.detail || data.message || "Streaming request failed");
      }
      if (!response.body) {
        throw new Error("Streaming-Antwort enthaelt keinen Body.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      const handleEvent = async (payload) => {
        if (payload.type === "start") {
          setStreamState((current) => ({
            ...current,
            active: true,
            sampleRate: payload.sample_rate,
            bufferMs: payload.buffer_ms,
          }));
          await ensureAudioContext(payload.sample_rate);
          return;
        }

        if (payload.type === "progress") {
          setStreamState((current) => ({
            ...current,
            active: true,
            step: payload.step,
            decodedSeconds: payload.decoded_seconds,
            emittedSeconds: payload.emitted_seconds,
          }));
          return;
        }

        if (payload.type === "chunk") {
          await queueAudioChunk(payload.pcm16_b64, payload.sample_rate);
          setStreamState((current) => ({
            ...current,
            active: true,
            chunks: current.chunks + 1,
            step: payload.step,
            sampleRate: payload.sample_rate,
            emittedSeconds: payload.emitted_seconds,
          }));
          return;
        }

        if (payload.type === "done") {
          setGenerated((current) => [payload.result, ...current].slice(0, 8));
          return;
        }

        if (payload.type === "error") {
          throw new Error(payload.message || "Streaming fehlgeschlagen.");
        }
      };

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        let newlineIndex = buffer.indexOf("\n");
        while (newlineIndex >= 0) {
          const line = buffer.slice(0, newlineIndex).trim();
          buffer = buffer.slice(newlineIndex + 1);
          if (line) {
            await handleEvent(JSON.parse(line));
          }
          newlineIndex = buffer.indexOf("\n");
        }
      }

      const tail = buffer.trim();
      if (tail) {
        await handleEvent(JSON.parse(tail));
      }
    } catch (err) {
      if (controller.signal.aborted) {
        return;
      }
      setError(err.message || "Streaming fehlgeschlagen.");
    } finally {
      if (streamAbortRef.current === controller) {
        streamAbortRef.current = null;
      }
      setLoadingState((current) => ({ ...current, stream: false }));
      setStreamState((current) => ({ ...current, active: false }));
    }
  }

  return (
    <main className="app-shell">
      <section className="hero-card">
        <div>
          <p className="eyebrow">HumeAI TADA</p>
          <h1>Simple Voice-Clone UI</h1>
          <p className="hero-copy">
            Referenzstimme hochladen, Prompt cachen und danach beliebigen Text mit derselben
            Stimme generieren.
          </p>
        </div>

        <div className="status-grid">
          <div className="status-pill">
            <span>Modell</span>
            <strong>{health?.model_name || "lade..."}</strong>
          </div>
          <div className="status-pill">
            <span>Device</span>
            <strong>{health?.device || "lade..."}</strong>
          </div>
          <div className="status-pill">
            <span>GPU</span>
            <strong>{health?.gpu_name || "CPU"}</strong>
          </div>
          <div className="status-pill">
            <span>Encoder</span>
            <strong>{health?.encoder_device || "lade..."}</strong>
          </div>
        </div>

        {!health?.hf_token_present ? (
          <div className="warning-box">
            Fuer Voice Clone brauchst du Zugriff auf meta-llama/Llama-3.2-1B und einen gesetzten HF_TOKEN oder einen vorhandenen huggingface-cli login.
          </div>
        ) : null}
        {health?.model_error ? <div className="warning-box">{health.model_error}</div> : null}
        {error ? <div className="error-box">{error}</div> : null}
      </section>

      <section className="panel-grid">
        <form className="panel-card" onSubmit={handleCreateVoice}>
          <div className="panel-head">
            <p className="eyebrow">1. Stimme anlegen</p>
            <h2>Custom Voice Clone</h2>
          </div>

          <label>
            Name
            <input
              type="text"
              placeholder="z. B. Meine Teststimme"
              value={voiceForm.name}
              onChange={(event) =>
                setVoiceForm((current) => ({ ...current, name: event.target.value }))
              }
            />
          </label>

          <label>
            Sprache
            <select
              value={voiceForm.language}
              onChange={(event) =>
                setVoiceForm((current) => ({ ...current, language: event.target.value }))
              }
            >
              {Object.entries(languages).map(([code, label]) => (
                <option key={code} value={code}>
                  {label} ({code})
                </option>
              ))}
            </select>
          </label>

          <label>
            Transkript der Referenz
            <textarea
              rows="5"
              placeholder="Fuer Deutsch und alle nicht-englischen Sprachen bitte den exakten gesprochenen Text eintragen."
              value={voiceForm.transcript}
              onChange={(event) =>
                setVoiceForm((current) => ({ ...current, transcript: event.target.value }))
              }
            />
          </label>

          <label className="file-input">
            Referenz-Audio
            <input
              type="file"
              accept="audio/*,.wav,.mp3,.flac,.m4a"
              onChange={(event) =>
                setVoiceForm((current) => ({
                  ...current,
                  file: event.target.files?.[0] || null,
                }))
              }
            />
          </label>

          <button type="submit" disabled={loadingState.voice || loadingState.stream}>
            {loadingState.voice ? "Stimme wird verarbeitet..." : "Stimme speichern"}
          </button>
        </form>

        <form className="panel-card" onSubmit={handleGenerate}>
          <div className="panel-head">
            <p className="eyebrow">2. Generieren</p>
            <h2>Text zu Sprache</h2>
          </div>

          <label>
            Stimme
            <select
              value={selectedVoiceId}
              onChange={(event) => setSelectedVoiceId(event.target.value)}
            >
              <option value="">Bitte Stimme waehlen</option>
              {voices.map((voice) => (
                <option key={voice.voice_id} value={voice.voice_id}>
                  {voice.name} ({voice.language})
                </option>
              ))}
            </select>
          </label>

          <label>
            Modell
            <select
              value={health?.model_name || ""}
              disabled={loadingState.model || loadingState.voice || loadingState.generate || loadingState.stream}
              onChange={handleModelChange}
            >
              {(health?.available_models || []).map((model) => (
                <option key={model.id} value={model.id}>
                  {model.label}
                </option>
              ))}
            </select>
            <span className="field-help">
              {(health?.available_models || []).find((model) => model.id === health?.model_name)?.description ||
                "Waehle zwischen mehr Qualitaet (3B) und mehr Geschwindigkeit (1B)."}
            </span>
          </label>

          <label>
            Zieltext
            <textarea rows="7" value={text} onChange={(event) => setText(event.target.value)} />
          </label>

          <label>
            Generierungs-Schritte (Steps)
            <input
              type="number"
              min="1"
              max="150"
              value={steps}
              onChange={(event) => setSteps(event.target.value)}
            />
            <span className="field-help">
              Standard: 10. Auf 8 GB VRAM ist das meist der beste Speed/Qualitaet-Kompromiss. Mehr = deutlich langsamer.
            </span>
          </label>

          <div className="button-row">
            <button type="submit" disabled={loadingState.generate || loadingState.stream || loadingState.model}>
              {loadingState.generate ? "Audio wird generiert..." : "Audio erzeugen"}
            </button>
            <button
              type="button"
              className="secondary-button"
              disabled={loadingState.generate || loadingState.model}
              onClick={handleGenerateStream}
            >
              {loadingState.stream ? "Live-Stream stoppen" : "Live-Stream starten"}
            </button>
          </div>

          <div className="stream-box">
            <div>
              <strong>Live-Streaming</strong>
              <p className="muted stream-copy">
                Der Server schiebt sichere Audio-Prefixe mit ca. {STREAM_BUFFER_MS} ms Rueckhalte-Puffer in den Browser, waehrend die restliche Generierung weiterlaeuft.
              </p>
            </div>
            <div className="stream-metrics">
              <span>Status: {streamState.active ? "aktiv" : loadingState.stream ? "startet" : "bereit"}</span>
              <span>Chunks: {streamState.chunks}</span>
              <span>Schritt: {streamState.step}</span>
              <span>Decodiert: {formatDuration(streamState.decodedSeconds)}</span>
              <span>Ausgespielt: {formatDuration(streamState.emittedSeconds)}</span>
            </div>
          </div>
        </form>
      </section>

      <section className="panel-grid">
        <div className="panel-card">
          <div className="panel-head">
            <p className="eyebrow">Gespeicherte Stimmen</p>
            <h2>Voice Library</h2>
          </div>

          <div className="voice-list">
            {voices.length === 0 ? (
              <p className="muted">Noch keine Stimme angelegt.</p>
            ) : (
              voices.map((voice) => (
                <article className="voice-card" key={voice.voice_id}>
                  <div>
                    <h3>{voice.name}</h3>
                    <p className="muted">
                      {languages[voice.language] || voice.language} - {formatDuration(voice.duration_seconds)}
                    </p>
                    <p className="muted">{formatDate(voice.created_at)}</p>
                  </div>
                  <audio controls src={voice.reference_url} />
                </article>
              ))
            )}
          </div>
        </div>

        <div className="panel-card">
          <div className="panel-head">
            <p className="eyebrow">Ergebnisse</p>
            <h2>Generierte Clips</h2>
          </div>

          <div className="voice-list">
            {generated.length === 0 ? (
              <p className="muted">Hier tauchen deine letzten Generierungen auf.</p>
            ) : (
              generated.map((item) => (
                <article className="voice-card" key={item.generation_id}>
                  <div>
                    <h3>{item.voice_id}</h3>
                    <p className="muted">
                      {item.model_name} - {formatDuration(item.duration_seconds)}
                    </p>
                    {item.rtf !== undefined && item.rtf !== null && (
                      <p className="muted" style={{ fontSize: "0.85rem", marginTop: "4px" }}>
                        GPU Zeit: {formatDuration(item.processing_time)} | RTF: {item.rtf.toFixed(2)}x
                      </p>
                    )}
                    <p style={{ marginTop: "12px" }}>{item.text}</p>
                  </div>
                  <audio controls src={item.audio_url} />
                </article>
              ))
            )}
          </div>
        </div>
      </section>
    </main>
  );
}

