import { useEffect, useMemo, useRef, useState } from "react";

import { apiFetch, formatDate, formatMs, formatSeconds, loadProtectedAudioUrl } from "../shared/api";

const ADMIN_KEY_STORAGE = "tada_admin_key";
const MIN_REFERENCE_SECONDS = 3;
const MAX_REFERENCE_SECONDS = 14.5;
const DEFAULT_REFERENCE_SECONDS = 10;
const REFERENCE_TAIL_SILENCE_SECONDS = 0.5;

const initialVoiceForm = {
  name: "",
  language: "de",
  transcript: "",
  file: null,
  trimStartSeconds: "0.00",
  trimEndSeconds: "",
};

const EMPTY_VOICE_PREVIEW = {
  url: "",
  durationMs: 0,
  waveform: [],
  decodeError: "",
};

function formatTrimSeconds(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "";
  }
  return (Number(value) / 1000).toFixed(2);
}

function formatReferenceLimit(value) {
  if (!Number.isFinite(Number(value))) {
    return "-";
  }
  return Number.isInteger(Number(value)) ? Number(value).toFixed(0) : Number(value).toFixed(1);
}

function parseSecondsToMs(value) {
  if (value === null || value === undefined || String(value).trim() === "") {
    return null;
  }
  const number = Number(String(value).replace(",", "."));
  if (!Number.isFinite(number)) {
    return null;
  }
  return Math.max(0, Math.round(number * 1000));
}

function computeTrimSelection(startMs, endMs, durationMs) {
  const safeDurationMs = Math.max(1, Number(durationMs || 0));
  const minDurationMs = MIN_REFERENCE_SECONDS * 1000;
  const maxDurationMs = MAX_REFERENCE_SECONDS * 1000;
  const sourceTooShort = safeDurationMs < minDurationMs;
  const effectiveMinMs = sourceTooShort ? Math.max(100, safeDurationMs) : minDurationMs;
  const maxStartMs = Math.max(0, safeDurationMs - effectiveMinMs);
  let nextStart = Math.min(Math.max(0, Number(startMs || 0)), maxStartMs);
  let nextEnd = Number(endMs || 0);
  if (!Number.isFinite(nextEnd) || nextEnd <= 0) {
    nextEnd = Math.min(safeDurationMs, DEFAULT_REFERENCE_SECONDS * 1000);
  }
  nextEnd = Math.min(Math.max(nextStart + 100, nextEnd), safeDurationMs);
  if (nextEnd - nextStart > maxDurationMs) {
    nextEnd = nextStart + maxDurationMs;
  }
  if (nextEnd - nextStart < effectiveMinMs) {
    nextEnd = Math.min(safeDurationMs, nextStart + effectiveMinMs);
    if (nextEnd - nextStart < effectiveMinMs) {
      nextStart = Math.max(0, nextEnd - effectiveMinMs);
    }
  }
  const durationSelectedMs = nextEnd - nextStart;
  return {
    startMs: nextStart,
    endMs: nextEnd,
    durationMs: durationSelectedMs,
    sourceTooShort,
    isValid: !sourceTooShort && durationSelectedMs >= minDurationMs && durationSelectedMs <= maxDurationMs,
  };
}

function buildWaveformSamples(channelData, bucketCount = 160) {
  if (!channelData || channelData.length === 0) {
    return [];
  }
  const samples = [];
  const bucketSize = Math.max(1, Math.floor(channelData.length / bucketCount));
  for (let index = 0; index < bucketCount; index += 1) {
    const start = index * bucketSize;
    const end = Math.min(channelData.length, start + bucketSize);
    let peak = 0;
    for (let cursor = start; cursor < end; cursor += 1) {
      const value = Math.abs(channelData[cursor]);
      if (value > peak) {
        peak = value;
      }
    }
    samples.push(peak);
  }
  return samples;
}

function sliceWaveformSamples(samples, durationMs, startMs, endMs) {
  if (!samples || samples.length === 0 || !durationMs) {
    return [];
  }
  const safeStart = Math.max(0, Number(startMs || 0));
  const safeEnd = Math.max(safeStart, Math.min(Number(durationMs), Number(endMs || durationMs)));
  const startIndex = Math.max(0, Math.floor((safeStart / durationMs) * samples.length));
  const endIndex = Math.min(samples.length, Math.max(startIndex + 8, Math.ceil((safeEnd / durationMs) * samples.length)));
  return samples.slice(startIndex, endIndex);
}

function WaveformPreview({ samples, startRatio, endRatio, viewportStartRatio = null, viewportEndRatio = null, playheadRatio = null }) {
  if (!samples || samples.length === 0) {
    return <div className="waveform-empty muted">Waveform preview will appear here after the audio has been decoded.</div>;
  }

  const width = 100;
  const height = 44;
  const step = width / Math.max(1, samples.length - 1);
  const points = samples
    .map((value, index) => {
      const x = index * step;
      const amplitude = Math.max(0.05, Math.min(1, Number(value || 0)));
      const yTop = (height / 2) - amplitude * (height / 2 - 2);
      const yBottom = (height / 2) + amplitude * (height / 2 - 2);
      return `${x.toFixed(2)},${yTop.toFixed(2)} ${x.toFixed(2)},${yBottom.toFixed(2)}`;
    })
    .join(" ");

  const clampedStart = Math.max(0, Math.min(1, Number(startRatio || 0)));
  const clampedEnd = Math.max(clampedStart, Math.min(1, Number(endRatio || 1)));
  const viewportStart = viewportStartRatio === null ? null : Math.max(0, Math.min(1, Number(viewportStartRatio || 0)));
  const viewportEnd = viewportEndRatio === null ? null : Math.max(viewportStart ?? 0, Math.min(1, Number(viewportEndRatio || 1)));
  const playhead = playheadRatio === null ? null : Math.max(0, Math.min(1, Number(playheadRatio || 0)));

  return (
    <svg className="waveform" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <rect x="0" y="0" width={width} height={height} rx="10" className="waveform-bg" />
      {viewportStart !== null && viewportEnd !== null ? (
        <rect
          x={(viewportStart * width).toFixed(2)}
          y="1"
          width={Math.max(1, (viewportEnd - viewportStart) * width).toFixed(2)}
          height={height - 2}
          className="waveform-viewport"
        />
      ) : null}
      <rect x={(clampedStart * width).toFixed(2)} y="1" width={Math.max(1, (clampedEnd - clampedStart) * width).toFixed(2)} height={height - 2} className="waveform-selection" />
      <polyline points={points} className="waveform-line" />
      <line x1={(clampedStart * width).toFixed(2)} x2={(clampedStart * width).toFixed(2)} y1="2" y2={height - 2} className="waveform-handle" />
      <line x1={(clampedEnd * width).toFixed(2)} x2={(clampedEnd * width).toFixed(2)} y1="2" y2={height - 2} className="waveform-handle" />
      {playhead !== null ? <line x1={(playhead * width).toFixed(2)} x2={(playhead * width).toFixed(2)} y1="2" y2={height - 2} className="waveform-playhead" /> : null}
    </svg>
  );
}

function formatCompactDate(value) {
  if (!value) {
    return "-";
  }
  return new Intl.DateTimeFormat("de-DE", {
    year: "2-digit",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function readStoredAdminKey() {
  try {
    return localStorage.getItem(ADMIN_KEY_STORAGE) || sessionStorage.getItem(ADMIN_KEY_STORAGE) || "";
  } catch {
    return "";
  }
}

function writeStoredAdminKey(value) {
  try {
    localStorage.setItem(ADMIN_KEY_STORAGE, value);
    sessionStorage.setItem(ADMIN_KEY_STORAGE, value);
  } catch {}
}

function clearStoredAdminKey() {
  try {
    localStorage.removeItem(ADMIN_KEY_STORAGE);
    sessionStorage.removeItem(ADMIN_KEY_STORAGE);
  } catch {}
}

async function copyTextToClipboard(value) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
    return;
  }

  const textArea = document.createElement("textarea");
  textArea.value = value;
  textArea.setAttribute("readonly", "readonly");
  textArea.style.position = "fixed";
  textArea.style.opacity = "0";
  document.body.appendChild(textArea);
  textArea.select();
  textArea.setSelectionRange(0, textArea.value.length);
  document.execCommand("copy");
  document.body.removeChild(textArea);
}

function isUnauthorizedError(error) {
  return Number(error?.status) === 401;
}

function isVoiceConflictError(error) {
  return Number(error?.status) === 409 && error?.payload?.code === "voice_name_exists";
}

function Sparkline({ history, metricKey, color }) {
  const points = history.slice(-60).map((item) => Number(item?.[metricKey] || 0));
  const max = Math.max(1, ...points);
  const min = Math.min(0, ...points);
  const width = 100;
  const height = 40;
  const span = Math.max(1, max - min);
  const path = points
    .map((value, index) => {
      const x = points.length <= 1 ? 0 : (index / (points.length - 1)) * width;
      const y = height - ((value - min) / span) * height;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");

  return (
    <svg className="graph" viewBox="0 0 100 40" preserveAspectRatio="none">
      <path d={path} fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" />
    </svg>
  );
}

export default function AdminApp() {
  const [adminKey, setAdminKey] = useState(() => readStoredAdminKey());
  const [adminKeyInput, setAdminKeyInput] = useState(() => readStoredAdminKey());
  const [settingsSnapshot, setSettingsSnapshot] = useState(null);
  const [runtimeSnapshot, setRuntimeSnapshot] = useState(null);
  const [modelSnapshot, setModelSnapshot] = useState([]);
  const [voices, setVoices] = useState([]);
  const [languages, setLanguages] = useState({});
  const [keys, setKeys] = useState({ admin_key: null });
  const [generations, setGenerations] = useState([]);
  const [dashboard, setDashboard] = useState(null);
  const [voiceForm, setVoiceForm] = useState(initialVoiceForm);
  const [voicePreview, setVoicePreview] = useState(EMPTY_VOICE_PREVIEW);
  const [zoomFactor, setZoomFactor] = useState(1);
  const [viewportStartMs, setViewportStartMs] = useState(0);
  const [isLoopingPreview, setIsLoopingPreview] = useState(false);
  const [loopPlayheadMs, setLoopPlayheadMs] = useState(0);
  const [newlyCreatedKey, setNewlyCreatedKey] = useState(null);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const [voiceAudioUrls, setVoiceAudioUrls] = useState({});
  const dashboardAbortRef = useRef(null);
  const trimLoopAudioRef = useRef(null);

  const selectedTrim = useMemo(
    () => computeTrimSelection(parseSecondsToMs(voiceForm.trimStartSeconds), parseSecondsToMs(voiceForm.trimEndSeconds), voicePreview.durationMs || (MAX_REFERENCE_SECONDS * 1000)),
    [voiceForm.trimStartSeconds, voiceForm.trimEndSeconds, voicePreview.durationMs],
  );

  const viewportDurationMs = useMemo(() => {
    if (!voicePreview.durationMs) {
      return MIN_REFERENCE_SECONDS * 1000;
    }
    return Math.min(
      voicePreview.durationMs,
      Math.max(MIN_REFERENCE_SECONDS * 1000, Math.round(voicePreview.durationMs / zoomFactor)),
    );
  }, [voicePreview.durationMs, zoomFactor]);

  const maxViewportStartMs = Math.max(0, (voicePreview.durationMs || 0) - viewportDurationMs);
  const viewportEndMs = Math.min((voicePreview.durationMs || 0), viewportStartMs + viewportDurationMs);
  const zoomedWaveform = useMemo(
    () => sliceWaveformSamples(voicePreview.waveform, voicePreview.durationMs, viewportStartMs, viewportEndMs),
    [voicePreview.waveform, voicePreview.durationMs, viewportStartMs, viewportEndMs],
  );

  const settingsForm = useMemo(
    () => ({
      active_model: settingsSnapshot?.active_model || "HumeAI/tada-3b-ml",
      steps: settingsSnapshot?.steps || 10,
      sentence_chunking: settingsSnapshot?.sentence_chunking ?? true,
      short_sentence_merge_max_chars: settingsSnapshot?.short_sentence_merge_max_chars ?? 30,
      following_sentence_merge_min_chars: settingsSnapshot?.following_sentence_merge_min_chars ?? 20,
      allow_lan_access: settingsSnapshot?.allow_lan_access ?? false,
      stream_start_buffer_ms: settingsSnapshot?.stream_start_buffer_ms || 500,
      stream_chunk_ms: settingsSnapshot?.stream_chunk_ms || 500,
      batch_wait_ms: settingsSnapshot?.batch_wait_ms || 500,
      max_batch_size: settingsSnapshot?.max_batch_size || 8,
      max_parallel_requests: settingsSnapshot?.max_parallel_requests || 8,
      max_queue_size: settingsSnapshot?.max_queue_size || 256,
      model_storage_path: settingsSnapshot?.model_storage_path || "",
      whisper_base_url: settingsSnapshot?.whisper_base_url || "",
    }),
    [settingsSnapshot],
  );

  const [draftSettings, setDraftSettings] = useState(settingsForm);

  useEffect(() => {
    setDraftSettings(settingsForm);
  }, [settingsForm]);

  useEffect(() => {
    const stored = readStoredAdminKey();
    if (stored && stored !== adminKey) {
      setAdminKey(stored);
      setAdminKeyInput(stored);
    }
  }, []);

  useEffect(() => {
    if (!adminKey) {
      return undefined;
    }

    let cancelled = false;

    async function loadAudioUrls() {
      const nextEntries = await Promise.all(
        voices.map(async (voice) => {
          try {
            const url = await loadProtectedAudioUrl(voice.reference_url, { adminKey });
            return [voice.voice_id, url];
          } catch {
            return [voice.voice_id, ""];
          }
        }),
      );
      if (!cancelled) {
        setVoiceAudioUrls(Object.fromEntries(nextEntries.filter((entry) => entry[1])));
      }
    }

    if (voices.length > 0) {
      loadAudioUrls();
    }

    return () => {
      cancelled = true;
    };
  }, [adminKey, voices]);

  useEffect(() => () => {
    Object.values(voiceAudioUrls).forEach((url) => URL.revokeObjectURL(url));
  }, [voiceAudioUrls]);

  useEffect(() => {
    if (!voiceForm.file) {
      if (trimLoopAudioRef.current) {
        trimLoopAudioRef.current.pause();
      }
      setVoicePreview(EMPTY_VOICE_PREVIEW);
      setZoomFactor(1);
      setViewportStartMs(0);
      setIsLoopingPreview(false);
      setLoopPlayheadMs(0);
      return undefined;
    }

    let cancelled = false;
    const objectUrl = URL.createObjectURL(voiceForm.file);
    setVoicePreview({
      url: objectUrl,
      durationMs: 0,
      waveform: [],
      decodeError: "",
    });

    async function decodePreview() {
      let durationMs = 0;
      try {
        const audioElement = document.createElement("audio");
        audioElement.preload = "metadata";
        audioElement.src = objectUrl;
        durationMs = await new Promise((resolve) => {
          const finish = (value) => resolve(Number.isFinite(value) ? Math.round(value * 1000) : 0);
          audioElement.onloadedmetadata = () => finish(audioElement.duration);
          audioElement.onerror = () => finish(0);
        });

        let waveform = [];
        let decodeError = "";
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (AudioContextClass) {
          const context = new AudioContextClass();
          try {
            const buffer = await voiceForm.file.arrayBuffer();
            const decoded = await context.decodeAudioData(buffer.slice(0));
            waveform = buildWaveformSamples(decoded.getChannelData(0));
            if (!durationMs) {
              durationMs = Math.round(decoded.duration * 1000);
            }
          } catch {
            decodeError = "Waveform preview could not be decoded in the browser, but the file can still be imported.";
          } finally {
            if (typeof context.close === "function") {
              context.close().catch(() => {});
            }
          }
        }

        if (cancelled) {
          return;
        }

        const defaultEndMs = Math.min(durationMs || (DEFAULT_REFERENCE_SECONDS * 1000), DEFAULT_REFERENCE_SECONDS * 1000);
        setVoicePreview({
          url: objectUrl,
          durationMs,
          waveform,
          decodeError,
        });
        setZoomFactor(1);
        setViewportStartMs(0);
        setIsLoopingPreview(false);
        setLoopPlayheadMs(0);
        setVoiceForm((current) => {
          if (current.file !== voiceForm.file) {
            return current;
          }
          return {
            ...current,
            trimStartSeconds: current.trimStartSeconds || "0.00",
            trimEndSeconds: current.trimEndSeconds || formatTrimSeconds(defaultEndMs),
          };
        });
      } catch {
        if (!cancelled) {
          setVoicePreview({
            url: objectUrl,
            durationMs,
            waveform: [],
            decodeError: "Audio preview metadata could not be loaded.",
          });
        }
      }
    }

    decodePreview();

    return () => {
      cancelled = true;
      if (trimLoopAudioRef.current) {
        trimLoopAudioRef.current.pause();
      }
      URL.revokeObjectURL(objectUrl);
    };
  }, [voiceForm.file]);

  useEffect(() => {
    setViewportStartMs((current) => Math.min(Math.max(0, current), maxViewportStartMs));
  }, [maxViewportStartMs]);

  useEffect(() => {
    const audio = trimLoopAudioRef.current;
    if (!audio) {
      return undefined;
    }

    const handleTimeUpdate = () => {
      const nowMs = audio.currentTime * 1000;
      setLoopPlayheadMs(nowMs);
      if (isLoopingPreview && nowMs >= selectedTrim.endMs - 30) {
        audio.currentTime = selectedTrim.startMs / 1000;
        if (audio.paused) {
          audio.play().catch(() => {});
        }
      }
    };

    const handlePause = () => {
      setIsLoopingPreview(false);
    };

    audio.addEventListener("timeupdate", handleTimeUpdate);
    audio.addEventListener("pause", handlePause);

    return () => {
      audio.removeEventListener("timeupdate", handleTimeUpdate);
      audio.removeEventListener("pause", handlePause);
    };
  }, [isLoopingPreview, selectedTrim.startMs, selectedTrim.endMs]);

  useEffect(() => {
    if (!adminKey) {
      return undefined;
    }

    const controller = new AbortController();
    dashboardAbortRef.current = controller;

    async function connect() {
      try {
        const response = await fetch("/api/admin/dashboard/stream", {
          headers: { "X-Admin-Key": adminKey },
          signal: controller.signal,
        });
        if (!response.ok || !response.body) {
          throw new Error("Dashboard stream could not be opened.");
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
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
              setDashboard(JSON.parse(line));
            }
            newlineIndex = buffer.indexOf("\n");
          }
        }
      } catch (streamError) {
        if (!controller.signal.aborted) {
          if (isUnauthorizedError(streamError)) {
            clearPersistedAdminKey({
              nextMessage: "The stored admin key is no longer valid. Please enter the current admin key again.",
              keepInputValue: adminKey,
            });
            return;
          }
          setError(streamError.message || "Dashboard stream failed.");
        }
      }
    }

    connect();
    return () => {
      controller.abort();
      dashboardAbortRef.current = null;
    };
  }, [adminKey]);

  async function loadAll() {
    if (!adminKey) {
      return;
    }
    setError("");
    const [settingsData, voicesData, keysData, generationsData] = await Promise.all([
      apiFetch("/api/admin/settings", { adminKey }),
      apiFetch("/api/admin/voices", { adminKey }),
      apiFetch("/api/admin/keys", { adminKey }),
      apiFetch("/api/admin/generations", { adminKey }),
    ]);
    setSettingsSnapshot(settingsData.settings);
    setRuntimeSnapshot(settingsData.runtime);
    setModelSnapshot(settingsData.models || []);
    setVoices(voicesData.voices || []);
    setLanguages(voicesData.languages || {});
    setKeys(keysData);
    setGenerations(generationsData.generations || []);
  }

  useEffect(() => {
    if (!adminKey) {
      return;
    }
    loadAll().catch((loadError) => {
      if (isUnauthorizedError(loadError)) {
        clearPersistedAdminKey({
          nextMessage: "The stored admin key is no longer valid. Please enter the current admin key again.",
          keepInputValue: adminKey,
        });
        return;
      }
      setError(loadError.message);
    });
  }, [adminKey]);

  function persistAdminKey(nextKey) {
    writeStoredAdminKey(nextKey);
    setAdminKey(nextKey);
    setAdminKeyInput(nextKey);
  }

  function clearPersistedAdminKey({ nextMessage = "", keepInputValue = "" } = {}) {
    clearStoredAdminKey();
    setAdminKey("");
    setAdminKeyInput(keepInputValue);
    setSettingsSnapshot(null);
    setRuntimeSnapshot(null);
    setModelSnapshot([]);
    setVoices([]);
    setLanguages({});
    setKeys({ admin_key: null });
    setGenerations([]);
    setDashboard(null);
    setVoicePreview(EMPTY_VOICE_PREVIEW);
    if (nextMessage) {
      setError(nextMessage);
    }
  }

  async function handleOpenAdmin() {
    const candidate = adminKeyInput.trim();
    if (!candidate) {
      return;
    }
    setError("");
    setMessage("");
    try {
      await apiFetch("/api/admin/settings", { adminKey: candidate });
      persistAdminKey(candidate);
    } catch (loginError) {
      clearStoredAdminKey();
      setAdminKey("");
      setError(loginError.message || "Admin login failed.");
    }
  }

  async function handleSettingsSubmit(event) {
    event.preventDefault();
    setSaving(true);
    setMessage("");
    setError("");
    try {
      const data = await apiFetch("/api/admin/settings", {
        method: "PUT",
        adminKey,
        body: draftSettings,
      });
      setSettingsSnapshot(data.settings);
      setRuntimeSnapshot(data.runtime);
      setModelSnapshot(data.models || []);
      setMessage(data.settings.restart_required ? "Settings saved. Restart required for startup-level changes." : "Settings saved.");
    } catch (submitError) {
      if (isUnauthorizedError(submitError)) {
        clearPersistedAdminKey({
          nextMessage: "Your admin key is no longer valid. Please sign in again.",
          keepInputValue: adminKey,
        });
        return;
      }
      setError(submitError.message);
    } finally {
      setSaving(false);
    }
  }

  function applyTrimPreset(seconds) {
    const endMs = Math.min(voicePreview.durationMs || (seconds * 1000), seconds * 1000, MAX_REFERENCE_SECONDS * 1000);
    setVoiceForm((current) => ({
      ...current,
      trimStartSeconds: "0.00",
      trimEndSeconds: formatTrimSeconds(endMs),
    }));
  }

  function handleTrimFieldChange(field, value) {
    setVoiceForm((current) => ({ ...current, [field]: value }));
  }

  function syncTrimFieldsToSelection() {
    setVoiceForm((current) => ({
      ...current,
      trimStartSeconds: formatTrimSeconds(selectedTrim.startMs),
      trimEndSeconds: formatTrimSeconds(selectedTrim.endMs),
    }));
  }

  function handleLoopPreviewPlay() {
    const audio = trimLoopAudioRef.current;
    if (!audio || !voicePreview.url || !selectedTrim.isValid) {
      return;
    }
    audio.currentTime = selectedTrim.startMs / 1000;
    setLoopPlayheadMs(selectedTrim.startMs);
    audio.play().then(() => {
      setIsLoopingPreview(true);
    }).catch(() => {});
  }

  function handleLoopPreviewPause() {
    const audio = trimLoopAudioRef.current;
    if (!audio) {
      return;
    }
    audio.pause();
    setIsLoopingPreview(false);
  }

  async function handleVoiceCreate(event) {
    event.preventDefault();
    if (!voiceForm.file) {
      setError("Please pick a reference audio file.");
      return;
    }
    async function submitVoiceCreate(overwriteExisting) {
      const formData = new FormData();
      formData.append("name", voiceForm.name || "Meine Stimme");
      formData.append("language", voiceForm.language);
      formData.append("transcript", voiceForm.transcript);
      formData.append("trim_start_ms", String(selectedTrim.startMs));
      formData.append("trim_end_ms", String(selectedTrim.endMs));
      formData.append("overwrite_existing", overwriteExisting ? "true" : "false");
      formData.append("audio", voiceForm.file);
      return apiFetch("/api/admin/voices", { method: "POST", adminKey, body: formData });
    }

    setSaving(true);
    setError("");
    setMessage("");
    try {
      let data;
      let didOverwrite = false;
      try {
        data = await submitVoiceCreate(false);
      } catch (submitError) {
        if (isUnauthorizedError(submitError)) {
          clearPersistedAdminKey({
            nextMessage: "Your admin key is no longer valid. Please sign in again.",
            keepInputValue: adminKey,
          });
          return;
        }
        if (isVoiceConflictError(submitError)) {
          const existingVoiceName = submitError?.payload?.existing_voice_name || (voiceForm.name || "this voice");
          const confirmed = window.confirm(`A voice named "${existingVoiceName}" already exists. Do you want to overwrite it?`);
          if (!confirmed) {
            setError(`Voice "${existingVoiceName}" already exists. Save cancelled.`);
            return;
          }
          didOverwrite = true;
          data = await submitVoiceCreate(true);
        } else {
          throw submitError;
        }
      }

      setVoiceForm(initialVoiceForm);
      setVoicePreview(EMPTY_VOICE_PREVIEW);
      await loadAll();
      const savedVoice = data.voice || {};
      const tailSilenceText = savedVoice.tail_silence_ms
        ? ` incl. ${formatSeconds(savedVoice.tail_silence_ms / 1000)} safety silence`
        : "";
      const trimText = savedVoice.duration_seconds
        ? `${didOverwrite ? "Voice overwritten" : "Saved trimmed reference"} (${formatSeconds(savedVoice.duration_seconds)}${tailSilenceText}).`
        : didOverwrite ? "Voice overwritten." : "Voice saved.";
      setMessage(trimText);
    } catch (submitError) {
      if (isUnauthorizedError(submitError)) {
        clearPersistedAdminKey({
          nextMessage: "Your admin key is no longer valid. Please sign in again.",
          keepInputValue: adminKey,
        });
        return;
      }
      setError(submitError.message);
    } finally {
      setSaving(false);
    }
  }

  async function handleTranscribe() {
    if (!voiceForm.file) {
      setError("Upload the voice sample first.");
      return;
    }
    setSaving(true);
    setError("");
    const formData = new FormData();
    formData.append("trim_start_ms", String(selectedTrim.startMs));
    formData.append("trim_end_ms", String(selectedTrim.endMs));
    formData.append("audio", voiceForm.file);
    try {
      const data = await apiFetch("/api/admin/voices/transcribe", {
        method: "POST",
        adminKey,
        body: formData,
      });
      setVoiceForm((current) => ({ ...current, transcript: data.text }));
      setMessage(
        data.was_auto_trimmed
          ? `Transcript loaded from Whisper. The spoken reference was clamped to ${formatReferenceLimit(MAX_REFERENCE_SECONDS)} s.`
          : "Transcript loaded from Whisper.",
      );
    } catch (submitError) {
      if (isUnauthorizedError(submitError)) {
        clearPersistedAdminKey({
          nextMessage: "Your admin key is no longer valid. Please sign in again.",
          keepInputValue: adminKey,
        });
        return;
      }
      setError(submitError.message);
    } finally {
      setSaving(false);
    }
  }

  async function handleModelDownload(modelId) {
    setError("");
    try {
      const data = await apiFetch("/api/admin/models/download", {
        method: "POST",
        adminKey,
        body: { model_id: modelId },
      });
      setModelSnapshot(data.models || []);
      setMessage(`Download job queued for ${modelId}.`);
    } catch (downloadError) {
      if (isUnauthorizedError(downloadError)) {
        clearPersistedAdminKey({
          nextMessage: "Your admin key is no longer valid. Please sign in again.",
          keepInputValue: adminKey,
        });
        return;
      }
      setError(downloadError.message);
    }
  }

  async function handleRotateAdminKey() {
    setError("");
    try {
      const data = await apiFetch("/api/admin/keys", {
        method: "POST",
        adminKey,
      });
      setKeys(data.keys);
      setNewlyCreatedKey(data.key);
      if (data.key?.token) {
        persistAdminKey(data.key.token);
      }
      setMessage("Admin key rotated.");
    } catch (keyError) {
      if (isUnauthorizedError(keyError)) {
        clearPersistedAdminKey({
          nextMessage: "Your admin key is no longer valid. Please sign in again.",
          keepInputValue: adminKey,
        });
        return;
      }
      setError(keyError.message);
    }
  }

  async function handleCopyKey(token, label) {
    if (!token) {
      setError(`The key "${label}" is not available in this browser anymore.`);
      setMessage("");
      return;
    }
    try {
      await copyTextToClipboard(token);
      setError("");
      setMessage(`Key "${label}" copied to clipboard.`);
    } catch {
      setMessage("");
      setError(`The key "${label}" could not be copied automatically.`);
    }
  }

  async function handleDeleteVoice(voiceId) {
    setSaving(true);
    setError("");
    try {
      const data = await apiFetch(`/api/admin/voices/${voiceId}`, {
        method: "DELETE",
        adminKey,
      });
      setVoices(data.voices || []);
      setLanguages(data.languages || {});
      setVoiceAudioUrls((current) => {
        const next = { ...current };
        if (next[voiceId]) {
          URL.revokeObjectURL(next[voiceId]);
        }
        delete next[voiceId];
        return next;
      });
      setMessage("Voice deleted.");
    } catch (deleteError) {
      if (isUnauthorizedError(deleteError)) {
        clearPersistedAdminKey({
          nextMessage: "Your admin key is no longer valid. Please sign in again.",
          keepInputValue: adminKey,
        });
        return;
      }
      setError(deleteError.message);
    } finally {
      setSaving(false);
    }
  }

  if (!adminKey) {
    return (
      <main className="auth-shell">
        <section className="panel stack">
          <p className="eyebrow">Private Access</p>
          <h1>TADA Admin</h1>
          <p className="muted">The public generation API stays open. The dashboard is protected by the master admin key and the temporary startup key shown during launch.</p>
          <input value={adminKeyInput} onChange={(event) => setAdminKeyInput(event.target.value)} placeholder="tada_admin_..." />
          <div className="button-row">
            <button type="button" onClick={() => handleOpenAdmin()} disabled={!adminKeyInput.trim()}>
              Open Admin
            </button>
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className="page-shell">
      <section className="hero">
        <div>
          <p className="eyebrow">Powered by SONS</p>
          <h1>G3 TADA Admin</h1>
          <p className="hero-copy">
            Manage runtime settings, model downloads, voices, the admin-key workflow and the live scheduler dashboard.
          </p>
          <div className="button-row" style={{ marginTop: 16 }}>
            <button type="button" className="secondary" onClick={() => loadAll().catch((loadError) => setError(loadError.message))}>
              Refresh
            </button>
            <button type="button" className="secondary" onClick={() => handleRotateAdminKey()}>
              Rotate Admin Key
            </button>
            <button
              type="button"
              className="ghost"
              onClick={() => {
                clearPersistedAdminKey();
              }}
            >
              Logout
            </button>
          </div>
        </div>

        <div className="status-grid">
          <div className="status-pill"><span>Configured Model</span><strong>{settingsSnapshot?.active_model || "-"}</strong></div>
          <div className="status-pill"><span>Loaded Model</span><strong>{runtimeSnapshot?.loaded_model_name || "-"}</strong></div>
          <div className="status-pill"><span>GPU</span><strong>{runtimeSnapshot?.gpu_name || "CPU"}</strong></div>
          <div className="status-pill"><span>Voices</span><strong>{runtimeSnapshot?.voices_count ?? 0}</strong></div>
        </div>
      </section>

      {message ? <div className="message success">{message}</div> : null}
      {error ? <div className="message error">{error}</div> : null}

      <section className="panel-grid">
        <section className="panel stack">
          <p className="eyebrow">Dashboard</p>
          <h2>Live Queue</h2>
          <div className="metric-grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))" }}>
            <div className="metric-card"><span>Queue</span><strong>{dashboard?.queue_length ?? 0}</strong></div>
            <div className="metric-card"><span>Active</span><strong>{dashboard?.active_request_count ?? 0}</strong></div>
            <div className="metric-card"><span>Mean TTFT</span><strong>{formatMs(dashboard?.mean_ttft_ms)}</strong></div>
            <div className="metric-card"><span>Audio/s</span><strong>{dashboard?.throughput_audio_sps ?? 0}</strong></div>
          </div>
          <Sparkline history={dashboard?.history || []} metricKey="throughput_audio_sps" color="#0e7af6" />
          <div className="card">
            <h3>Current Batch</h3>
            {dashboard?.current_batch ? (
              <div className="list">
                <div className="muted mono">#{dashboard.current_batch.batch_id} | {dashboard.current_batch.size} items</div>
                {dashboard.current_batch.items.map((item) => (
                  <div key={`${item.request_id}-${item.sentence_index}`} className="card">
                    <strong>{item.request_id}</strong>
                    <div className="muted">{item.voice_id} | sentence {item.sentence_index + 1}</div>
                    <div>{item.text_preview}</div>
                  </div>
                ))}
              </div>
            ) : <p className="muted">No batch is currently running.</p>}
          </div>
        </section>

        <section className="panel stack">
          <p className="eyebrow">Settings</p>
          <h2>Runtime Config</h2>
          <form className="stack" onSubmit={handleSettingsSubmit}>
            <div className="two-col">
              <label>Model<select value={draftSettings.active_model} onChange={(event) => setDraftSettings((current) => ({ ...current, active_model: event.target.value }))}>{runtimeSnapshot?.available_models?.map((model) => <option key={model.id} value={model.id}>{model.label}</option>)}</select></label>
              <label>Steps<input type="number" min="1" max="128" value={draftSettings.steps} onChange={(event) => setDraftSettings((current) => ({ ...current, steps: Number(event.target.value) }))} /></label>
              <label>Batch Wait (ms)<input type="number" min="0" max="5000" value={draftSettings.batch_wait_ms} onChange={(event) => setDraftSettings((current) => ({ ...current, batch_wait_ms: Number(event.target.value) }))} /></label>
              <label>Chunk Size (ms)<input type="number" min="50" max="5000" value={draftSettings.stream_chunk_ms} onChange={(event) => setDraftSettings((current) => ({ ...current, stream_chunk_ms: Number(event.target.value) }))} /></label>
              <label>Start Buffer (ms)<input type="number" min="0" max="5000" value={draftSettings.stream_start_buffer_ms} onChange={(event) => setDraftSettings((current) => ({ ...current, stream_start_buffer_ms: Number(event.target.value) }))} /></label>
              <label>Max Batch Size<input type="number" min="1" max="128" value={draftSettings.max_batch_size} onChange={(event) => setDraftSettings((current) => ({ ...current, max_batch_size: Number(event.target.value) }))} /></label>
              <label>Max Parallel Requests<input type="number" min="1" max="128" value={draftSettings.max_parallel_requests} onChange={(event) => setDraftSettings((current) => ({ ...current, max_parallel_requests: Number(event.target.value) }))} /></label>
              <label>Queue Limit<input type="number" min="1" max="2048" value={draftSettings.max_queue_size} onChange={(event) => setDraftSettings((current) => ({ ...current, max_queue_size: Number(event.target.value) }))} /></label>
              <label>Short Merge Max Chars<input type="number" min="0" max="200" value={draftSettings.short_sentence_merge_max_chars} onChange={(event) => setDraftSettings((current) => ({ ...current, short_sentence_merge_max_chars: Number(event.target.value) }))} /></label>
              <label>Next Sentence Min Chars<input type="number" min="0" max="500" value={draftSettings.following_sentence_merge_min_chars} onChange={(event) => setDraftSettings((current) => ({ ...current, following_sentence_merge_min_chars: Number(event.target.value) }))} /></label>
            </div>
            <label>Sentence Chunking<select value={String(draftSettings.sentence_chunking)} onChange={(event) => setDraftSettings((current) => ({ ...current, sentence_chunking: event.target.value === "true" }))}><option value="true">Enabled</option><option value="false">Disabled</option></select></label>
            <label>LAN Access<select value={String(draftSettings.allow_lan_access)} onChange={(event) => setDraftSettings((current) => ({ ...current, allow_lan_access: event.target.value === "true" }))}><option value="false">Local only (127.0.0.1)</option><option value="true">Enable LAN (0.0.0.0)</option></select></label>
            <label>Model Storage Path<input value={draftSettings.model_storage_path} onChange={(event) => setDraftSettings((current) => ({ ...current, model_storage_path: event.target.value }))} /></label>
            <label>Whisper Base URL<input value={draftSettings.whisper_base_url} onChange={(event) => setDraftSettings((current) => ({ ...current, whisper_base_url: event.target.value }))} placeholder="http://127.0.0.1:8001/v1" /></label>
            <label>Hugging Face Token<input type="password" placeholder={settingsSnapshot?.hf_token_present ? "Stored" : "Not configured"} onChange={(event) => setDraftSettings((current) => ({ ...current, hf_token: event.target.value }))} /></label>
            <label>Whisper API Key<input type="password" placeholder={settingsSnapshot?.whisper_api_key_present ? "Stored" : "Optional"} onChange={(event) => setDraftSettings((current) => ({ ...current, whisper_api_key: event.target.value }))} /></label>
            <button type="submit" disabled={saving}>Save Settings</button>
          </form>
        </section>
      </section>

      <section className="panel-grid">
        <section className="panel stack full-span">
          <p className="eyebrow">Models</p>
          <h2>Downloads</h2>
          <div className="table-wrap">
            <table className="table downloads-table">
              <thead><tr><th>Model</th><th className="status-cell">Status</th><th>Path</th><th className="action-cell">Action</th></tr></thead>
              <tbody>
                {modelSnapshot.map((model) => (
                  <tr key={model.id}>
                    <td>{model.label}</td>
                    <td className="status-cell">{model.status}</td>
                    <td className="mono path-cell">{model.local_path || model.storage_root}</td>
                    <td className="action-cell"><button type="button" className="secondary" onClick={() => handleModelDownload(model.id)} disabled={model.status === "downloading"}>{model.status === "downloading" ? "Downloading..." : "Download"}</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="panel stack full-span">
          <p className="eyebrow">Admin Key</p>
          <h2>Dashboard Access</h2>
          {newlyCreatedKey ? (
            <div className="card stack compact">
              <div className="card-header">
                <div className="stack compact">
                  <strong>{newlyCreatedKey.label}</strong>
                  <span className="muted">Admin Key</span>
                </div>
                <button type="button" className="secondary" onClick={() => handleCopyKey(newlyCreatedKey.token, newlyCreatedKey.label)}>
                  Copy
                </button>
              </div>
              <div className="mono key-token-value">{newlyCreatedKey.token}</div>
              <div className="muted">The server returns this token only once and the browser switches to it immediately.</div>
            </div>
          ) : null}
          <div className="metric-grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))" }}>
            <div className="metric-card"><span>Name</span><strong>{keys.admin_key?.label || "Master Admin Key"}</strong></div>
            <div className="metric-card"><span>Created</span><strong>{formatCompactDate(keys.admin_key?.created_at)}</strong></div>
            <div className="metric-card"><span>Last Used</span><strong>{formatCompactDate(keys.admin_key?.last_used_at)}</strong></div>
            <div className="metric-card"><span>Browser Token</span><strong>{adminKey ? "stored" : "missing"}</strong></div>
          </div>
          <div className="button-row">
            <button type="button" onClick={() => handleRotateAdminKey()}>Rotate Admin Key</button>
            <button type="button" className="secondary" onClick={() => handleCopyKey(adminKey, keys.admin_key?.label || "Master Admin Key")}>Copy Current Key</button>
          </div>
          <div className="table-wrap">
            <table className="table keys-table">
              <thead><tr><th>Type</th><th>Name</th><th>Token</th><th className="date-cell">Created</th><th className="date-cell">Last Used</th><th className="action-cell">Action</th></tr></thead>
              <tbody>
                {keys.admin_key ? (
                  <tr>
                    <td>Admin</td>
                    <td>{keys.admin_key.label}</td>
                    <td className="key-token-cell">
                      <div className="mono key-token-value">{adminKey}</div>
                    </td>
                    <td className="date-cell">{formatCompactDate(keys.admin_key.created_at)}</td>
                    <td className="date-cell">{formatCompactDate(keys.admin_key.last_used_at)}</td>
                    <td className="action-cell">
                      <div className="button-row compact">
                        <button type="button" className="secondary" onClick={() => handleCopyKey(adminKey, keys.admin_key.label)}>
                          Copy
                        </button>
                      </div>
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </section>
      </section>

      <section className="panel-grid">
        <section className="panel stack full-span">
          <p className="eyebrow">History</p>
          <h2>Latest Generations</h2>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Created</th>
                  <th>Voice</th>
                  <th>Text</th>
                  <th>Audio</th>
                  <th>TTFT</th>
                  <th>Total</th>
                  <th>Batches</th>
                  <th>Output</th>
                </tr>
              </thead>
              <tbody>
                {generations.length > 0 ? (
                  generations.map((item) => (
                    <tr key={item.generation_id}>
                      <td className="date-cell">{formatCompactDate(item.created_at)}</td>
                      <td>{item.voice_name || item.voice_id || "-"}</td>
                      <td>{item.text || "-"}</td>
                      <td>{formatSeconds(item.duration_seconds)}</td>
                      <td>{formatMs(item.ttft_ms)}</td>
                      <td>{formatMs(item.total_wall_ms)}</td>
                      <td>{item.batch_count ?? "-"}</td>
                      <td className="action-cell">
                        {item.audio_url ? (
                          <button type="button" className="secondary" onClick={() => window.open(item.audio_url, "_blank", "noopener,noreferrer")}>
                            Open WAV
                          </button>
                        ) : "-"}
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={8}>No generation history has been recorded yet.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>

        <section className="panel stack">
          <p className="eyebrow">Voices</p>
          <h2>Create Voice</h2>
          <form className="stack" onSubmit={handleVoiceCreate}>
            <div className="two-col">
              <label>Name<input value={voiceForm.name} onChange={(event) => setVoiceForm((current) => ({ ...current, name: event.target.value }))} /></label>
              <label>Language<select value={voiceForm.language} onChange={(event) => setVoiceForm((current) => ({ ...current, language: event.target.value }))}>{Object.entries(languages).map(([code, label]) => <option key={code} value={code}>{label}</option>)}</select></label>
            </div>
            <label>Transcript<textarea value={voiceForm.transcript} onChange={(event) => setVoiceForm((current) => ({ ...current, transcript: event.target.value }))} /></label>
            <input
              type="file"
              accept="audio/*,.wav,.mp3,.flac,.m4a"
              onChange={(event) => setVoiceForm((current) => ({
                ...current,
                file: event.target.files?.[0] || null,
                trimStartSeconds: "0.00",
                trimEndSeconds: "",
              }))}
            />
            {voiceForm.file ? (
              <div className="voice-trim-card">
                <div className="card-header">
                  <div className="stack compact">
                    <strong>{voiceForm.file.name}</strong>
                    <span className="muted">
                      Source: {voicePreview.durationMs ? formatSeconds(voicePreview.durationMs / 1000) : "loading..."} | Selected: {formatSeconds(selectedTrim.durationMs / 1000)}
                    </span>
                  </div>
                  <span className={`trim-badge ${!selectedTrim.isValid ? "danger" : ""}`}>
                    {formatReferenceLimit(MIN_REFERENCE_SECONDS)}-{formatReferenceLimit(MAX_REFERENCE_SECONDS)} s speech
                  </span>
                </div>
                {voicePreview.url ? <audio ref={trimLoopAudioRef} src={voicePreview.url} preload="metadata" className="trim-loop-audio" /> : null}
                <div className="stack compact">
                  <span className="muted">Overview</span>
                  <WaveformPreview
                    samples={voicePreview.waveform}
                    startRatio={voicePreview.durationMs ? selectedTrim.startMs / voicePreview.durationMs : 0}
                    endRatio={voicePreview.durationMs ? selectedTrim.endMs / voicePreview.durationMs : 1}
                    viewportStartRatio={voicePreview.durationMs ? viewportStartMs / voicePreview.durationMs : 0}
                    viewportEndRatio={voicePreview.durationMs ? viewportEndMs / voicePreview.durationMs : 1}
                    playheadRatio={voicePreview.durationMs ? loopPlayheadMs / voicePreview.durationMs : null}
                  />
                </div>
                <div className="trim-grid">
                  <label>
                    Zoom
                    <input
                      type="range"
                      min="1"
                      max="12"
                      step="0.25"
                      value={zoomFactor}
                      onChange={(event) => setZoomFactor(Number(event.target.value))}
                    />
                    <span className="muted">{zoomFactor.toFixed(2)}x</span>
                  </label>
                  <label>
                    Global View
                    <input
                      type="range"
                      min="0"
                      max={Math.max(0, maxViewportStartMs)}
                      step="10"
                      value={Math.min(viewportStartMs, maxViewportStartMs)}
                      onChange={(event) => setViewportStartMs(Number(event.target.value))}
                      disabled={maxViewportStartMs <= 0}
                    />
                    <span className="muted">
                      {formatSeconds(viewportStartMs / 1000)} - {formatSeconds(viewportEndMs / 1000)}
                    </span>
                  </label>
                </div>
                <div className="stack compact">
                  <span className="muted">Zoomed Trim View</span>
                  <WaveformPreview
                    samples={zoomedWaveform}
                    startRatio={viewportDurationMs ? Math.max(0, (selectedTrim.startMs - viewportStartMs) / viewportDurationMs) : 0}
                    endRatio={viewportDurationMs ? Math.min(1, (selectedTrim.endMs - viewportStartMs) / viewportDurationMs) : 1}
                    playheadRatio={viewportDurationMs ? Math.max(0, Math.min(1, (loopPlayheadMs - viewportStartMs) / viewportDurationMs)) : null}
                  />
                </div>
                <div className="trim-grid">
                  <label>
                    In Point (s)
                    <input
                      type="number"
                      min="0"
                      max={voicePreview.durationMs ? (voicePreview.durationMs / 1000).toFixed(2) : undefined}
                      step="0.01"
                      value={voiceForm.trimStartSeconds}
                      onChange={(event) => handleTrimFieldChange("trimStartSeconds", event.target.value)}
                      onBlur={syncTrimFieldsToSelection}
                    />
                  </label>
                  <label>
                    Out Point (s)
                    <input
                      type="number"
                      min="0"
                      max={voicePreview.durationMs ? (voicePreview.durationMs / 1000).toFixed(2) : undefined}
                      step="0.01"
                      value={voiceForm.trimEndSeconds}
                      onChange={(event) => handleTrimFieldChange("trimEndSeconds", event.target.value)}
                      onBlur={syncTrimFieldsToSelection}
                    />
                  </label>
                </div>
                <div className="trim-stats">
                  <span className={`trim-stat ${selectedTrim.isValid ? "" : "danger"}`}>
                    Selection Length: {formatSeconds(selectedTrim.durationMs / 1000)}
                  </span>
                  <span className="trim-stat">Loop Position: {formatSeconds(loopPlayheadMs / 1000)}</span>
                </div>
                {voicePreview.durationMs ? (
                  <div className="trim-sliders">
                    <label>
                      In Point
                      <input
                        type="range"
                        min="0"
                        max={Math.max(0, voicePreview.durationMs - (MIN_REFERENCE_SECONDS * 1000))}
                        step="10"
                        value={selectedTrim.startMs}
                        onChange={(event) => handleTrimFieldChange("trimStartSeconds", formatTrimSeconds(event.target.value))}
                      />
                    </label>
                    <label>
                      Out Point
                      <input
                        type="range"
                        min={Math.min(MIN_REFERENCE_SECONDS * 1000, voicePreview.durationMs)}
                        max={voicePreview.durationMs}
                        step="10"
                        value={selectedTrim.endMs}
                        onChange={(event) => handleTrimFieldChange("trimEndSeconds", formatTrimSeconds(event.target.value))}
                      />
                    </label>
                  </div>
                ) : null}
                <div className="button-row compact">
                  <button type="button" className="secondary" onClick={handleLoopPreviewPlay} disabled={!voiceForm.file || !selectedTrim.isValid}>Play Loop</button>
                  <button type="button" className="ghost" onClick={handleLoopPreviewPause} disabled={!voiceForm.file}>Pause</button>
                  <button type="button" className="ghost" onClick={() => applyTrimPreset(MIN_REFERENCE_SECONDS)} disabled={!voiceForm.file}>Use 3 s</button>
                  <button type="button" className="ghost" onClick={() => applyTrimPreset(8)} disabled={!voiceForm.file}>Use 8 s</button>
                  <button type="button" className="ghost" onClick={() => applyTrimPreset(10)} disabled={!voiceForm.file}>Use 10 s</button>
                  <button type="button" className="ghost" onClick={() => applyTrimPreset(MAX_REFERENCE_SECONDS)} disabled={!voiceForm.file}>Use {formatReferenceLimit(MAX_REFERENCE_SECONDS)} s</button>
                </div>
                <div className="muted">
                  TADA works best with short prompts. Pick a clean spoken segment between {formatReferenceLimit(MIN_REFERENCE_SECONDS)} and {formatReferenceLimit(MAX_REFERENCE_SECONDS)} seconds. When you save it, the importer adds {formatReferenceLimit(REFERENCE_TAIL_SILENCE_SECONDS)} seconds of tail silence automatically.
                </div>
                {!selectedTrim.isValid ? (
                  <div className="message error">
                    {selectedTrim.sourceTooShort
                      ? `The uploaded file is too short. Please use a reference that is at least ${formatReferenceLimit(MIN_REFERENCE_SECONDS)} seconds long.`
                      : `Choose a spoken segment between ${formatReferenceLimit(MIN_REFERENCE_SECONDS)} and ${formatReferenceLimit(MAX_REFERENCE_SECONDS)} seconds.`}
                  </div>
                ) : null}
                {voicePreview.decodeError ? <div className="muted">{voicePreview.decodeError}</div> : null}
              </div>
            ) : null}
            <div className="button-row">
              <button type="button" className="secondary" onClick={handleTranscribe} disabled={saving || !selectedTrim.isValid}>Transcribe with Whisper</button>
              <button type="submit" disabled={saving || !selectedTrim.isValid}>Save Voice</button>
            </div>
          </form>
        </section>

        <section className="panel stack">
          <p className="eyebrow">Voice Library</p>
          <h2>Saved Voices</h2>
          <div className="list">
            {voices.length === 0 ? <div className="card muted">No voices saved yet.</div> : null}
            {voices.map((voice) => (
              <div className="card" key={voice.voice_id}>
                <strong>{voice.name}</strong>
                <div className="muted">
                  {languages[voice.language] || voice.language} | {formatSeconds(voice.duration_seconds)} | {formatDate(voice.created_at)}
                </div>
                <div className="muted">
                  Trim: {formatSeconds((voice.trim_start_ms || 0) / 1000)} - {formatSeconds((voice.trim_end_ms || Math.round((voice.duration_seconds || 0) * 1000)) / 1000)}
                  {voice.source_duration_seconds ? ` from ${formatSeconds(voice.source_duration_seconds)}` : ""}
                  {voice.was_auto_trimmed ? " | auto-clamped" : ""}
                  {voice.tail_silence_ms ? ` | +${formatSeconds(voice.tail_silence_ms / 1000)} safety silence` : ""}
                </div>
                <div>{voice.transcript || <span className="muted">No transcript stored.</span>}</div>
                {voiceAudioUrls[voice.voice_id] ? <audio controls src={voiceAudioUrls[voice.voice_id]} /> : null}
                <div className="button-row">
                  <button type="button" className="ghost danger" onClick={() => handleDeleteVoice(voice.voice_id)} disabled={saving}>Delete Voice</button>
                </div>
              </div>
            ))}
          </div>
        </section>
      </section>
    </main>
  );
}
