export class ApiError extends Error {
  constructor(message, status, payload) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

export async function apiFetch(url, options = {}) {
  const {
    adminKey,
    headers = {},
    body,
    responseType = "json",
    ...rest
  } = options;

  const nextHeaders = { ...headers };
  if (adminKey) {
    nextHeaders["X-Admin-Key"] = adminKey;
  }

  const requestOptions = {
    ...rest,
    headers: nextHeaders,
  };

  if (body instanceof FormData || typeof body === "string" || body === undefined) {
    requestOptions.body = body;
  } else {
    requestOptions.body = JSON.stringify(body);
    requestOptions.headers["Content-Type"] = "application/json";
  }

  const response = await fetch(url, requestOptions);
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const payload = typeof data?.detail === "object" && data?.detail !== null ? { ...data, ...data.detail } : data;
    const message = typeof data?.detail === "string" ? data.detail : payload.message || data.message || `Request failed with ${response.status}`;
    throw new ApiError(message, response.status, payload);
  }

  if (responseType === "blob") {
    return response.blob();
  }
  if (responseType === "text") {
    return response.text();
  }
  return response.json().catch(() => ({}));
}

export async function loadProtectedAudioUrl(url, { adminKey } = {}) {
  const blob = await apiFetch(url, { adminKey, responseType: "blob" });
  return URL.createObjectURL(blob);
}

export async function streamNdjson(url, options) {
  const { adminKey, body, signal, onEvent } = options;
  const headers = {};
  if (adminKey) {
    headers["X-Admin-Key"] = adminKey;
  }
  if (!(body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  const response = await fetch(url, {
    method: "POST",
    headers,
    body: body instanceof FormData ? body : JSON.stringify(body || {}),
    signal,
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const payload = typeof data?.detail === "object" && data?.detail !== null ? { ...data, ...data.detail } : data;
    const message = typeof data?.detail === "string" ? data.detail : payload.message || data.message || "Streaming request failed";
    throw new ApiError(message, response.status, payload);
  }
  if (!response.body) {
    throw new Error("Streaming response does not include a readable body.");
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
        await onEvent(JSON.parse(line));
      }
      newlineIndex = buffer.indexOf("\n");
    }
  }

  const tail = buffer.trim();
  if (tail) {
    await onEvent(JSON.parse(tail));
  }
}

export function formatDate(value) {
  if (!value) {
    return "-";
  }
  return new Date(value).toLocaleString("de-DE");
}

export function formatMs(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return `${Number(value).toFixed(0)} ms`;
}

export function formatSeconds(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return `${Number(value).toFixed(2)} s`;
}
