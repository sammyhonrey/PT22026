/**
 * Thin HTTP client for the VoiceForge backend.
 *
 * Every network call lives here so the UI code never deals with fetch(),
 * status codes or JSON parsing directly.
 */

export class ApiError extends Error {
  constructor(message, status = 0, code = "request_failed") {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

async function readJson(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

async function requestJson(url, options = {}) {
  let response;

  try {
    response = await fetch(url, options);
  } catch (error) {
    if (error.name === "AbortError") {
      throw error;
    }

    throw new ApiError(
      "Cannot reach the VoiceForge server. Is it still running?",
      0,
      "network_error",
    );
  }

  const payload = await readJson(response);

  if (!response.ok || !payload || payload.ok !== true) {
    const message =
      (payload && payload.error) ||
      `The server returned an error (HTTP ${response.status}).`;

    throw new ApiError(message, response.status, payload?.code || "request_failed");
  }

  return payload;
}

/**
 * Engine capabilities, voices and limits.
 * This is the single source of truth for the Studio controls.
 */
export async function fetchEngineInfo() {
  return requestJson("/api/engines");
}

/**
 * Generate speech.
 *
 * @param {{text: string, engine: string, voice?: string|null,
 *          speed?: number|null, signal?: AbortSignal}} request
 */
export async function generateSpeech({
  text,
  engine,
  voice,
  speed,
  pronunciation,
  signal,
}) {
  const body = { text, engine };

  if (pronunciation) {
    body.pronunciation = true;
  }

  if (voice) {
    body.voice = voice;
  }

  if (typeof speed === "number") {
    body.speed = speed;
  }

  return requestJson("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
}
