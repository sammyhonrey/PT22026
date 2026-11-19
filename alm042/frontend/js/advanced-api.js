/**
 * HTTP client for documents, pronunciation rules, presets and MP3 export.
 */

import { ApiError } from "./api.js";

async function requestJson(url, options = {}) {
  let response;

  try {
    response = await fetch(url, options);
  } catch {
    throw new ApiError("Cannot reach the VoiceForge server.", 0, "network_error");
  }

  let payload = null;

  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok || !payload || payload.ok !== true) {
    throw new ApiError(
      (payload && payload.error) || `Request failed (HTTP ${response.status}).`,
      response.status,
      (payload && payload.code) || "request_failed",
    );
  }

  return payload;
}

function jsonBody(body) {
  return {
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

export function toBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  const chunk = 0x8000;
  let binary = "";

  for (let index = 0; index < bytes.length; index += chunk) {
    binary += String.fromCharCode(...bytes.subarray(index, index + chunk));
  }

  return btoa(binary);
}

export async function extractDocument(file) {
  const buffer = await file.arrayBuffer();
  const payload = await requestJson("/api/documents/extract", {
    method: "POST",
    ...jsonBody({
      name: file.name,
      data_base64: toBase64(buffer),
    }),
  });

  return payload.document;
}

export async function listPronunciation(query = "") {
  const suffix = query ? `?q=${encodeURIComponent(query)}` : "";
  const payload = await requestJson(`/api/pronunciation${suffix}`);

  return payload.entries;
}

export async function createPronunciation(entry) {
  const payload = await requestJson("/api/pronunciation", {
    method: "POST",
    ...jsonBody(entry),
  });

  return payload.entry;
}

export async function updatePronunciation(entryId, entry) {
  const payload = await requestJson(
    `/api/pronunciation/${encodeURIComponent(entryId)}`,
    { method: "PATCH", ...jsonBody(entry) },
  );

  return payload.entry;
}

export async function deletePronunciation(entryId) {
  const payload = await requestJson(
    `/api/pronunciation/${encodeURIComponent(entryId)}`,
    { method: "DELETE" },
  );

  return payload.deleted;
}

export async function listPresets() {
  const payload = await requestJson("/api/presets");

  return payload.presets;
}

export async function createPreset(preset) {
  const payload = await requestJson("/api/presets", {
    method: "POST",
    ...jsonBody(preset),
  });

  return payload.preset;
}

export async function updatePreset(presetId, preset) {
  const payload = await requestJson(
    `/api/presets/${encodeURIComponent(presetId)}`,
    { method: "PATCH", ...jsonBody(preset) },
  );

  return payload.preset;
}

export async function deletePreset(presetId) {
  const payload = await requestJson(
    `/api/presets/${encodeURIComponent(presetId)}`,
    { method: "DELETE" },
  );

  return payload.deleted;
}
