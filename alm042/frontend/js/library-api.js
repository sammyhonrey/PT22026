/**
 * Audio Library HTTP client.
 */

import { ApiError } from "./api.js";
import { buildLibraryQuery } from "./library-state.js";

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

export async function listLibrary(params) {
  return requestJson(`/api/library?${buildLibraryQuery(params)}`);
}

export async function getRecord(recordId) {
  const payload = await requestJson(
    `/api/library/${encodeURIComponent(recordId)}`,
  );

  return payload.record;
}

export async function renameRecord(recordId, name) {
  const payload = await requestJson(
    `/api/library/${encodeURIComponent(recordId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    },
  );

  return payload.record;
}

export async function deleteRecord(recordId) {
  const payload = await requestJson(
    `/api/library/${encodeURIComponent(recordId)}`,
    { method: "DELETE" },
  );

  return payload.deleted;
}

export function audioUrl(record) {
  return `/api/library/${encodeURIComponent(record.id)}/audio`;
}

export function downloadUrl(record) {
  return `${audioUrl(record)}?download=1`;
}
