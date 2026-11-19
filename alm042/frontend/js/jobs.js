/**
 * Job API client and polling helper.
 */

import { ApiError } from "./api.js";
import { isTerminal } from "./job-format.js";

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

export async function submitJob(payload) {
  const response = await requestJson("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  return response.job;
}

export async function fetchJob(jobId) {
  const response = await requestJson(`/api/jobs/${encodeURIComponent(jobId)}`);

  return response.job;
}

export async function uploadTextFiles(files) {
  const response = await requestJson("/api/uploads", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ files }),
  });

  return response.files;
}

/**
 * Poll a job until it reaches a terminal status.
 *
 * Fetching pauses while the tab is hidden and resumes when it is visible
 * again, so an idle page does not poll forever.
 */
export function createJobPoller({
  jobId,
  onUpdate,
  onError,
  interval = 900,
  isHidden = () => document.hidden,
}) {
  let stopped = false;
  let timer = null;

  function schedule() {
    if (stopped) {
      return;
    }

    timer = window.setTimeout(tick, interval);
  }

  async function tick() {
    if (stopped) {
      return;
    }

    if (isHidden()) {
      schedule();
      return;
    }

    try {
      const job = await fetchJob(jobId);

      if (stopped) {
        return;
      }

      onUpdate(job);

      if (isTerminal(job.status)) {
        stopped = true;
        return;
      }
    } catch (error) {
      if (stopped) {
        return;
      }

      onError?.(error);
      stopped = true;
      return;
    }

    schedule();
  }

  return {
    start() {
      tick();
    },

    stop() {
      stopped = true;

      if (timer !== null) {
        window.clearTimeout(timer);
        timer = null;
      }
    },

    get isStopped() {
      return stopped;
    },
  };
}
