/**
 * Pure helpers for job status display and polling decisions.
 * No DOM access, so they are unit-tested with Node.
 */

export const TERMINAL_STATUSES = ["completed", "failed"];

export function isTerminal(status) {
  return TERMINAL_STATUSES.includes(status);
}

const LABELS = {
  queued: "Queued",
  running: "Processing",
  completed: "Completed",
  failed: "Failed",
};

export function statusLabel(status) {
  return LABELS[status] || String(status || "Unknown");
}

/** "2 of 3 items completed" — real counts, never a fake percentage. */
export function formatProgress(job) {
  if (!job) {
    return "";
  }

  const total = Number(job.total) || 0;
  const completed = Number(job.completed) || 0;
  const failed = Number(job.failed) || 0;

  let text = `${completed + failed} of ${total} items finished`;

  if (failed > 0) {
    text += ` (${failed} failed)`;
  }

  return text;
}

/** Should the UI keep polling? Stops on terminal status and hidden tabs. */
export function shouldPoll(job, { hidden = false } = {}) {
  if (!job) {
    return true;
  }

  if (hidden) {
    return false;
  }

  return !isTerminal(job.status);
}

/** Completed items that expose a playable/downloadable WAV. */
export function completedItems(job) {
  if (!job || !Array.isArray(job.items)) {
    return [];
  }

  return job.items.filter(
    (item) => item.status === "completed" && item.download_url,
  );
}
