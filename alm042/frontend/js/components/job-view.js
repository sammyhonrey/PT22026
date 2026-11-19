/**
 * Renders a job: overall status, real item counts and per-item results.
 *
 * Reused by Batch Studio and by Long Text mode in Speech Studio.
 */

import {
  buildJobItemMp3Url,
  buildJobZipUrl,
} from "../advanced-state.js";
import {
  completedItems,
  formatProgress,
  statusLabel,
} from "../job-format.js";

const STATUS_CLASS = {
  queued: "badge",
  running: "badge badge-accent",
  completed: "badge badge-success",
  failed: "badge badge-danger",
};

function statusBadge(status) {
  const badge = document.createElement("span");
  badge.className = STATUS_CLASS[status] || "badge";
  badge.textContent = statusLabel(status);
  return badge;
}

export function createJobView({ title = "Results" } = {}) {
  const panel = document.createElement("section");
  panel.className = "panel";
  panel.hidden = true;

  const head = document.createElement("div");
  head.className = "panel-head";

  const heading = document.createElement("h2");
  heading.className = "panel-title";
  heading.textContent = title;

  const headStatus = document.createElement("span");
  headStatus.className = "panel-hint";

  head.append(heading, headStatus);

  const body = document.createElement("div");
  body.className = "panel-body stack";

  const summary = document.createElement("p");
  summary.className = "job-summary";

  const list = document.createElement("div");
  list.className = "job-items";

  const actions = document.createElement("div");
  actions.className = "output-actions";

  const zipLink = document.createElement("a");
  zipLink.className = "button button-primary";
  zipLink.textContent = "\u2b07 Download ZIP";
  zipLink.setAttribute("download", "");
  zipLink.hidden = true;

  const mp3ZipLink = document.createElement("a");
  mp3ZipLink.className = "button button-ghost";
  mp3ZipLink.textContent = "\u2b07 Download MP3 ZIP";
  mp3ZipLink.setAttribute("download", "");
  mp3ZipLink.hidden = true;

  actions.append(zipLink, mp3ZipLink);
  body.append(summary, list, actions);
  panel.append(head, body);

  function renderItem(item, job) {
    const row = document.createElement("div");
    row.className = "job-item";
    row.dataset.status = item.status;

    const info = document.createElement("div");
    info.className = "job-item-info";

    const name = document.createElement("span");
    name.className = "job-item-name";
    name.textContent = item.name || item.id;

    const meta = document.createElement("span");
    meta.className = "job-item-meta";

    if (item.status === "completed") {
      meta.textContent =
        `${item.filename || ""} · ${item.chunks_completed}/${item.chunks_total} chunks`;
    } else if (item.status === "failed") {
      meta.textContent = item.error || "Generation failed";
    } else if (item.status === "running") {
      meta.textContent =
        `${item.chunks_completed || 0}/${item.chunks_total || "?"} chunks`;
    } else {
      meta.textContent = `${item.characters} characters`;
    }

    info.append(name, meta);
    row.append(info, statusBadge(item.status));

    if (item.status === "completed" && item.download_url) {
      const tools = document.createElement("div");
      tools.className = "job-item-tools";

      const audio = document.createElement("audio");
      audio.controls = true;
      audio.preload = "none";
      audio.src = item.download_url;
      audio.className = "job-item-audio";

      const download = document.createElement("a");
      download.className = "button button-ghost";
      download.href = item.download_url;
      download.setAttribute("download", item.filename || "");
      download.textContent = "WAV";

      const mp3 = document.createElement("a");
      mp3.className = "button button-ghost";
      mp3.href = buildJobItemMp3Url(job.id, item.id);
      mp3.setAttribute("download", "");
      mp3.textContent = "MP3";

      tools.append(audio, download, mp3);
      row.append(tools);
    }

    return row;
  }

  return {
    element: panel,

    render(job) {
      panel.hidden = false;
      headStatus.textContent = statusLabel(job.status);
      summary.textContent = formatProgress(job);

      if (job.error) {
        summary.textContent += ` — ${job.error}`;
      }

      list.replaceChildren(
        ...job.items.map((item) => renderItem(item, job)),
      );

      const results = completedItems(job);

      zipLink.hidden = results.length === 0;
      zipLink.href = job.download_url || "#";

      mp3ZipLink.hidden = results.length === 0;
      mp3ZipLink.href = buildJobZipUrl(job.id, "mp3");

      return job;
    },

    clear() {
      panel.hidden = true;
      list.replaceChildren();
      summary.textContent = "";
      zipLink.hidden = true;
    },
  };
}
