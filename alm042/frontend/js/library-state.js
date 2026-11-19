/**
 * Pure helpers for the Audio Library page (no DOM, unit-tested).
 */

export const SORT_OPTIONS = ["newest", "oldest", "name", "duration"];
export const DEFAULT_PAGE_SIZE = 20;
export const MAX_NAME_LENGTH = 80;
export const SOURCE_OPTIONS = ["studio", "long_text", "batch"];
export const SOURCE_LABELS = {
  studio: "Speech Studio",
  long_text: "Long text",
  batch: "Batch",
};

/**
 * Build the query string for GET /api/library with safe defaults.
 */
export function buildLibraryQuery({
  q = "",
  sort = "newest",
  engine = "",
  source = "",
  page = 1,
  limit = DEFAULT_PAGE_SIZE,
} = {}) {
  const params = new URLSearchParams();
  const text = String(q || "").trim();

  if (text) {
    params.set("q", text);
  }

  params.set("sort", SORT_OPTIONS.includes(sort) ? sort : "newest");

  if (engine) {
    params.set("engine", engine);
  }

  if (source && SOURCE_OPTIONS.includes(source)) {
    params.set("source", source);
  }

  const safePage = Math.max(1, Number(page) || 1);
  const safeLimit = Math.min(
    Math.max(1, Number(limit) || DEFAULT_PAGE_SIZE),
    100,
  );

  params.set("page", String(safePage));
  params.set("limit", String(safeLimit));

  return params.toString();
}

export function validateLibraryName(name) {
  const cleaned = String(name ?? "").trim();

  if (!cleaned) {
    return { ok: false, error: "The name cannot be empty." };
  }

  if (cleaned.length > MAX_NAME_LENGTH) {
    return {
      ok: false,
      error: `The name is too long (${cleaned.length} characters). The limit is ${MAX_NAME_LENGTH}.`,
    };
  }

  return { ok: true, name: cleaned };
}

export function paginationSummary(pagination) {
  if (!pagination || !pagination.total) {
    return "No recordings";
  }

  const pages = Math.max(1, Number(pagination.pages) || 1);
  const page = Math.max(1, Number(pagination.page) || 1);
  const total = Number(pagination.total) || 0;

  return `Page ${page} of ${pages} · ${total} recording${total === 1 ? "" : "s"}`;
}

export function canGoPrevious(page) {
  return Number(page) > 1;
}

export function canGoNext(page, pages) {
  return Number(page) < Math.max(1, Number(pages) || 1);
}

export function isEmptyLibrary(payload) {
  return Boolean(payload) && Array.isArray(payload.records) && payload.records.length === 0;
}

export function formatBytes(bytes) {
  const value = Number(bytes);

  if (!Number.isFinite(value) || value <= 0) {
    return "—";
  }

  if (value < 1024) {
    return `${value} B`;
  }

  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }

  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatDuration(seconds) {
  const value = Number(seconds);

  if (!Number.isFinite(value) || value < 0) {
    return "—";
  }

  if (value < 60) {
    return `${value.toFixed(2)} s`;
  }

  const minutes = Math.floor(value / 60);
  const rest = Math.round(value % 60);

  return `${minutes}:${String(rest).padStart(2, "0")}`;
}

export function formatDate(iso) {
  if (!iso) {
    return "—";
  }

  const date = new Date(iso);

  if (Number.isNaN(date.getTime())) {
    return "—";
  }

  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function truncateText(text, maxLength = 160) {
  const cleaned = String(text || "").replace(/\s+/g, " ").trim();

  if (cleaned.length <= maxLength) {
    return cleaned;
  }

  return `${cleaned.slice(0, maxLength - 1).trimEnd()}\u2026`;
}

export function describeRecord(record) {
  if (!record) {
    return "";
  }

  const parts = [
    formatDate(record.created_at),
    record.engine,
    record.voice || "default voice",
  ];

  if (typeof record.speed === "number") {
    parts.push(`${record.speed.toFixed(2)}×`);
  }

  parts.push(formatDuration(record.duration_seconds));
  parts.push(formatBytes(record.file_size));
  parts.push(SOURCE_LABELS[record.source_type] || record.source_type);

  return parts.filter(Boolean).join(" · ");
}
