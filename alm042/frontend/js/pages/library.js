/**
 * Audio Library page: search, sort, filter, play, rename, delete.
 *
 * Records come from `GET /api/library`; the browser never filters a stale
 * local list for search.
 */

import { ApiError } from "../api.js";
import { buildMp3Url } from "../advanced-state.js";
import {
  audioUrl,
  deleteRecord,
  downloadUrl,
  listLibrary,
  renameRecord,
} from "../library-api.js";
import {
  SOURCE_LABELS,
  SOURCE_OPTIONS,
  canGoNext,
  canGoPrevious,
  describeRecord,
  formatBytes,
  paginationSummary,
  truncateText,
  validateLibraryName,
} from "../library-state.js";

const PAGE_SIZE = 10;

function option(value, label) {
  const element = document.createElement("option");
  element.value = value;
  element.textContent = label;
  return element;
}

export function createLibraryPage({ capabilities, notifications }) {
  const engines = capabilities.engines || {};
  const engineIds = Object.keys(engines);

  const state = {
    q: "",
    sort: "newest",
    engine: "",
    source: "",
    page: 1,
    pages: 0,
    total: 0,
    mp3Bitrate: 192,
    records: [],
    loading: false,
    error: null,
  };

  // --- Toolbar ----------------------------------------------------------
  const panel = document.createElement("section");
  panel.className = "panel";

  const head = document.createElement("div");
  head.className = "panel-head";

  const title = document.createElement("h2");
  title.className = "panel-title";
  title.textContent = "Recordings";

  const summary = document.createElement("span");
  summary.className = "panel-hint";

  head.append(title, summary);

  const body = document.createElement("div");
  body.className = "panel-body stack";

  const toolbar = document.createElement("div");
  toolbar.className = "library-toolbar";

  const search = document.createElement("input");
  search.className = "input";
  search.type = "search";
  search.placeholder = "Search by name or text…";
  search.setAttribute("aria-label", "Search recordings");

  const sortSelect = document.createElement("select");
  sortSelect.className = "select";
  sortSelect.setAttribute("aria-label", "Sort recordings");
  sortSelect.append(
    option("newest", "Newest first"),
    option("oldest", "Oldest first"),
    option("name", "Name (A–Z)"),
    option("duration", "Longest first"),
  );

  const engineSelect = document.createElement("select");
  engineSelect.className = "select";
  engineSelect.setAttribute("aria-label", "Filter by engine");
  engineSelect.append(option("", "All engines"));

  for (const id of engineIds) {
    engineSelect.append(option(id, engines[id].name || id));
  }

  const sourceSelect = document.createElement("select");
  sourceSelect.className = "select";
  sourceSelect.setAttribute("aria-label", "Filter by source");
  sourceSelect.append(option("", "All sources"));

  for (const source of SOURCE_OPTIONS) {
    sourceSelect.append(option(source, SOURCE_LABELS[source]));
  }

  const bitrateSelect = document.createElement("select");
  bitrateSelect.className = "select";
  bitrateSelect.setAttribute("aria-label", "MP3 quality");
  bitrateSelect.append(
    option("128", "MP3 128 kbps"),
    option("192", "MP3 192 kbps"),
    option("320", "MP3 320 kbps"),
  );
  bitrateSelect.value = "192";

  toolbar.append(search, sortSelect, engineSelect, sourceSelect, bitrateSelect);

  const list = document.createElement("div");
  list.className = "library-list";

  const status = document.createElement("p");
  status.className = "library-status";
  status.hidden = true;

  const pager = document.createElement("div");
  pager.className = "library-pager";

  const prevButton = document.createElement("button");
  prevButton.type = "button";
  prevButton.className = "button button-ghost";
  prevButton.textContent = "← Previous";

  const pageLabel = document.createElement("span");
  pageLabel.className = "library-page-label";

  const nextButton = document.createElement("button");
  nextButton.type = "button";
  nextButton.className = "button button-ghost";
  nextButton.textContent = "Next →";

  pager.append(prevButton, pageLabel, nextButton);
  body.append(toolbar, status, list, pager);
  panel.append(head, body);

  const wrapper = document.createElement("div");
  wrapper.append(panel);

  // --- Rendering --------------------------------------------------------
  function setStatus(message, kind = "info") {
    if (!message) {
      status.hidden = true;
      status.textContent = "";
      return;
    }

    status.hidden = false;
    status.textContent = message;
    status.dataset.kind = kind;
  }

  function renderEmpty() {
    const empty = document.createElement("div");
    empty.className = "output-empty";

    const icon = document.createElement("span");
    icon.className = "output-empty-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = "♪";

    const line = document.createElement("p");
    line.textContent = state.q || state.engine || state.source
      ? "No recordings match these filters."
      : "Your generated speech will be saved here automatically.";

    const hint = document.createElement("p");
    hint.className = "meta-key";
    hint.textContent = "Generate speech in Speech Studio or Batch Studio";

    empty.append(icon, line, hint);

    return empty;
  }

  function renderRecord(record) {
    const card = document.createElement("article");
    card.className = "library-card";
    card.dataset.id = record.id;

    const top = document.createElement("div");
    top.className = "library-card-head";

    const name = document.createElement("h3");
    name.className = "library-card-name";
    name.textContent = record.name;

    const actions = document.createElement("div");
    actions.className = "library-card-actions";

    const renameButton = document.createElement("button");
    renameButton.type = "button";
    renameButton.className = "button button-ghost button-small";
    renameButton.textContent = "Rename";

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "button button-ghost button-small is-danger";
    deleteButton.textContent = "Delete";

    actions.append(renameButton, deleteButton);
    top.append(name, actions);

    const meta = document.createElement("p");
    meta.className = "library-card-meta";
    meta.textContent = describeRecord(record);

    const preview = document.createElement("p");
    preview.className = "library-card-text";
    preview.textContent = truncateText(record.original_text);

    const audio = document.createElement("audio");
    audio.className = "audio-player";
    audio.controls = true;
    audio.preload = "none";
    audio.src = audioUrl(record);

    const download = document.createElement("a");
    download.className = "button button-ghost button-small";
    download.href = downloadUrl(record);
    download.setAttribute("download", `${record.name}.wav`);
    download.textContent = "⬇ WAV";

    const mp3 = document.createElement("a");
    mp3.className = "button button-ghost button-small";
    mp3.href = buildMp3Url(record.id, state.mp3Bitrate);
    mp3.setAttribute("download", "");
    mp3.textContent = "MP3";

    const player = document.createElement("div");
    player.className = "library-card-player";
    player.append(audio, download, mp3);

    card.append(top, meta, preview, player);

    // --- Rename -----------------------------------------------------
    renameButton.addEventListener("click", () => {
      const editor = document.createElement("div");
      editor.className = "library-rename";

      const input = document.createElement("input");
      input.className = "input";
      input.type = "text";
      input.value = record.name;
      input.maxLength = 80;
      input.setAttribute("aria-label", "New name");

      const save = document.createElement("button");
      save.type = "button";
      save.className = "button button-primary button-small";
      save.textContent = "Save";

      const cancel = document.createElement("button");
      cancel.type = "button";
      cancel.className = "button button-ghost button-small";
      cancel.textContent = "Cancel";

      editor.append(input, save, cancel);
      top.replaceWith(editor);
      input.focus();
      input.select();

      const close = () => {
        editor.replaceWith(top);
      };

      cancel.addEventListener("click", close);

      input.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          save.click();
        } else if (event.key === "Escape") {
          close();
        }
      });

      save.addEventListener("click", async () => {
        const validation = validateLibraryName(input.value);

        if (!validation.ok) {
          notifications.error(validation.error);
          return;
        }

        save.disabled = true;

        try {
          const updated = await renameRecord(record.id, validation.name);
          notifications.success(`Renamed to "${updated.name}".`);
          await load();
        } catch (error) {
          notifications.error(
            error instanceof ApiError ? error.message : "Rename failed.",
          );
          save.disabled = false;
        }
      });
    });

    // --- Delete -----------------------------------------------------
    deleteButton.addEventListener("click", async () => {
      const confirmed = window.confirm(
        `Delete "${record.name}"?\n\nThe audio file will be removed permanently.`,
      );

      if (!confirmed) {
        return;
      }

      deleteButton.disabled = true;

      try {
        await deleteRecord(record.id);
        notifications.success("Recording deleted.");
        await load();
      } catch (error) {
        notifications.error(
          error instanceof ApiError ? error.message : "Delete failed.",
        );
        deleteButton.disabled = false;
      }
    });

    return card;
  }

  function render() {
    list.replaceChildren();
    summary.textContent = paginationSummary({
      page: state.page,
      pages: state.pages,
      total: state.total,
    });

    if (state.loading) {
      setStatus("Loading recordings…", "info");
      return;
    }

    if (state.error) {
      setStatus(state.error, "error");

      const retry = document.createElement("button");
      retry.type = "button";
      retry.className = "button button-ghost button-small";
      retry.textContent = "Retry";
      retry.addEventListener("click", () => load());
      list.append(retry);
      return;
    }

    setStatus("");

    if (state.records.length === 0) {
      list.append(renderEmpty());
    } else {
      list.append(...state.records.map(renderRecord));
    }

    pageLabel.textContent = `${state.page} / ${Math.max(1, state.pages)}`;
    prevButton.disabled = !canGoPrevious(state.page);
    nextButton.disabled = !canGoNext(state.page, state.pages);
    pager.hidden = state.total === 0;
  }

  // --- Data -------------------------------------------------------------
  async function load() {
    state.loading = true;
    state.error = null;
    render();

    try {
      const payload = await listLibrary({
        q: state.q,
        sort: state.sort,
        engine: state.engine,
        source: state.source,
        page: state.page,
        limit: PAGE_SIZE,
      });

      state.records = payload.records;
      state.page = payload.pagination.page;
      state.pages = payload.pagination.pages;
      state.total = payload.pagination.total;

      // If the last item on a page was deleted, step back one page.
      if (state.records.length === 0 && state.page > 1) {
        state.page -= 1;
        state.loading = false;
        return load();
      }
    } catch (error) {
      state.error =
        error instanceof ApiError ? error.message : "Could not load the library.";
      state.records = [];
    } finally {
      state.loading = false;
      render();
    }
  }

  // --- Events -----------------------------------------------------------
  let debounce = null;

  search.addEventListener("input", () => {
    window.clearTimeout(debounce);
    debounce = window.setTimeout(() => {
      state.q = search.value;
      state.page = 1;
      load();
    }, 300);
  });

  sortSelect.addEventListener("change", () => {
    state.sort = sortSelect.value;
    state.page = 1;
    load();
  });

  engineSelect.addEventListener("change", () => {
    state.engine = engineSelect.value;
    state.page = 1;
    load();
  });

  sourceSelect.addEventListener("change", () => {
    state.source = sourceSelect.value;
    state.page = 1;
    load();
  });

  bitrateSelect.addEventListener("change", () => {
    state.mp3Bitrate = Number(bitrateSelect.value);
    render();
  });

  prevButton.addEventListener("click", () => {
    if (canGoPrevious(state.page)) {
      state.page -= 1;
      load();
    }
  });

  nextButton.addEventListener("click", () => {
    if (canGoNext(state.page, state.pages)) {
      state.page += 1;
      load();
    }
  });

  load();

  return {
    element: wrapper,
    refresh: load,

    deactivate() {
      window.clearTimeout(debounce);
    },
  };
}
