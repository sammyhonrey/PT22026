/**
 * VoiceForge frontend bootstrap.
 *
 * Responsibilities: fetch capabilities from the backend, build the sidebar,
 * route between pages and keep the engine status indicator in sync.
 */

import { ApiError, fetchEngineInfo } from "./api.js";
import { createNotifications } from "./components/notifications.js";
import { createBatchPage } from "./pages/batch.js";
import { createDocumentsPage } from "./pages/documents.js";
import { createLibraryPage } from "./pages/library.js";
import { createPresetsPage } from "./pages/presets.js";
import { createPronunciationPage } from "./pages/pronunciation.js";
import { createComingSoonPage } from "./pages/coming-soon.js";
import { createStudioPage } from "./pages/studio.js";
import { createStore } from "./state.js";

const NAV_ITEMS = [
  {
    id: "studio",
    label: "Speech Studio",
    icon: "\u25c9",
    title: "Speech Studio",
    subtitle: "Turn text into speech with Kokoro or the VoiceForge custom model.",
    kind: "studio",
  },
  {
    id: "documents",
    label: "Document to Speech",
    icon: "\u25a3",
    title: "Document to Speech",
    subtitle: "Turn a PDF or DOCX into speech.",
    kind: "documents",
  },
  {
    id: "batch",
    label: "Batch Studio",
    icon: "\u25a4",
    title: "Batch Studio",
    subtitle: "Generate several entries at once and download them as a ZIP.",
    kind: "batch",
  },
  {
    id: "library",
    label: "Audio Library",
    icon: "\u266a",
    title: "Audio Library",
    subtitle: "Browse and reuse everything you have generated.",
    kind: "library",
  },
  {
    id: "pronunciation",
    label: "Pronunciation",
    icon: "\u270e",
    title: "Pronunciation Dictionary",
    subtitle: "Teach VoiceForge how names and terms should sound.",
    kind: "pronunciation",
  },
  {
    id: "presets",
    label: "Voice Presets",
    icon: "\u2605",
    title: "Voice Presets",
    subtitle: "Save engine, voice and speed combinations.",
    kind: "presets",
  },
];

const dom = {
  nav: document.getElementById("nav"),
  pageRoot: document.getElementById("page-root"),
  pageTitle: document.getElementById("page-title"),
  pageSubtitle: document.getElementById("page-subtitle"),
  badges: document.getElementById("topbar-badges"),
  statusDot: document.getElementById("engine-status-dot"),
  statusText: document.getElementById("engine-status-text"),
  sidebar: document.getElementById("sidebar"),
  navToggle: document.getElementById("nav-toggle"),
  toastStack: document.getElementById("toast-stack"),
};

const notifications = createNotifications(dom.toastStack);
const store = createStore({ capabilities: null, page: "studio", error: null });

let studioPage = null;
let studioHost = null;
let batchPage = null;
let batchHost = null;
let libraryPage = null;
let libraryHost = null;
let documentsPage = null;
let documentsHost = null;
let pronunciationPage = null;
let pronunciationHost = null;
let presetsPage = null;
let presetsHost = null;
let activePage = null;

function mountPage(page) {
  if (activePage && activePage !== page) {
    activePage.deactivate?.();
  }

  activePage = page;
}

function currentItem() {
  const { page } = store.get();
  return NAV_ITEMS.find((item) => item.id === page) || NAV_ITEMS[0];
}

function buildNav() {
  dom.nav.replaceChildren();

  for (const item of NAV_ITEMS) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "nav-item";
    button.dataset.page = item.id;
    button.setAttribute("aria-current", "false");

    const icon = document.createElement("span");
    icon.className = "nav-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = item.icon;

    const label = document.createElement("span");
    label.className = "nav-label";
    label.textContent = item.label;

    button.append(icon, label);

    if (item.kind === "soon") {
      const tag = document.createElement("span");
      tag.className = "nav-tag";
      tag.textContent = "Soon";
      button.append(tag);
    }

    button.addEventListener("click", () => {
      store.set({ page: item.id });
      dom.sidebar.classList.remove("is-open");
      dom.navToggle.setAttribute("aria-expanded", "false");
    });

    dom.nav.append(button);
  }
}

function markActiveNav() {
  const { page } = store.get();

  for (const button of dom.nav.querySelectorAll(".nav-item")) {
    const active = button.dataset.page === page;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-current", active ? "page" : "false");
  }
}

function renderStatus() {
  const { capabilities, error } = store.get();

  if (capabilities) {
    const engineList = capabilities.engines || {};
    const engineCount = Object.keys(engineList).length;
    const defaultId = capabilities.default;
    const defaultName = engineList[defaultId]?.name || defaultId || "engine";

    dom.statusDot.className = "status-dot is-ready";
    dom.statusText.textContent = `${engineCount} engines \u00b7 ${defaultName.split(" —")[0]} default`;
    return;
  }

  dom.statusDot.className = error ? "status-dot is-down" : "status-dot";
  dom.statusText.textContent = error ? "Backend unavailable" : "Checking engines\u2026";
}

function renderBadges() {
  const { capabilities } = store.get();
  dom.badges.replaceChildren();

  if (!capabilities) {
    return;
  }

  const defaultId = capabilities.default;
  const defaultName = capabilities.engines?.[defaultId]?.name;

  if (defaultName) {
    const engineBadge = document.createElement("span");
    engineBadge.className = "badge badge-accent";
    engineBadge.textContent = defaultName.split(" —")[0];
    dom.badges.append(engineBadge);
  }

  // The short-text limit only applies to Speech Studio, so it is shown
  // there by the character counter rather than as a global badge.
}

function renderErrorState(error) {
  const panel = document.createElement("div");
  panel.className = "coming-soon";

  const badge = document.createElement("span");
  badge.className = "badge";
  badge.textContent = "Connection problem";

  const heading = document.createElement("h2");
  heading.textContent = "The VoiceForge backend did not respond";

  const message = document.createElement("p");
  message.textContent = error?.message || "Unknown error.";

  const retry = document.createElement("button");
  retry.type = "button";
  retry.className = "button button-ghost";
  retry.textContent = "Retry";
  retry.addEventListener("click", () => boot());

  const hint = document.createElement("p");
  hint.className = "meta-key";
  hint.textContent =
    "Start the server with: python app.py \u2014 then reload this page.";

  panel.append(badge, heading, message, retry, hint);
  return panel;
}

function renderLoadingState() {
  const panel = document.createElement("div");
  panel.className = "coming-soon";

  const spinner = document.createElement("span");
  spinner.className = "spinner";

  const text = document.createElement("p");
  text.textContent = "Loading engines\u2026";

  panel.append(spinner, text);
  return panel;
}

function renderPage() {
  const item = currentItem();

  dom.pageTitle.textContent = item.title;
  dom.pageSubtitle.textContent = item.subtitle;
  markActiveNav();

  const { capabilities, error } = store.get();

  if (item.kind === "studio") {
    if (!capabilities) {
      dom.pageRoot.replaceChildren(
        error ? renderErrorState(error) : renderLoadingState(),
      );
      return;
    }

    if (!studioPage) {
      studioPage = createStudioPage({ capabilities, notifications });
      studioHost = document.createElement("div");
      studioHost.append(studioPage.element);
    }

    mountPage(studioPage);
    dom.pageRoot.replaceChildren(studioHost);
    return;
  }

  if (item.kind === "batch") {
    if (!capabilities) {
      dom.pageRoot.replaceChildren(
        error ? renderErrorState(error) : renderLoadingState(),
      );
      return;
    }

    if (!batchPage) {
      batchPage = createBatchPage({ capabilities, notifications });
      batchHost = document.createElement("div");
      batchHost.append(batchPage.element);
    }

    mountPage(batchPage);
    dom.pageRoot.replaceChildren(batchHost);
    return;
  }

  if (item.kind === "documents") {
    if (!capabilities) {
      dom.pageRoot.replaceChildren(
        error ? renderErrorState(error) : renderLoadingState(),
      );
      return;
    }

    if (!documentsPage) {
      documentsPage = createDocumentsPage({ capabilities, notifications });
      documentsHost = document.createElement("div");
      documentsHost.append(documentsPage.element);
    }

    mountPage(documentsPage);
    dom.pageRoot.replaceChildren(documentsHost);
    return;
  }

  if (item.kind === "pronunciation") {
    if (!pronunciationPage) {
      pronunciationPage = createPronunciationPage({ notifications });
      pronunciationHost = document.createElement("div");
      pronunciationHost.append(pronunciationPage.element);
    }

    mountPage(pronunciationPage);
    dom.pageRoot.replaceChildren(pronunciationHost);
    return;
  }

  if (item.kind === "presets") {
    if (!capabilities) {
      dom.pageRoot.replaceChildren(
        error ? renderErrorState(error) : renderLoadingState(),
      );
      return;
    }

    if (!presetsPage) {
      presetsPage = createPresetsPage({ capabilities, notifications });
      presetsHost = document.createElement("div");
      presetsHost.append(presetsPage.element);
    }

    mountPage(presetsPage);
    dom.pageRoot.replaceChildren(presetsHost);
    return;
  }

  if (item.kind === "library") {
    if (!capabilities) {
      dom.pageRoot.replaceChildren(
        error ? renderErrorState(error) : renderLoadingState(),
      );
      return;
    }

    if (!libraryPage) {
      libraryPage = createLibraryPage({ capabilities, notifications });
      libraryHost = document.createElement("div");
      libraryHost.append(libraryPage.element);
    }

    mountPage(libraryPage);
    dom.pageRoot.replaceChildren(libraryHost);
    return;
  }

  mountPage(null);


  const page = createComingSoonPage({
    title: item.title,
    description: item.description,
  });

  dom.pageRoot.replaceChildren(page.element);
}

async function boot() {
  store.set({ error: null });
  renderPage();

  try {
    const capabilities = await fetchEngineInfo();
    store.set({ capabilities });
  } catch (error) {
    const message =
      error instanceof ApiError
        ? error.message
        : "Could not load engine information.";

    store.set({ capabilities: null, error: new Error(message) });
    notifications.error(message);
  }
}

// --- wiring ---------------------------------------------------------------
buildNav();

store.subscribe(() => {
  renderStatus();
  renderBadges();
  renderPage();
});

dom.navToggle.addEventListener("click", () => {
  const open = dom.sidebar.classList.toggle("is-open");
  dom.navToggle.setAttribute("aria-expanded", open ? "true" : "false");
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    dom.sidebar.classList.remove("is-open");
    dom.navToggle.setAttribute("aria-expanded", "false");
  }
});

boot();
