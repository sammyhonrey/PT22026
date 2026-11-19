import { initDashboard, refreshDashboard } from "./dashboard.js";
import { initHistory, refreshHistory } from "./history.js";
import { initPlayground } from "./playground.js";
import { initReview } from "./review.js";
import { initSocial } from "./social.js";

const pageNames = {
  dashboard: "Sentiment Analytics Engine",
  playground: "AI Playground",
  review: "ReviewAI",
  social: "SocialMind",
  history: "History",
};

function showPage(name, updateLocation = true) {
  if (!pageNames[name]) name = "dashboard";
  for (const page of document.querySelectorAll("[data-page-panel]")) {
    const active = page.dataset.pagePanel === name;
    page.hidden = !active;
    page.classList.toggle("is-visible", active);
  }
  for (const button of document.querySelectorAll(".nav-item[data-page]")) {
    const active = button.dataset.page === name;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-current", active ? "page" : "false");
  }
  document.querySelector("#breadcrumb-current").textContent = pageNames[name];
  document.title = `SmartFeel AI | ${pageNames[name]}`;
  if (updateLocation && window.location.hash !== `#${name}`) {
    window.history.replaceState(null, "", `#${name}`);
  }
  if (name === "dashboard") refreshDashboard();
  if (name === "history") refreshHistory();
}

function initNavigation() {
  document.querySelectorAll(".nav-item[data-page]").forEach((button) => {
    button.addEventListener("click", () => showPage(button.dataset.page));
  });
  document.querySelectorAll("[data-go-page]").forEach((button) => {
    button.addEventListener("click", () => showPage(button.dataset.goPage));
  });
  window.addEventListener("hashchange", () => showPage(window.location.hash.slice(1), false));
}

initNavigation();
initPlayground();
initReview();
initSocial();
initHistory();
initDashboard();
showPage(window.location.hash.slice(1) || "dashboard", false);
