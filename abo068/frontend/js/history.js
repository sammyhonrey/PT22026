import { clearHistory, exportHistory, getDashboard, getHistory } from "./api.js";
import { formatTime, showError } from "./ui.js";

const pageSize = 8;
let currentPage = 0;
let latestRequest = 0;
let searchTimer;

function currentFilters() {
  const search = document.querySelector("#history-search").value.trim();
  const sentiment = document.querySelector("#history-sentiment").value;
  const source = document.querySelector("#history-source").value;
  const days = document.querySelector("#history-days").value;
  return {
    ...(search ? { search } : {}),
    ...(sentiment !== "All" ? { sentiment } : {}),
    ...(source !== "All" ? { source } : {}),
    ...(days ? { days } : {}),
  };
}

function setMetrics(data) {
  document.querySelector("#history-total").textContent = Number(data.total_analyses).toLocaleString();
  for (const sentiment of ["Positive", "Neutral", "Negative"]) {
    document.querySelector(`#history-${sentiment.toLowerCase()}`).textContent =
      Number(data.counts[sentiment] || 0).toLocaleString();
  }
  const demoCount = Number(data.demo_records_count || 0);
  const demoNote = document.querySelector("#history-demo-count");
  demoNote.textContent = demoCount
    ? `${demoCount} fictional demo reviews or comments included`
    : "";
}

function addSourceCell(row, record) {
  const cell = document.createElement("td");
  const tag = document.createElement("span");
  tag.className = "source-tag";
  tag.textContent = record.source || "Unknown";
  cell.append(tag);
  if (record.is_demo) {
    const demoTag = document.createElement("span");
    demoTag.className = "demo-tag";
    demoTag.textContent = "DEMO";
    cell.append(demoTag);
  }
  row.append(cell);
}

function addSentimentCell(row, sentiment) {
  const cell = document.createElement("td");
  const badge = document.createElement("span");
  badge.className = `sentiment-badge ${(sentiment || "").toLowerCase()}`;
  badge.textContent = sentiment || "Unknown";
  cell.append(badge);
  row.append(cell);
}

function renderRows(records) {
  const body = document.querySelector("#history-rows");
  body.replaceChildren();
  if (!records.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 5;
    cell.className = "table-message";
    cell.textContent = "No saved analyses match these filters.";
    row.append(cell);
    body.append(row);
    return;
  }

  for (const record of records) {
    const row = document.createElement("tr");
    addSourceCell(row, record);
    const text = document.createElement("td");
    const recordText = typeof record.text === "string" ? record.text : "";
    const reviewText = recordText.split("Review: ").pop().replace(/\s+/g, " ");
    text.textContent = record.customer_name
      ? `${record.customer_name} · ${record.product_name || "Customer review"} · ${reviewText}`
      : recordText.replace(/\s+/g, " ");
    text.title = text.textContent;
    row.append(text);
    addSentimentCell(row, record.sentiment);

    const confidence = document.createElement("td");
    confidence.className = "confidence-cell";
    const value = Number(record.confidence || 0);
    confidence.append(document.createTextNode(`${value.toFixed(1)}%`));
    const track = document.createElement("span");
    track.className = "confidence-track";
    const fill = document.createElement("i");
    fill.style.width = `${Math.max(0, Math.min(100, value))}%`;
    fill.style.backgroundColor = record.sentiment === "Negative"
      ? "var(--red)"
      : record.sentiment === "Neutral" ? "var(--amber)" : "var(--green)";
    track.append(fill);
    confidence.append(track);
    row.append(confidence);

    const date = document.createElement("td");
    date.textContent = formatTime(record.time);
    row.append(date);
    body.append(row);
  }
}

function updatePagination(total) {
  const pageCount = Math.max(1, Math.ceil(total / pageSize));
  document.querySelector("#history-page-number").textContent = `${currentPage + 1} / ${pageCount}`;
  document.querySelector("#history-prev").disabled = currentPage <= 0;
  document.querySelector("#history-next").disabled = currentPage + 1 >= pageCount;
  document.querySelector("#history-showing").textContent = total
    ? `Showing ${currentPage * pageSize + 1}–${Math.min((currentPage + 1) * pageSize, total)} of ${total.toLocaleString()} records`
    : "Showing 0 records";
}

export async function refreshHistory() {
  const requestId = ++latestRequest;
  showError("history-error");
  document.querySelector("#history-showing").textContent = "Loading records…";
  try {
    const [page, dashboard] = await Promise.all([
      getHistory({ ...currentFilters(), limit: pageSize, offset: currentPage * pageSize }),
      getDashboard(),
    ]);
    if (requestId !== latestRequest) return;
    setMetrics(dashboard);
    renderRows(page.records || []);
    updatePagination(page.total || 0);
  } catch (error) {
    if (requestId !== latestRequest) return;
    showError("history-error", error.message);
    document.querySelector("#history-showing").textContent = "Could not load history.";
  }
}

function scheduleRefresh() {
  currentPage = 0;
  window.clearTimeout(searchTimer);
  searchTimer = window.setTimeout(refreshHistory, 250);
}

export function initHistory() {
  document.querySelector("#history-search").addEventListener("input", scheduleRefresh);
  for (const selector of ["#history-sentiment", "#history-source", "#history-days"]) {
    document.querySelector(selector).addEventListener("change", scheduleRefresh);
  }

  document.querySelector("#history-prev").addEventListener("click", () => {
    if (currentPage > 0) {
      currentPage -= 1;
      refreshHistory();
    }
  });
  document.querySelector("#history-next").addEventListener("click", () => {
    currentPage += 1;
    refreshHistory();
  });

  document.querySelector("#history-export").addEventListener("click", async (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    showError("history-error");
    try {
      const blob = await exportHistory(currentFilters());
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "smartfeel_history.csv";
      document.body.append(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) {
      showError("history-error", error.message);
    } finally {
      button.disabled = false;
    }
  });

  document.querySelector("#history-clear").addEventListener("click", async (event) => {
    if (!window.confirm("Clear all saved analysis history? This cannot be undone.")) return;
    const button = event.currentTarget;
    button.disabled = true;
    showError("history-error");
    try {
      await clearHistory();
      currentPage = 0;
      await refreshHistory();
    } catch (error) {
      showError("history-error", error.message);
    } finally {
      button.disabled = false;
    }
  });
}
