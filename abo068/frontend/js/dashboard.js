import { exportHistory, getDashboard } from "./api.js";

const colors = {
  Positive: "#087b4b",
  Neutral: "#d99213",
  Negative: "#d7252b",
};

let distributionChart;
let activityChart;
let latestRequest = 0;

function formatTime(value) {
  if (!value) return "—";
  const parsed = new Date(String(value).replace(" ", "T"));
  return Number.isNaN(parsed.getTime())
    ? String(value)
    : new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(parsed);
}

function formatMeanConfidence(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? `${value.toFixed(1)}%`
    : "—";
}

function updateChangeBadge(key, value, days) {
  const badge = document.querySelector(`[data-change="${key}"]`);
  badge.classList.remove("change-positive", "change-negative", "change-neutral");
  if (typeof value !== "number" || !Number.isFinite(value)) {
    badge.textContent = "—";
    badge.classList.add("change-neutral");
    badge.title = "No comparable previous-period data";
    return;
  }

  const favorable = key === "negative" ? value <= 0 : value >= 0;
  badge.textContent = `${value > 0 ? "+" : ""}${value.toFixed(1)}%`;
  badge.classList.add(key === "neutral" ? "change-neutral" : favorable ? "change-positive" : "change-negative");
  badge.title = `Change versus the previous ${days}-day period`;
}

function makeCell(row, value, className) {
  const cell = document.createElement("td");
  if (className) cell.className = className;
  cell.textContent = value ?? "";
  row.append(cell);
  return cell;
}

function sentimentBadge(value) {
  const badge = document.createElement("span");
  badge.className = `sentiment-badge ${(value || "").toLowerCase()}`;
  badge.textContent = value || "Unknown";
  return badge;
}

function formatCompactCount(value) {
  const count = Number(value) || 0;
  return count >= 1000 ? `${(count / 1000).toFixed(1)}k` : count.toLocaleString();
}

function renderRecent(records) {
  const body = document.querySelector("#recent-rows");
  body.replaceChildren();

  if (!records?.length) {
    const row = document.createElement("tr");
    const cell = makeCell(row, "No saved analyses yet.", "table-message");
    cell.colSpan = 5;
    body.append(row);
    document.querySelector("#recent-count").textContent = "0 records";
    return;
  }

  for (const record of records) {
    const row = document.createElement("tr");
    const source = makeCell(row, "", "");
    const sourceTag = document.createElement("span");
    sourceTag.className = "source-tag";
    sourceTag.textContent = record.source || "Unknown";
    source.append(sourceTag);
    if (record.is_demo) {
      const demoTag = document.createElement("span");
      demoTag.className = "demo-tag";
      demoTag.textContent = "DEMO";
      source.append(demoTag);
    }

    const recordText = typeof record.text === "string" ? record.text : "";
    const reviewText = recordText.split("Review: ").pop().replace(/\s+/g, " ");
    const snippet = record.customer_name
      ? `${record.customer_name} · ${record.product_name || "Customer review"} · ${reviewText}`
      : recordText.replace(/\s+/g, " ");
    const textCell = makeCell(row, snippet, "");
    textCell.title = snippet;
    textCell.classList.add("snippet-cell");
    const sentimentCell = makeCell(row, "", "");
    sentimentCell.append(sentimentBadge(record.sentiment));
    makeCell(row, `${Number(record.confidence || 0).toFixed(1)}%`, "confidence-cell");
    makeCell(row, formatTime(record.time), "");
    body.append(row);
  }
  document.querySelector("#recent-count").textContent = `${records.length} latest records`;
}

function updateCharts(data) {
  if (typeof Chart === "undefined") {
    throw new Error("Chart.js did not load. Check your internet connection and refresh.");
  }

  const distributionContext = document.querySelector("#distribution-chart");
  const activityContext = document.querySelector("#activity-chart");
  distributionChart?.destroy();
  activityChart?.destroy();

  const labels = ["Positive", "Neutral", "Negative"];
  const values = labels.map((label) => data.distribution?.[label]?.count || 0);
  distributionChart = new Chart(distributionContext, {
    type: "doughnut",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: labels.map((label) => colors[label]),
        borderColor: "#fff",
        borderWidth: 3,
        hoverOffset: 5,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "67%",
      plugins: {
        legend: {
          display: false,
        },
        tooltip: { callbacks: { label: (context) => `${context.label}: ${context.raw} analyses` } },
      },
    },
    plugins: [{
      id: "dashboardDoughnutCenter",
      afterDraw(chart) {
        const { ctx, chartArea } = chart;
        if (!chartArea) return;
        const centerX = (chartArea.left + chartArea.right) / 2;
        const centerY = (chartArea.top + chartArea.bottom) / 2;
        ctx.save();
        ctx.textAlign = "center";
        ctx.fillStyle = "#171d30";
        ctx.font = "700 20px Inter, Arial, sans-serif";
        ctx.fillText(formatCompactCount(data.total_analyses), centerX, centerY);
        ctx.fillStyle = "#7d879d";
        ctx.font = "10px Inter, Arial, sans-serif";
        ctx.fillText("TOTAL LOGS", centerX, centerY + 17);
        ctx.restore();
      },
    }],
  });

  const activity = aggregateWeeks(data.activity || []);
  const latestWeek = activity[activity.length - 1];
  const activityTooltip = document.querySelector("#activity-tooltip");
  if (latestWeek && Object.values(latestWeek).some((value) => typeof value === "number" && value > 0)) {
    activityTooltip.hidden = false;
    activityTooltip.textContent =
      `${latestWeek.label}\nPos: ${latestWeek.Positive.toLocaleString()}  Neu: ${latestWeek.Neutral.toLocaleString()}  Neg: ${latestWeek.Negative.toLocaleString()}`;
  } else {
    activityTooltip.hidden = true;
  }

  activityChart = new Chart(activityContext, {
    type: "line",
    data: {
      labels: activity.map((item) => item.week),
      datasets: labels.map((label) => ({
        label,
        data: activity.map((item) => item[label] || 0),
        borderColor: colors[label],
        backgroundColor: colors[label],
        borderWidth: label === "Negative" ? 3 : 2,
        borderDash: label === "Negative" ? [4, 3] : [],
        pointStyle: label === "Negative" ? "rectRot" : "circle",
        pointRadius: label === "Negative" ? 5 : 3,
        pointHoverRadius: 5,
        order: label === "Negative" ? 3 : label === "Neutral" ? 2 : 1,
        tension: 0.3,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          position: "top",
          align: "end",
          labels: { usePointStyle: true, pointStyle: "circle", boxWidth: 7, padding: 12, font: { size: 10 } },
        },
      },
      scales: {
        x: { grid: { display: false }, ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 6, font: { size: 10 } } },
        y: { beginAtZero: true, ticks: { precision: 0, font: { size: 10 } }, grid: { color: "#edf0f8" } },
      },
    },
  });
}

function aggregateWeeks(activity) {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const firstDay = new Date(today);
  firstDay.setDate(firstDay.getDate() - 27);
  const weeks = Array.from({ length: 4 }, (_, index) => {
    const start = new Date(firstDay);
    start.setDate(firstDay.getDate() + index * 7);
    const end = new Date(start);
    end.setDate(start.getDate() + 6);
    const startLabel = start.toLocaleDateString(undefined, { month: "short", day: "numeric" });
    const endLabel = end.toLocaleDateString(undefined, { month: "short", day: "numeric" });
    return {
      week: `Week ${index + 1}`,
      label: `Week ${index + 1} (${startLabel}–${endLabel})${index === 3 ? " · Active" : ""}`,
      Positive: 0,
      Neutral: 0,
      Negative: 0,
    };
  });

  for (const item of activity) {
    const date = new Date(`${item.date}T00:00:00`);
    const weekIndex = Math.floor((date.getTime() - firstDay.getTime()) / (7 * 24 * 60 * 60 * 1000));
    if (weekIndex < 0 || weekIndex >= weeks.length) continue;
    for (const label of ["Positive", "Neutral", "Negative"]) {
      weeks[weekIndex][label] += Number(item[label]) || 0;
    }
  }
  return weeks;
}

export async function refreshDashboard() {
  const requestId = ++latestRequest;
  const status = document.querySelector("#dashboard-status");
  status.textContent = "Loading dashboard data…";
  const days = document.querySelector("#dashboard-range").value;
  const source = document.querySelector("#dashboard-source").value;
  try {
    const data = await getDashboard({
      ...(days ? { days } : {}),
      ...(source !== "All" ? { source } : {}),
    });
    if (requestId !== latestRequest) return;
    document.querySelector('[data-metric="total"]').textContent = Number(data.total_analyses).toLocaleString();
    updateChangeBadge("total", data.period_change?.total, days);
    document.querySelector('[data-confidence="overall"]').textContent =
      `Mean confidence ${formatMeanConfidence(data.mean_confidence)}`;
    document.querySelector("#dashboard-mean-confidence").textContent = formatMeanConfidence(data.mean_confidence);
    document.querySelector('[data-progress="overall"]').style.width =
      `${Math.max(0, Math.min(100, Number(data.mean_confidence) || 0))}%`;
    for (const label of ["Positive", "Neutral", "Negative"]) {
      const key = label.toLowerCase();
      const classData = data.distribution[label];
      document.querySelector(`[data-metric="${key}"]`).textContent = Number(classData.count).toLocaleString();
      document.querySelector(`[data-share="${key}"]`).textContent = `${Number(classData.percentage).toFixed(1)}%`;
      updateChangeBadge(key, data.period_change?.distribution?.[label], days);
      document.querySelector(`[data-confidence="${key}"]`).textContent =
        `Mean confidence ${formatMeanConfidence(classData.mean_confidence)}`;
      document.querySelector(`[data-progress="${key}"]`).style.width =
        `${Math.max(0, Math.min(100, Number(classData.mean_confidence) || 0))}%`;
      document.querySelector(`[data-distribution-count="${key}"]`).textContent =
        Number(classData.count).toLocaleString();
      document.querySelector(`[data-distribution-share="${key}"]`).textContent =
        `${Number(classData.percentage).toFixed(1)}%`;
    }

    document.querySelector("#model-accuracy").textContent = `${Number(data.model_evaluation.accuracy).toFixed(2)}%`;
    document.querySelector("#model-f1").textContent = Number(data.model_evaluation.macro_f1).toFixed(3);
    document.querySelector("#model-dataset").textContent = data.model_evaluation.dataset;
    document.querySelector("#model-base").textContent = data.model_evaluation.base_model;
    document.querySelector("#dashboard-updated").textContent =
      new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date());
    const demoNotice = document.querySelector("#dashboard-demo-notice");
    const demoCount = Number(data.demo_records_count || 0);
    demoNotice.hidden = demoCount === 0;
    demoNotice.textContent = demoCount
      ? `${demoCount} fictional customer reviews or comments are included in these statistics and clearly labeled as demo data.`
      : "";
    renderRecent(data.recent_analyses);
    updateCharts(data);
    status.textContent = "";
  } catch (error) {
    if (requestId !== latestRequest) return;
    status.textContent = error.message;
  }
}

async function downloadReport(button) {
  const status = document.querySelector("#page-dashboard").classList.contains("is-visible")
    ? document.querySelector("#dashboard-status")
    : document.querySelector("#topbar-status");
  const days = document.querySelector("#dashboard-range").value;
  const source = document.querySelector("#dashboard-source").value;
  button.disabled = true;
  status.textContent = "Preparing filtered report…";
  try {
    const blob = await exportHistory({
      ...(days ? { days } : {}),
      ...(source !== "All" ? { source } : {}),
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "smartfeel_dashboard_report.csv";
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    status.textContent = "Filtered report downloaded.";
  } catch (error) {
    status.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

export function initDashboard() {
  document.querySelector("#dashboard-refresh").addEventListener("click", refreshDashboard);
  document.querySelector("#dashboard-range").addEventListener("change", refreshDashboard);
  document.querySelector("#dashboard-source").addEventListener("change", refreshDashboard);
  for (const button of ["#dashboard-export", "#topbar-export"]) {
    document.querySelector(button).addEventListener("click", (event) => {
      downloadReport(event.currentTarget);
    });
  }
}
