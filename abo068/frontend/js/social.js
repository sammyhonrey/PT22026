import { analyzeSocial } from "./api.js";
import { setBusy, showError } from "./ui.js";

const sampleComments = [
  "Sofia Martinez: I love how quickly the new dashboard turns customer feedback into clear insights.",
  "Daniel Brooks: Setup was easy, but the latest update keeps logging me out during reviews.",
  "Aisha Khan: Support helped me fix an import issue the same day. Great experience.",
  "Ethan Cole: The reports are useful, though I wish there were more export options.",
  "Maya Chen: Sentiment tracking is accurate and has helped our team respond to customers faster.",
  "Noah Rivera: My billing question has been waiting for a reply for three days.",
];

export function initSocial() {
  const input = document.querySelector("#social-comments");
  const analyzeButton = document.querySelector("#social-analyze");
  const resultsPanel = document.querySelector("#social-results");
  const demoCheckbox = document.querySelector("#social-is-demo");
  const countComments = () => input.value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  input.value = sampleComments.join("\n");

  function updateCount() {
    const comments = countComments();
    document.querySelector("#social-char-count").textContent = `${input.value.length.toLocaleString()} characters`;
    document.querySelector("#social-line-count").textContent = `${comments.length} comment${comments.length === 1 ? "" : "s"}`;
  }

  function renderSummary(data) {
    const summary = document.querySelector("#social-summary");
    summary.replaceChildren();
    const overallCard = document.createElement("article");
    overallCard.className = "social-summary-card";
    overallCard.append(Object.assign(document.createElement("small"), {
      textContent: "MOST COMMON SENTIMENT",
    }));
    const overall = document.createElement("strong");
    overall.textContent = data.overall_sentiment;
    const sentimentColor = {
      Positive: "var(--green)",
      Neutral: "var(--amber)",
      Negative: "var(--red)",
    };
    overall.style.color = sentimentColor[data.overall_sentiment] || "var(--ink)";
    overallCard.append(overall);
    const total = document.createElement("span");
    total.className = "summary-count";
    total.textContent = `${data.total_comments_analyzed} comments analyzed`;
    overallCard.append(total);
    summary.append(overallCard);

    for (const label of ["Positive", "Neutral", "Negative"]) {
      const distribution = data.distribution[label];
      const card = document.createElement("article");
      card.className = `social-summary-card ${label.toLowerCase()}-metric`;
      card.append(Object.assign(document.createElement("small"), { textContent: `${label.toUpperCase()} SENTIMENT` }));
      card.append(Object.assign(document.createElement("strong"), { textContent: `${distribution.count} comments` }));
      card.append(Object.assign(document.createElement("span"), {
        className: "summary-count",
        textContent: `${Number(distribution.percentage).toFixed(1)}% of batch`,
      }));
      const bar = document.createElement("div");
      bar.className = "probability-bar";
      const fill = document.createElement("i");
      fill.className = label.toLowerCase();
      fill.style.width = `${Math.max(0, Math.min(100, distribution.percentage))}%`;
      bar.append(fill);
      card.append(bar);
      summary.append(card);
    }

    const list = document.querySelector("#social-comment-list");
    list.replaceChildren();
    data.results.forEach((result, index) => {
      const row = document.createElement("div");
      row.className = "comment-result";
      row.append(Object.assign(document.createElement("span"), {
        className: "comment-index",
        textContent: String(index + 1).padStart(2, "0"),
      }));
      const content = document.createElement("div");
      content.append(Object.assign(document.createElement("p"), { textContent: result.text }));
      content.append(Object.assign(document.createElement("small"), {
        textContent: `Confidence: ${Number(result.confidence).toFixed(1)}%`,
      }));
      row.append(content);
      const badge = document.createElement("span");
      badge.className = `sentiment-badge ${result.sentiment.toLowerCase()}`;
      badge.textContent = result.sentiment;
      row.append(badge);
      if (result.is_demo) {
        row.append(Object.assign(document.createElement("span"), {
          className: "demo-tag",
          textContent: "DEMO",
        }));
      }
      list.append(row);
    });
    document.querySelector("#social-total-chip").textContent = `${data.total_comments_analyzed} analyzed`;
    resultsPanel.hidden = false;
  }

  input.addEventListener("input", updateCount);
  document.querySelector("#social-sample").addEventListener("click", () => {
    input.value = sampleComments.join("\n");
    demoCheckbox.checked = true;
    updateCount();
    input.focus();
  });
  document.querySelector("#social-clear").addEventListener("click", () => {
    input.value = "";
    demoCheckbox.checked = false;
    updateCount();
    showError("social-error");
    resultsPanel.hidden = true;
    input.focus();
  });

  analyzeButton.addEventListener("click", async () => {
    showError("social-error");
    const comments = countComments();
    if (comments.length < 2) {
      showError("social-error", "Enter at least 2 non-empty comments.");
      return;
    }
    if (comments.length > 30) {
      showError("social-error", "A batch can contain no more than 30 comments.");
      return;
    }

    setBusy(analyzeButton, true, "Analyzing comments…");
    try {
      const data = await analyzeSocial(comments, demoCheckbox.checked);
      renderSummary(data);
    } catch (error) {
      showError("social-error", error.message);
    } finally {
      setBusy(analyzeButton, false);
    }
  });
  updateCount();
}
