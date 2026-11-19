import { analyzeText } from "./api.js";
import { renderPrediction, setBusy, setPlaceholder, showError } from "./ui.js";

const samples = {
  positive: "I absolutely love this product! The quality is excellent and the service was amazing.",
  neutral: "The product arrived yesterday. It has the features described on the website.",
  negative: "I am very disappointed with this product. The quality is terrible and customer service was unhelpful.",
};

export function initPlayground() {
  const text = document.querySelector("#playground-text");
  const button = document.querySelector("#playground-analyze");
  const result = document.querySelector("#playground-result");
  const counter = document.querySelector("#playground-char-count");

  const updateCount = () => {
    counter.textContent = `${text.value.length.toLocaleString()} characters`;
  };
  text.addEventListener("input", updateCount);

  document.querySelectorAll("[data-sample]").forEach((sampleButton) => {
    sampleButton.addEventListener("click", () => {
      text.value = samples[sampleButton.dataset.sample];
      updateCount();
      text.focus();
    });
  });

  async function submit() {
    showError("playground-error");
    const value = text.value.trim();
    if (!value) {
      showError("playground-error", "Enter some text before analyzing.");
      text.focus();
      return;
    }

    setBusy(button, true, "Analyzing sentiment…");
    try {
      const prediction = await analyzeText(value);
      renderPrediction(result, prediction);
    } catch (error) {
      showError("playground-error", error.message);
    } finally {
      setBusy(button, false);
    }
  }

  button.addEventListener("click", submit);
  text.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") submit();
  });
  document.querySelector("#playground-clear").addEventListener("click", () => {
    text.value = "";
    updateCount();
    showError("playground-error");
    setPlaceholder(result, "Ready for analysis", "Submit text to see its sentiment and class probabilities.");
    text.focus();
  });
}
