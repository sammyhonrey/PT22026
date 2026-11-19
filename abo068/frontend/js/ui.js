const classes = ["Positive", "Neutral", "Negative"];

export function setBusy(button, busy, busyText = "Analyzing…") {
  if (busy) {
    button.dataset.originalText = button.textContent;
    button.textContent = busyText;
  } else if (button.dataset.originalText) {
    button.textContent = button.dataset.originalText;
    delete button.dataset.originalText;
  }
  button.disabled = busy;
  button.setAttribute("aria-busy", String(busy));
}

export function showError(id, message = "") {
  document.querySelector(`#${id}`).textContent = message;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

export function renderPrediction(
  container,
  result,
  productName = "",
  customerName = "",
  isDemo = false,
) {
  const content = element("div", "result-content");
  if (productName || customerName || isDemo) {
    const product = element("div", "result-product");
    if (isDemo) product.append(element("span", "demo-tag", "DEMO SAMPLE"));
    product.append(element("small", "", "PRODUCT / SERVICE"));
    product.append(element("strong", "", productName || "—"));
    if (customerName) {
      product.append(element("small", "review-customer", `Fictional customer: ${customerName}`));
    }
    content.append(product);
  }

  const sentimentClass = (result.sentiment || "").toLowerCase();
  const top = element("div", `result-top ${sentimentClass}`);
  const classification = element("div", "");
  classification.append(element("div", "result-kicker", "CLASSIFICATION"));
  classification.append(
    element("div", `result-sentiment ${sentimentClass}`, result.sentiment || "Unknown"),
  );
  top.append(classification);
  top.append(element("span", `result-confidence ${sentimentClass}`, `${Number(result.confidence).toFixed(1)}% confidence`));
  content.append(top);

  const heading = element("div", "probability-heading");
  heading.append(element("span", "", "Class probabilities"));
  heading.append(element("small", "", "Calibrated model scores"));
  content.append(heading);

  for (const label of classes) {
    const value = Number(result.scores?.[label] ?? 0);
    const row = element("div", "probability-row");
    const labelRow = element("div", "probability-label");
    labelRow.append(element("span", "", label));
    labelRow.append(element("strong", "", `${value.toFixed(2)}%`));
    const track = element("div", "probability-bar");
    const fill = element("i", label.toLowerCase());
    fill.style.width = `${Math.max(0, Math.min(100, value))}%`;
    track.append(fill);
    row.append(labelRow, track);
    content.append(row);
  }

  content.append(
    element("div", "result-note", "Confidence is the highest predicted class probability. It is a model score, not a guarantee of correctness."),
  );
  container.replaceChildren(content);
}

export function setPlaceholder(container, title, description, icon = "◉") {
  const placeholder = element("div", "result-placeholder");
  placeholder.append(element("span", "placeholder-icon", icon));
  placeholder.append(element("strong", "", title));
  placeholder.append(element("span", "", description));
  container.replaceChildren(placeholder);
}

export function formatTime(value) {
  if (!value) return "—";
  const formattedValue = String(value);
  const date = new Date(formattedValue.replace(" ", "T"));
  return Number.isNaN(date.getTime())
    ? formattedValue
    : new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(date);
}
