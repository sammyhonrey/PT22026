async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(path, {
      ...options,
      headers: {
        Accept: "application/json",
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...options.headers,
      },
    });
  } catch {
    throw new Error("Cannot connect to SmartFeel AI. Check that the server is running.");
  }

  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const detail = typeof payload === "object" ? payload?.detail : payload;
    throw new Error(Array.isArray(detail) ? detail.map((item) => item.msg).join(", ") : detail || "The request failed.");
  }
  return payload;
}

function queryString(params = {}) {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") query.set(key, value);
  }
  const encoded = query.toString();
  return encoded ? `?${encoded}` : "";
}

export function analyzeText(text) {
  return request("/api/analyze", {
    method: "POST",
    body: JSON.stringify({ text }),
  });
}

export function analyzeReview(productName, reviewText, customerName = "", isDemo = false) {
  return request("/api/review", {
    method: "POST",
    body: JSON.stringify({
      product_name: productName,
      review_text: reviewText,
      customer_name: customerName,
      is_demo: isDemo,
    }),
  });
}

export function analyzeSocial(comments, isDemo = false) {
  return request("/api/social", {
    method: "POST",
    body: JSON.stringify({ comments, is_demo: isDemo }),
  });
}

export function getDashboard(params = {}) {
  return request(`/api/dashboard${queryString(params)}`);
}

export function getHistory(filters = {}) {
  return request(`/api/history${queryString(filters)}`);
}

export async function exportHistory(filters = {}) {
  let response;
  try {
    response = await fetch(`/api/history/export.csv${queryString(filters)}`, {
      headers: { Accept: "text/csv" },
    });
  } catch {
    throw new Error("Cannot connect to SmartFeel AI. Check that the server is running.");
  }
  if (!response.ok) {
    let message = "Could not export history.";
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch {
      // Keep the readable export error if the server response is not JSON.
    }
    throw new Error(message);
  }
  return response.blob();
}

export function clearHistory() {
  return request("/api/history", { method: "DELETE" });
}
