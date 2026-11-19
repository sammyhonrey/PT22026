/**
 * Toast notifications.
 *
 * Messages are inserted with textContent, never innerHTML, so server error
 * strings cannot inject markup.
 */

const DEFAULT_TIMEOUT = 5200;

export function createNotifications(container, { timeout = DEFAULT_TIMEOUT } = {}) {
  function dismiss(element) {
    if (!element.isConnected) {
      return;
    }

    element.remove();
  }

  function show(message, type = "info", duration = timeout) {
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.setAttribute("role", type === "error" ? "alert" : "status");

    const text = document.createElement("span");
    text.className = "toast-text";
    text.textContent = String(message);

    const close = document.createElement("button");
    close.type = "button";
    close.className = "toast-close";
    close.setAttribute("aria-label", "Dismiss notification");
    close.textContent = "\u00d7";
    close.addEventListener("click", () => dismiss(toast));

    toast.append(text, close);
    container.append(toast);

    if (duration > 0) {
      window.setTimeout(() => dismiss(toast), duration);
    }

    return toast;
  }

  return {
    info: (message) => show(message, "info"),
    success: (message) => show(message, "success"),
    error: (message) => show(message, "error", timeout + 2000),
    clear: () => container.replaceChildren(),
  };
}
