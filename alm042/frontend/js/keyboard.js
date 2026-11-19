/**
 * Keyboard behaviour for the speech composer.
 *
 * The rule set is pure and side-effect free so it can be unit-tested with
 * Node, and so the UI handler stays a thin wrapper around it:
 *
 *   Enter            -> generate
 *   Shift + Enter    -> insert a new line (the browser default)
 *   Enter while IME composing -> ignore
 *   Enter on empty text       -> ignore
 */

export const KeyAction = Object.freeze({
  GENERATE: "generate",
  NEWLINE: "newline",
  IGNORE: "ignore",
});

/**
 * @param {{key: string, shiftKey?: boolean, ctrlKey?: boolean,
 *          metaKey?: boolean, altKey?: boolean, isComposing?: boolean,
 *          keyCode?: number}} event
 * @param {string} text current textarea value
 * @returns {"generate" | "newline" | "ignore"}
 */
export function resolveKeyAction(event, text) {
  if (!event || event.key !== "Enter") {
    return KeyAction.IGNORE;
  }

  // IME composition (e.g. Japanese/Chinese input) must never trigger send.
  if (event.isComposing || event.keyCode === 229) {
    return KeyAction.IGNORE;
  }

  // Shift+Enter is the browser's newline; let it happen.
  if (event.shiftKey) {
    return KeyAction.NEWLINE;
  }

  // Other shortcuts (Ctrl/Cmd/Alt + Enter) are not ours to intercept.
  if (event.ctrlKey || event.metaKey || event.altKey) {
    return KeyAction.IGNORE;
  }

  if (typeof text !== "string" || text.trim() === "") {
    return KeyAction.IGNORE;
  }

  return KeyAction.GENERATE;
}

/**
 * Keyboard behaviour for the long-text composer, where Enter must insert a
 * newline and Ctrl/Cmd + Enter submits.
 *
 * @param {{key: string, shiftKey?: boolean, ctrlKey?: boolean,
 *          metaKey?: boolean, altKey?: boolean, isComposing?: boolean,
 *          keyCode?: number}} event
 * @param {string} text
 * @returns {"generate" | "newline" | "ignore"}
 */
export function resolveLongTextKey(event, text) {
  if (!event || event.key !== "Enter") {
    return KeyAction.IGNORE;
  }

  if (event.isComposing || event.keyCode === 229) {
    return KeyAction.IGNORE;
  }

  const submits = Boolean(event.ctrlKey || event.metaKey);

  if (!submits) {
    // Plain Enter (and Shift+Enter) stay as newlines.
    return KeyAction.NEWLINE;
  }

  if (event.shiftKey || event.altKey) {
    return KeyAction.IGNORE;
  }

  if (typeof text !== "string" || text.trim() === "") {
    return KeyAction.IGNORE;
  }

  return KeyAction.GENERATE;
}
