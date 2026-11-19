/**
 * Text validation shared by the Studio page.
 *
 * These rules mirror the backend so the user gets an immediate answer, but
 * the backend remains the source of truth and re-checks everything.
 */

export const DEFAULT_MAX_LENGTH = 400;

/**
 * @param {unknown} raw
 * @param {number} maxLength
 * @returns {{ok: true, text: string} | {ok: false, error: string}}
 */
export function validateText(raw, maxLength = DEFAULT_MAX_LENGTH) {
  if (typeof raw !== "string") {
    return { ok: false, error: "Please enter text as a string." };
  }

  const text = raw.trim();

  if (!text) {
    return { ok: false, error: "Enter some text before generating speech." };
  }

  if (text.length > maxLength) {
    return {
      ok: false,
      error: `Text is too long (${text.length}/${maxLength} characters).`,
    };
  }

  return { ok: true, text };
}

/** Raw character count for the counter (spaces included). */
export function countCharacters(raw) {
  return typeof raw === "string" ? raw.length : 0;
}

/**
 * Counter state used by the UI: "normal", "warning" (near the limit) or
 * "over" (past it).
 */
export function counterState(count, maxLength = DEFAULT_MAX_LENGTH) {
  if (count > maxLength) {
    return "over";
  }

  if (count > maxLength * 0.9) {
    return "warning";
  }

  return "normal";
}
