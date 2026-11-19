/**
 * Batch entry state, kept pure so it can be tested without a DOM.
 */

export const MAX_ENTRIES = 20;
export const MAX_ENTRY_CHARS = 20000;

let counter = 0;

export function createEntry({ name = "", text = "" } = {}) {
  counter += 1;

  return {
    id: `entry-${Date.now()}-${counter}`,
    name: String(name || "").slice(0, 80),
    text: String(text || ""),
  };
}

export function addEntry(entries, entry) {
  if (entries.length >= MAX_ENTRIES) {
    return entries;
  }

  return [...entries, entry];
}

export function removeEntry(entries, id) {
  return entries.filter((entry) => entry.id !== id);
}

export function updateEntry(entries, id, patch) {
  return entries.map((entry) =>
    entry.id === id ? { ...entry, ...patch } : entry,
  );
}

export function validateEntries(entries) {
  if (!Array.isArray(entries) || entries.length === 0) {
    return { ok: false, error: "Add at least one text entry." };
  }

  if (entries.length > MAX_ENTRIES) {
    return {
      ok: false,
      error: `Too many entries: ${entries.length}. The limit is ${MAX_ENTRIES}.`,
    };
  }

  const items = [];

  for (let index = 0; index < entries.length; index += 1) {
    const entry = entries[index];
    const text = String(entry.text || "").trim();

    if (!text) {
      return {
        ok: false,
        error: `Entry ${index + 1} is empty.`,
      };
    }

    if (text.length > MAX_ENTRY_CHARS) {
      return {
        ok: false,
        error: `Entry ${index + 1} is too long (${text.length} characters).`,
      };
    }

    items.push({
      name: String(entry.name || "").trim() || `Entry ${index + 1}`,
      text,
    });
  }

  return { ok: true, items };
}
