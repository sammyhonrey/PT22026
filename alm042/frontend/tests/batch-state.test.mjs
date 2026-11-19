import test from "node:test";
import assert from "node:assert/strict";

import {
  MAX_ENTRIES,
  addEntry,
  createEntry,
  removeEntry,
  updateEntry,
  validateEntries,
} from "../js/batch-state.js";

test("a new entry has a unique id and empty fields", () => {
  const first = createEntry();
  const second = createEntry();

  assert.ok(first.id);
  assert.notEqual(first.id, second.id);
  assert.equal(first.name, "");
  assert.equal(first.text, "");
});

test("entries can be added", () => {
  const entries = addEntry([], createEntry({ name: "One", text: "Hello." }));

  assert.equal(entries.length, 1);
  assert.equal(entries[0].name, "One");
});

test("entries cannot exceed the maximum", () => {
  let entries = [];

  for (let index = 0; index < MAX_ENTRIES + 5; index += 1) {
    entries = addEntry(entries, createEntry());
  }

  assert.equal(entries.length, MAX_ENTRIES);
});

test("entries can be removed by id", () => {
  const first = createEntry({ text: "First" });
  const second = createEntry({ text: "Second" });

  const remaining = removeEntry([first, second], first.id);

  assert.equal(remaining.length, 1);
  assert.equal(remaining[0].id, second.id);
});

test("entries can be updated", () => {
  const entry = createEntry();
  const updated = updateEntry([entry], entry.id, { text: "Changed" });

  assert.equal(updated[0].text, "Changed");
});

test("validation rejects an empty list", () => {
  assert.equal(validateEntries([]).ok, false);
});

test("validation rejects empty text", () => {
  const result = validateEntries([createEntry({ text: "   " })]);

  assert.equal(result.ok, false);
  assert.match(result.error, /empty/i);
});

test("validation rejects overly long text", () => {
  const result = validateEntries([createEntry({ text: "a".repeat(20001) })]);

  assert.equal(result.ok, false);
  assert.match(result.error, /too long/i);
});

test("validation trims text and defaults the name", () => {
  const result = validateEntries([
    createEntry({ name: "  ", text: "  Hello world.  " }),
  ]);

  assert.equal(result.ok, true);
  assert.equal(result.items[0].text, "Hello world.");
  assert.equal(result.items[0].name, "Entry 1");
});
