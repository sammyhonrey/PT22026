import test from "node:test";
import assert from "node:assert/strict";

import { createGenerationGate, createStore } from "../js/state.js";

test("the generation gate allows only one request at a time", () => {
  const gate = createGenerationGate();

  assert.equal(gate.isRunning, false);
  assert.equal(gate.tryStart(), true);
  assert.equal(gate.isRunning, true);

  // A second Enter press or button click while running must be ignored.
  assert.equal(gate.tryStart(), false);
  assert.equal(gate.tryStart(), false);
});

test("the gate reopens after the request finishes", () => {
  const gate = createGenerationGate();

  assert.equal(gate.tryStart(), true);
  gate.finish();

  assert.equal(gate.isRunning, false);
  assert.equal(gate.tryStart(), true);
});

test("finishing without starting is harmless", () => {
  const gate = createGenerationGate();

  gate.finish();

  assert.equal(gate.tryStart(), true);
});

test("the store notifies subscribers on change", () => {
  const store = createStore({ page: "studio", capabilities: null });
  const seen = [];

  const unsubscribe = store.subscribe((state) => seen.push(state.page));

  store.set({ page: "library" });
  unsubscribe();
  store.set({ page: "settings" });

  assert.deepEqual(seen, ["studio", "library"]);
  assert.equal(store.get().page, "settings");
});

test("the store merges patches instead of replacing state", () => {
  const store = createStore({ a: 1, b: 2 });

  store.set({ b: 3 });

  assert.deepEqual(store.get(), { a: 1, b: 3 });
});
