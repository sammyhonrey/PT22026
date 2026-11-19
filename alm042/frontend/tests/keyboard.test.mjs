import test from "node:test";
import assert from "node:assert/strict";

import { KeyAction, resolveKeyAction, resolveLongTextKey } from "../js/keyboard.js";

const enter = (overrides = {}) => ({
  key: "Enter",
  shiftKey: false,
  ctrlKey: false,
  metaKey: false,
  altKey: false,
  isComposing: false,
  ...overrides,
});

test("Enter with text requests generation", () => {
  assert.equal(resolveKeyAction(enter(), "Hello world"), KeyAction.GENERATE);
});

test("Shift+Enter inserts a newline", () => {
  assert.equal(
    resolveKeyAction(enter({ shiftKey: true }), "Hello world"),
    KeyAction.NEWLINE,
  );
});

test("Enter while composing an IME sequence is ignored", () => {
  assert.equal(
    resolveKeyAction(enter({ isComposing: true }), "Hello"),
    KeyAction.IGNORE,
  );
});

test("keyCode 229 (IME) is ignored", () => {
  assert.equal(
    resolveKeyAction(enter({ keyCode: 229 }), "Hello"),
    KeyAction.IGNORE,
  );
});

test("Enter on empty or whitespace-only text does not generate", () => {
  assert.equal(resolveKeyAction(enter(), ""), KeyAction.IGNORE);
  assert.equal(resolveKeyAction(enter(), "    "), KeyAction.IGNORE);
  assert.equal(resolveKeyAction(enter(), "\n\t "), KeyAction.IGNORE);
});

test("other shortcut combinations are not hijacked", () => {
  for (const modifier of ["ctrlKey", "metaKey", "altKey"]) {
    assert.equal(
      resolveKeyAction(enter({ [modifier]: true }), "Hello"),
      KeyAction.IGNORE,
      modifier,
    );
  }
});

test("non-Enter keys are ignored", () => {
  for (const key of ["a", "Escape", "Tab", "NumpadEnter".slice(0, 1)]) {
    assert.equal(resolveKeyAction(enter({ key }), "Hello"), KeyAction.IGNORE);
  }
});

test("missing event is ignored", () => {
  assert.equal(resolveKeyAction(null, "Hello"), KeyAction.IGNORE);
});


// --- Long-text mode ------------------------------------------------------

const longEnter = (overrides = {}) => ({
  key: "Enter",
  shiftKey: false,
  ctrlKey: false,
  metaKey: false,
  altKey: false,
  isComposing: false,
  ...overrides,
});

test("long mode: Enter inserts a newline", () => {
  assert.equal(resolveLongTextKey(longEnter(), "Hello"), KeyAction.NEWLINE);
});

test("long mode: Shift+Enter inserts a newline", () => {
  assert.equal(
    resolveLongTextKey(longEnter({ shiftKey: true }), "Hello"),
    KeyAction.NEWLINE,
  );
});

test("long mode: Ctrl+Enter submits", () => {
  assert.equal(
    resolveLongTextKey(longEnter({ ctrlKey: true }), "Hello"),
    KeyAction.GENERATE,
  );
});

test("long mode: Cmd+Enter submits", () => {
  assert.equal(
    resolveLongTextKey(longEnter({ metaKey: true }), "Hello"),
    KeyAction.GENERATE,
  );
});

test("long mode: Ctrl+Enter with empty text does nothing", () => {
  assert.equal(
    resolveLongTextKey(longEnter({ ctrlKey: true }), "   "),
    KeyAction.IGNORE,
  );
});

test("long mode: IME composition is ignored", () => {
  assert.equal(
    resolveLongTextKey(longEnter({ isComposing: true }), "Hello"),
    KeyAction.IGNORE,
  );
});

test("long mode: Ctrl+Shift+Enter is not a submit", () => {
  assert.equal(
    resolveLongTextKey(longEnter({ ctrlKey: true, shiftKey: true }), "Hello"),
    KeyAction.IGNORE,
  );
});

test("long mode: other keys are ignored", () => {
  assert.equal(resolveLongTextKey(longEnter({ key: "a" }), "Hello"), KeyAction.IGNORE);
});
