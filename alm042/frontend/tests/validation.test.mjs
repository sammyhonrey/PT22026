import test from "node:test";
import assert from "node:assert/strict";

import {
  countCharacters,
  counterState,
  validateText,
} from "../js/validation.js";

test("a normal sentence is valid and trimmed", () => {
  const result = validateText("  Hello world.  ");

  assert.equal(result.ok, true);
  assert.equal(result.text, "Hello world.");
});

test("empty and whitespace-only text is rejected", () => {
  for (const value of ["", "   ", "\n\t "]) {
    const result = validateText(value);

    assert.equal(result.ok, false);
    assert.match(result.error, /enter some text/i);
  }
});

test("text longer than the limit is rejected", () => {
  const result = validateText("a".repeat(401));

  assert.equal(result.ok, false);
  assert.match(result.error, /too long/i);
});

test("text exactly at the limit is accepted", () => {
  const result = validateText("a".repeat(400));

  assert.equal(result.ok, true);
  assert.equal(result.text.length, 400);
});

test("non-string input is rejected", () => {
  assert.equal(validateText(null).ok, false);
  assert.equal(validateText(42).ok, false);
});

test("the limit is configurable", () => {
  assert.equal(validateText("abc", 3).ok, true);
  assert.equal(validateText("abcd", 3).ok, false);
});

test("character counting includes spaces", () => {
  assert.equal(countCharacters("ab c"), 4);
  assert.equal(countCharacters(""), 0);
  assert.equal(countCharacters(null), 0);
});

test("counter state moves through normal, warning and over", () => {
  assert.equal(counterState(10, 400), "normal");
  assert.equal(counterState(380, 400), "warning");
  assert.equal(counterState(401, 400), "over");
});
