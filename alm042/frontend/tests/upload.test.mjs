import test from "node:test";
import assert from "node:assert/strict";

import {
  MAX_FILE_BYTES,
  MAX_TOTAL_BYTES,
  MAX_UPLOAD_FILES,
  isTxtFile,
  toBase64,
  validateFiles,
} from "../js/upload.js";

const file = (name, size = 10) => ({ name, size });

test("only .txt files are recognised", () => {
  assert.equal(isTxtFile("notes.txt"), true);
  assert.equal(isTxtFile("NOTES.TXT"), true);
  assert.equal(isTxtFile("notes.md"), false);
  assert.equal(isTxtFile(""), false);
  assert.equal(isTxtFile(null), false);
});

test("a valid selection passes", () => {
  const result = validateFiles([file("a.txt"), file("b.txt", 500)]);

  assert.equal(result.ok, true);
  assert.equal(result.files.length, 2);
});

test("an empty selection is rejected", () => {
  assert.equal(validateFiles([]).ok, false);
  assert.equal(validateFiles(null).ok, false);
});

test("non-txt files are rejected", () => {
  const result = validateFiles([file("a.txt"), file("b.pdf")]);

  assert.equal(result.ok, false);
  assert.match(result.error, /not a \.txt file/i);
});

test("more than the maximum number of files is rejected", () => {
  const files = Array.from({ length: MAX_UPLOAD_FILES + 1 }, (_, i) =>
    file(`f${i}.txt`),
  );

  assert.equal(validateFiles(files).ok, false);
});

test("a file larger than 1 MB is rejected", () => {
  const result = validateFiles([file("big.txt", MAX_FILE_BYTES + 1)]);

  assert.equal(result.ok, false);
  assert.match(result.error, /larger than 1 MB/i);
});

test("a total larger than 5 MB is rejected", () => {
  const files = [
    file("a.txt", 3 * 1024 * 1024),
    file("b.txt", 3 * 1024 * 1024),
  ];

  assert.equal(validateFiles(files).ok, false);
  assert.ok(MAX_TOTAL_BYTES < 3 * 1024 * 1024 * 2);
});

test("base64 encoding round-trips ASCII and UTF-8", () => {
  const text = "Héllo wörld";
  const bytes = new TextEncoder().encode(text);
  const encoded = toBase64(bytes.buffer);
  const decoded = new TextDecoder().decode(
    Uint8Array.from(atob(encoded), (char) => char.charCodeAt(0)),
  );

  assert.equal(decoded, text);
});
