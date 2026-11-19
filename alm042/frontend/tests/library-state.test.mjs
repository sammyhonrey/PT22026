import test from "node:test";
import assert from "node:assert/strict";

import {
  SOURCE_OPTIONS,
  buildLibraryQuery,
  canGoNext,
  canGoPrevious,
  describeRecord,
  formatBytes,
  formatDate,
  formatDuration,
  isEmptyLibrary,
  paginationSummary,
  truncateText,
  validateLibraryName,
} from "../js/library-state.js";

test("query includes search, sort, filters and pagination", () => {
  const query = buildLibraryQuery({
    q: "  hello  ",
    sort: "name",
    engine: "kokoro",
    source: "batch",
    page: 2,
    limit: 10,
  });

  assert.match(query, /q=hello/);
  assert.match(query, /sort=name/);
  assert.match(query, /engine=kokoro/);
  assert.match(query, /source=batch/);
  assert.match(query, /page=2/);
  assert.match(query, /limit=10/);
});

test("query omits empty filters and clamps values", () => {
  const query = buildLibraryQuery({ sort: "bogus", page: -5, limit: 999 });

  assert.doesNotMatch(query, /q=/);
  assert.match(query, /sort=newest/);
  assert.match(query, /page=1/);
  assert.match(query, /limit=100/);
});

test("invalid source values are ignored", () => {
  const query = buildLibraryQuery({ source: "not_a_source" });

  assert.doesNotMatch(query, /source=/);
  assert.deepEqual(SOURCE_OPTIONS, ["studio", "long_text", "batch"]);
});

test("names are validated", () => {
  assert.equal(validateLibraryName("  Welcome  ").name, "Welcome");
  assert.equal(validateLibraryName("").ok, false);
  assert.equal(validateLibraryName("   ").ok, false);
  assert.equal(validateLibraryName("x".repeat(81)).ok, false);
});

test("pagination summary shows page and total", () => {
  assert.equal(
    paginationSummary({ page: 2, pages: 3, total: 25 }),
    "Page 2 of 3 · 25 recordings",
  );
  assert.equal(paginationSummary({ page: 1, pages: 1, total: 1 }), "Page 1 of 1 · 1 recording");
  assert.equal(paginationSummary(null), "No recordings");
});

test("pager bounds are respected", () => {
  assert.equal(canGoPrevious(1), false);
  assert.equal(canGoPrevious(2), true);
  assert.equal(canGoNext(1, 3), true);
  assert.equal(canGoNext(3, 3), false);
});

test("empty library detection", () => {
  assert.equal(isEmptyLibrary({ records: [] }), true);
  assert.equal(isEmptyLibrary({ records: [{}] }), false);
  assert.equal(isEmptyLibrary(null), false);
});

test("bytes are formatted", () => {
  assert.equal(formatBytes(512), "512 B");
  assert.equal(formatBytes(2048), "2.0 KB");
  assert.equal(formatBytes(2 * 1024 * 1024), "2.0 MB");
  assert.equal(formatBytes(0), "—");
});

test("durations are formatted", () => {
  assert.equal(formatDuration(2.5), "2.50 s");
  assert.equal(formatDuration(75), "1:15");
  assert.equal(formatDuration(-1), "—");
});

test("invalid dates degrade gracefully", () => {
  assert.equal(formatDate("not-a-date"), "—");
  assert.equal(formatDate(""), "—");
  assert.ok(formatDate("2024-05-01T10:00:00Z").length > 0);
});

test("long text is truncated with an ellipsis", () => {
  const text = "word ".repeat(100);

  const short = truncateText(text, 40);

  assert.ok(short.length <= 40);
  assert.ok(short.endsWith("…"));
  assert.equal(truncateText("  a   b  ", 40), "a b");
});

test("record description includes engine, duration and source", () => {
  const description = describeRecord({
    created_at: "2024-05-01T10:00:00Z",
    engine: "kokoro",
    voice: "af_heart",
    speed: 1.25,
    duration_seconds: 2,
    file_size: 1024,
    source_type: "studio",
  });

  assert.match(description, /kokoro/);
  assert.match(description, /1\.25×/);
  assert.match(description, /Speech Studio/);
});
