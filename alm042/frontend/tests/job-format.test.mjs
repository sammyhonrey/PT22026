import test from "node:test";
import assert from "node:assert/strict";

import {
  completedItems,
  formatProgress,
  isTerminal,
  shouldPoll,
  statusLabel,
} from "../js/job-format.js";

test("terminal statuses are recognised", () => {
  assert.equal(isTerminal("completed"), true);
  assert.equal(isTerminal("failed"), true);
  assert.equal(isTerminal("queued"), false);
  assert.equal(isTerminal("running"), false);
});

test("statuses have readable labels", () => {
  assert.equal(statusLabel("queued"), "Queued");
  assert.equal(statusLabel("running"), "Processing");
  assert.equal(statusLabel("completed"), "Completed");
  assert.equal(statusLabel("failed"), "Failed");
  assert.equal(statusLabel("weird"), "weird");
});

test("progress uses real counts and mentions failures", () => {
  const job = { total: 5, completed: 2, failed: 1 };

  assert.equal(formatProgress(job), "3 of 5 items finished (1 failed)");
});

test("progress is empty without a job", () => {
  assert.equal(formatProgress(null), "");
});

test("polling continues while the job is active", () => {
  assert.equal(shouldPoll({ status: "queued" }), true);
  assert.equal(shouldPoll({ status: "running" }), true);
});

test("polling stops on terminal status", () => {
  assert.equal(shouldPoll({ status: "completed" }), false);
  assert.equal(shouldPoll({ status: "failed" }), false);
});

test("polling pauses while the tab is hidden", () => {
  assert.equal(shouldPoll({ status: "running" }, { hidden: true }), false);
});

test("only completed items with audio are returned", () => {
  const job = {
    items: [
      { status: "completed", download_url: "/a.wav" },
      { status: "failed", download_url: null },
      { status: "completed", download_url: null },
    ],
  };

  const results = completedItems(job);

  assert.equal(results.length, 1);
  assert.equal(completedItems(null).length, 0);
});
