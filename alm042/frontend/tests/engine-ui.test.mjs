import test from "node:test";
import assert from "node:assert/strict";

import {
  describeSpeed,
  describeVoice,
  formatSpeed,
  speedControlVisible,
  speedForRequest,
  voiceControlsVisible,
  voiceForRequest,
} from "../js/engine-ui.js";

const LABELS = { af_heart: "American female — warm and clear" };

test("the voice selector is visible only for Kokoro", () => {
  assert.equal(voiceControlsVisible("kokoro"), true);
  assert.equal(voiceControlsVisible("voiceforge"), false);
  assert.equal(voiceControlsVisible(""), false);
});

test("the voice is sent only for Kokoro", () => {
  assert.equal(voiceForRequest("kokoro", "af_heart"), "af_heart");
  assert.equal(voiceForRequest("kokoro", ""), null);
  assert.equal(voiceForRequest("voiceforge", "af_heart"), null);
});

test("voice metadata falls back gracefully", () => {
  assert.equal(
    describeVoice("kokoro", "af_heart", LABELS),
    "American female — warm and clear",
  );
  assert.equal(describeVoice("kokoro", "unknown", LABELS), "unknown");
  assert.equal(describeVoice("voiceforge", "af_heart", LABELS), "not applicable");
});

test("the speed control is visible only for Kokoro", () => {
  assert.equal(speedControlVisible("kokoro"), true);
  assert.equal(speedControlVisible("voiceforge"), false);
  assert.equal(speedControlVisible(""), false);
});

test("the speed is sent only for Kokoro", () => {
  assert.equal(speedForRequest("kokoro", 0.5), 0.5);
  assert.equal(speedForRequest("kokoro", "1.25"), 1.25);
  assert.equal(speedForRequest("voiceforge", 1.5), null);
});

test("invalid slider values are not sent", () => {
  assert.equal(speedForRequest("kokoro", "abc"), null);
  assert.equal(speedForRequest("kokoro", undefined), null);
  assert.equal(speedForRequest("kokoro", ""), null);
});

test("speed formatting is consistent", () => {
  assert.equal(formatSpeed(1), "1.00×");
  assert.equal(formatSpeed(0.5), "0.50×");
  assert.equal(formatSpeed(2), "2.00×");
  assert.equal(formatSpeed("bad"), "1.00×");
});

test("speed metadata reflects engine support", () => {
  assert.equal(describeSpeed("kokoro", 1.25), "1.25×");
  assert.equal(describeSpeed("voiceforge", 1.25), "not applicable");
  assert.equal(describeSpeed("voiceforge", null), "not applicable");
});
