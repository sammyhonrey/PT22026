import test from "node:test";
import assert from "node:assert/strict";

import {
  MAX_DOCUMENT_CHARS,
  applyPresetToControls,
  buildJobItemMp3Url,
  buildJobZipUrl,
  buildMp3Url,
  describeDocument,
  describePreset,
  isDocumentFile,
  validateDocumentFile,
  validateDocumentText,
  validatePresetDraft,
  validatePronunciationDraft,
} from "../js/advanced-state.js";

const capabilities = {
  engines: {
    kokoro: {
      id: "kokoro",
      name: "Kokoro",
      voices: [
        { id: "af_heart", name: "US female", default: true },
        { id: "am_michael", name: "US male" },
      ],
      speed: { supported: true, min: 0.5, max: 2, default: 1 },
    },
    voiceforge: {
      id: "voiceforge",
      name: "Custom",
      voices: [{ id: "default", name: "Single speaker", default: true }],
      speed: { supported: false, min: null, max: null, default: null },
    },
  },
};

const preset = {
  name: "Fast",
  engine: "kokoro",
  voice: "am_michael",
  speed: 1.25,
};

test("document extensions are validated", () => {
  assert.equal(isDocumentFile("a.pdf"), true);
  assert.equal(isDocumentFile("a.DOCX"), true);
  assert.equal(isDocumentFile("a.txt"), false);
});

test("document size and type are validated", () => {
  assert.equal(validateDocumentFile({ name: "a.pdf", size: 100 }).ok, true);
  assert.equal(validateDocumentFile({ name: "a.exe", size: 10 }).ok, false);
  assert.equal(
    validateDocumentFile({ name: "a.pdf", size: 11 * 1024 * 1024 }).ok,
    false,
  );
});

test("document description summarises metadata", () => {
  const text = describeDocument({
    filename: "report.pdf",
    format: "pdf",
    pages: 3,
    characters: 1200,
  });

  assert.match(text, /report\.pdf/);
  assert.match(text, /PDF/);
  assert.match(text, /3 pages/);
  assert.match(text, /1200 characters/);
});

test("document text validation enforces the long-text limit", () => {
  assert.equal(validateDocumentText("Hello.").ok, true);
  assert.equal(validateDocumentText("   ").ok, false);
  assert.equal(
    validateDocumentText("a".repeat(MAX_DOCUMENT_CHARS + 1)).ok,
    false,
  );
});

test("pronunciation drafts are validated", () => {
  assert.equal(
    validatePronunciationDraft({ word: "Test", pronunciation: "T-est" }).ok,
    true,
  );
  assert.equal(
    validatePronunciationDraft({ word: "", pronunciation: "x" }).ok,
    false,
  );
  assert.equal(
    validatePronunciationDraft({ word: "x", pronunciation: "  " }).ok,
    false,
  );
});

test("presets are validated against capabilities", () => {
  assert.equal(validatePresetDraft(preset, capabilities).ok, true);
  assert.equal(
    validatePresetDraft({ ...preset, engine: "nope" }, capabilities).ok,
    false,
  );
  assert.equal(
    validatePresetDraft({ ...preset, voice: "nope" }, capabilities).ok,
    false,
  );
  assert.equal(
    validatePresetDraft({ ...preset, speed: 9 }, capabilities).ok,
    false,
  );
  assert.equal(validatePresetDraft({ ...preset, name: "" }, capabilities).ok, false);
});

test("custom-engine presets never store speed or voice", () => {
  const result = validatePresetDraft(
    { name: "Custom", engine: "voiceforge", voice: "af_heart", speed: 2 },
    capabilities,
  );

  assert.equal(result.ok, true);
  assert.equal(result.preset.voice, null);
  assert.equal(result.preset.speed, null);
});

test("applying a preset clamps to engine support", () => {
  const kokoro = applyPresetToControls(preset, capabilities);

  assert.equal(kokoro.engine, "kokoro");
  assert.equal(kokoro.voice, "am_michael");
  assert.equal(kokoro.speed, 1.25);

  const custom = applyPresetToControls(
    { engine: "voiceforge", voice: "af_heart", speed: 1 },
    capabilities,
  );

  assert.equal(custom.voice, null);
  assert.equal(custom.speed, null);
  assert.equal(applyPresetToControls({ engine: "nope" }, capabilities), null);
});

test("preset description uses readable names", () => {
  assert.match(describePreset(preset, capabilities), /Kokoro/);
  assert.match(describePreset(preset, capabilities), /1\.25×/);
});

test("MP3 URLs are safe and bitrate-limited", () => {
  assert.equal(buildMp3Url("abc", 192), "/api/library/abc/mp3?bitrate=192");
  assert.equal(buildMp3Url("abc", 999), "/api/library/abc/mp3?bitrate=192");
  assert.match(buildMp3Url("a b", 128), /a%20b/);
});

test("job archive and item MP3 URLs are built", () => {
  assert.match(buildJobZipUrl("job1", "mp3"), /format=mp3/);
  assert.match(buildJobZipUrl("job1", "wav"), /format=wav/);
  assert.match(buildJobItemMp3Url("job1", "item1"), /format=mp3&bitrate=192/);
});
