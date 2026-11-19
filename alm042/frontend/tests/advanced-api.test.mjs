import test from "node:test";
import assert from "node:assert/strict";

import { ApiError } from "../js/api.js";
import {
  createPreset,
  deletePreset,
  deletePronunciation,
  extractDocument,
  listPresets,
  listPronunciation,
  updatePreset,
  updatePronunciation,
} from "../js/advanced-api.js";

function mockFetch(body, { ok = true, status = 200 } = {}) {
  const calls = [];
  const original = globalThis.fetch;

  globalThis.fetch = async (url, options = {}) => {
    calls.push({ url, options });
    return { ok, status, json: async () => body };
  };

  return {
    calls,
    restore() {
      globalThis.fetch = original;
    },
  };
}

test("document upload posts name and base64", async () => {
  const mock = mockFetch({
    ok: true,
    document: { filename: "a.pdf", text: "Hello", characters: 5 },
  });

  try {
    const file = {
      name: "a.pdf",
      arrayBuffer: async () => new TextEncoder().encode("Hello").buffer,
    };

    const document = await extractDocument(file);

    assert.equal(document.text, "Hello");
    assert.equal(mock.calls[0].url, "/api/documents/extract");

    const body = JSON.parse(mock.calls[0].options.body);
    assert.equal(body.name, "a.pdf");
    assert.equal(atob(body.data_base64), "Hello");
  } finally {
    mock.restore();
  }
});

test("pronunciation list encodes the search query", async () => {
  const mock = mockFetch({ ok: true, entries: [] });

  try {
    await listPronunciation("ab dool");
    assert.equal(
      mock.calls[0].url,
      "/api/pronunciation?q=ab%20dool",
    );
  } finally {
    mock.restore();
  }
});

test("pronunciation updates use PATCH", async () => {
  const mock = mockFetch({ ok: true, entry: { id: "e1" } });

  try {
    await updatePronunciation("e1", {
      word: "Test",
      pronunciation: "T-est",
      description: "",
    });

    assert.equal(mock.calls[0].options.method, "PATCH");
    assert.match(mock.calls[0].url, /\/api\/pronunciation\/e1$/);
  } finally {
    mock.restore();
  }
});

test("pronunciation deletion uses DELETE", async () => {
  const mock = mockFetch({ ok: true, deleted: { id: "e1" } });

  try {
    await deletePronunciation("e1");
    assert.equal(mock.calls[0].options.method, "DELETE");
  } finally {
    mock.restore();
  }
});

test("presets load and save", async () => {
  const mock = mockFetch({ ok: true, presets: [{ id: "p1" }] });

  try {
    const presets = await listPresets();
    assert.equal(presets.length, 1);
    assert.equal(mock.calls[0].url, "/api/presets");
  } finally {
    mock.restore();
  }

  const createMock = mockFetch({ ok: true, preset: { id: "p2" } });

  try {
    await createPreset({
      name: "Fast",
      engine: "kokoro",
      voice: "af_heart",
      speed: 1.25,
    });

    const body = JSON.parse(createMock.calls[0].options.body);
    assert.equal(body.speed, 1.25);
    assert.equal(createMock.calls[0].options.method, "POST");
  } finally {
    createMock.restore();
  }
});

test("preset updates use PATCH with the id", async () => {
  const mock = mockFetch({ ok: true, preset: { id: "p1" } });

  try {
    await updatePreset("p1", { name: "X", engine: "voiceforge" });
    assert.match(mock.calls[0].url, /\/api\/presets\/p1$/);
    assert.equal(mock.calls[0].options.method, "PATCH");
  } finally {
    mock.restore();
  }
});

test("preset deletion uses DELETE", async () => {
  const mock = mockFetch({ ok: true, deleted: { id: "p1" } });

  try {
    await deletePreset("p1");
    assert.equal(mock.calls[0].options.method, "DELETE");
  } finally {
    mock.restore();
  }
});

test("server errors become ApiError with a code", async () => {
  const mock = mockFetch(
    { ok: false, error: "Preset not found.", code: "preset_not_found" },
    { ok: false, status: 404 },
  );

  try {
    await assert.rejects(
      () => updatePreset("missing", {}),
      (error) => {
        assert.ok(error instanceof ApiError);
        assert.equal(error.code, "preset_not_found");
        assert.equal(error.status, 404);
        return true;
      },
    );
  } finally {
    mock.restore();
  }
});
