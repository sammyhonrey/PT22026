import test from "node:test";
import assert from "node:assert/strict";

import { ApiError } from "../js/api.js";
import {
  audioUrl,
  deleteRecord,
  downloadUrl,
  getRecord,
  listLibrary,
  renameRecord,
} from "../js/library-api.js";

function mockFetch(body, { ok = true, status = 200 } = {}) {
  const calls = [];
  const original = globalThis.fetch;

  globalThis.fetch = async (url, options = {}) => {
    calls.push({ url, options });
    return {
      ok,
      status,
      json: async () => body,
    };
  };

  return {
    calls,
    restore() {
      globalThis.fetch = original;
    },
  };
}

const record = {
  id: "abc123",
  name: "Welcome",
  original_text: "Welcome to VoiceForge.",
  engine: "kokoro",
  voice: "af_heart",
  speed: 1,
  duration_seconds: 2,
  file_size: 1024,
  source_type: "studio",
  created_at: "2024-05-01T10:00:00Z",
  updated_at: "2024-05-01T10:00:00Z",
};

test("listLibrary builds the query string", async () => {
  const mock = mockFetch({
    ok: true,
    records: [record],
    pagination: { page: 1, limit: 10, total: 1, pages: 1 },
  });

  try {
    const payload = await listLibrary({ q: "welcome", sort: "name", page: 1 });

    assert.equal(payload.records.length, 1);
    assert.match(mock.calls[0].url, /\/api\/library\?/);
    assert.match(mock.calls[0].url, /q=welcome/);
    assert.match(mock.calls[0].url, /sort=name/);
  } finally {
    mock.restore();
  }
});

test("getRecord requests one record", async () => {
  const mock = mockFetch({ ok: true, record });

  try {
    const result = await getRecord("abc123");

    assert.equal(result.id, "abc123");
    assert.equal(mock.calls[0].url, "/api/library/abc123");
  } finally {
    mock.restore();
  }
});

test("renameRecord sends PATCH with the name", async () => {
  const mock = mockFetch({ ok: true, record: { ...record, name: "New" } });

  try {
    const updated = await renameRecord("abc123", "New");
    const options = mock.calls[0].options;

    assert.equal(updated.name, "New");
    assert.equal(options.method, "PATCH");
    assert.equal(JSON.parse(options.body).name, "New");
  } finally {
    mock.restore();
  }
});

test("deleteRecord sends DELETE", async () => {
  const mock = mockFetch({ ok: true, deleted: { id: "abc123" } });

  try {
    const deleted = await deleteRecord("abc123");

    assert.equal(deleted.id, "abc123");
    assert.equal(mock.calls[0].options.method, "DELETE");
  } finally {
    mock.restore();
  }
});

test("API errors become ApiError with the server code", async () => {
  const mock = mockFetch(
    { ok: false, error: "Audio record not found.", code: "record_not_found" },
    { ok: false, status: 404 },
  );

  try {
    await assert.rejects(
      () => getRecord("missing"),
      (error) => {
        assert.ok(error instanceof ApiError);
        assert.equal(error.code, "record_not_found");
        assert.equal(error.status, 404);
        return true;
      },
    );
  } finally {
    mock.restore();
  }
});

test("network failures become ApiError", async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async () => {
    throw new TypeError("failed to fetch");
  };

  try {
    await assert.rejects(() => listLibrary({}), ApiError);
  } finally {
    globalThis.fetch = original;
  }
});

test("audio and download URLs are encoded", () => {
  const encoded = { id: "a b/c" };

  assert.equal(audioUrl(encoded), "/api/library/a%20b%2Fc/audio");
  assert.equal(downloadUrl(encoded), "/api/library/a%20b%2Fc/audio?download=1");
});
