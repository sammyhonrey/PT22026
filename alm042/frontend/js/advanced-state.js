/**
 * Pure helpers for documents, pronunciation, MP3 export and presets.
 */

export const MAX_DOCUMENT_BYTES = 10 * 1024 * 1024;
export const MAX_DOCUMENT_CHARS = 20000;
export const DOCUMENT_EXTENSIONS = [".pdf", ".docx"];
export const MP3_BITRATES = [128, 192, 320];
export const DEFAULT_MP3_BITRATE = 192;
export const MAX_WORD_LENGTH = 80;
export const MAX_PRESET_NAME = 60;

export function isDocumentFile(name) {
  if (typeof name !== "string") {
    return false;
  }

  const lower = name.toLowerCase();

  return DOCUMENT_EXTENSIONS.some((extension) => lower.endsWith(extension));
}

export function validateDocumentFile(file) {
  if (!file || typeof file.name !== "string") {
    return { ok: false, error: "Choose a PDF or DOCX file." };
  }

  if (!isDocumentFile(file.name)) {
    return {
      ok: false,
      error: `"${file.name}" is not supported. Upload a PDF or DOCX file.`,
    };
  }

  if (file.size > MAX_DOCUMENT_BYTES) {
    return {
      ok: false,
      error: "The document is larger than 10 MB.",
    };
  }

  return { ok: true };
}

export function describeDocument(document) {
  if (!document) {
    return "";
  }

  const pages =
    typeof document.pages === "number"
      ? `${document.pages} page${document.pages === 1 ? "" : "s"}`
      : null;

  const parts = [document.filename, document.format?.toUpperCase()];

  if (pages) {
    parts.push(pages);
  }

  parts.push(`${document.characters} characters`);

  return parts.filter(Boolean).join(" · ");
}

export function validateDocumentText(text) {
  const value = String(text || "").trim();

  if (!value) {
    return { ok: false, error: "The document text is empty." };
  }

  if (value.length > MAX_DOCUMENT_CHARS) {
    return {
      ok: false,
      error: `The text is too long (${value.length} characters). The limit is ${MAX_DOCUMENT_CHARS}. Shorten it before generating.`,
    };
  }

  return { ok: true, text: value };
}

export function validatePronunciationDraft({ word, pronunciation }) {
  const cleanedWord = String(word || "").trim();
  const cleanedPronunciation = String(pronunciation || "").trim();

  if (!cleanedWord) {
    return { ok: false, error: "Enter the word or phrase." };
  }

  if (cleanedWord.length > MAX_WORD_LENGTH) {
    return {
      ok: false,
      error: `The word is too long (${cleanedWord.length} characters).`,
    };
  }

  if (!cleanedPronunciation) {
    return { ok: false, error: "Enter the pronunciation." };
  }

  return {
    ok: true,
    word: cleanedWord,
    pronunciation: cleanedPronunciation,
  };
}

/**
 * Validate a preset against the real engine capabilities from /api/engines.
 */
export function validatePresetDraft(
  { name, engine, voice, speed },
  capabilities,
) {
  const cleanedName = String(name || "").trim();

  if (!cleanedName) {
    return { ok: false, error: "Enter a preset name." };
  }

  if (cleanedName.length > MAX_PRESET_NAME) {
    return {
      ok: false,
      error: `The preset name is too long (${cleanedName.length} characters).`,
    };
  }

  const engines = capabilities?.engines || {};
  const info = engines[engine];

  if (!engine || !info) {
    return { ok: false, error: "Choose a valid engine." };
  }

  if (info.speed?.supported) {
    const voices = (info.voices || []).map((entry) => entry.id);

    if (!voices.includes(voice)) {
      return { ok: false, error: "Choose a valid voice for this engine." };
    }

    const value = Number(speed);

    if (
      !Number.isFinite(value) ||
      value < info.speed.min ||
      value > info.speed.max
    ) {
      return {
        ok: false,
        error: `Speed must be between ${info.speed.min} and ${info.speed.max}.`,
      };
    }

    return {
      ok: true,
      preset: {
        name: cleanedName,
        engine,
        voice,
        speed: value,
      },
    };
  }

  // Engines without speed support never store a speed or a voice.
  return {
    ok: true,
    preset: { name: cleanedName, engine, voice: null, speed: null },
  };
}

/**
 * Return control values for applying a preset, clamped to what the engine
 * actually supports.
 */
export function applyPresetToControls(preset, capabilities) {
  const engines = capabilities?.engines || {};
  const info = engines[preset?.engine];

  if (!preset || !info) {
    return null;
  }

  const voices = (info.voices || []).map((entry) => entry.id);

  return {
    engine: preset.engine,
    voice: info.speed?.supported && voices.includes(preset.voice)
      ? preset.voice
      : null,
    speed: info.speed?.supported ? Number(preset.speed) : null,
  };
}

export function describePreset(preset, capabilities) {
  if (!preset) {
    return "";
  }

  const info = capabilities?.engines?.[preset.engine];
  const parts = [info?.name || preset.engine];

  if (preset.voice) {
    const voice = (info?.voices || []).find(
      (entry) => entry.id === preset.voice,
    );
    parts.push(voice?.name || preset.voice);
  }

  if (typeof preset.speed === "number") {
    parts.push(`${preset.speed.toFixed(2)}×`);
  }

  return parts.join(" · ");
}

export function buildMp3Url(recordId, bitrate = DEFAULT_MP3_BITRATE) {
  const safeBitrate = MP3_BITRATES.includes(Number(bitrate))
    ? Number(bitrate)
    : DEFAULT_MP3_BITRATE;

  return `/api/library/${encodeURIComponent(recordId)}/mp3?bitrate=${safeBitrate}`;
}

export function buildJobZipUrl(
  jobId,
  format = "wav",
  bitrate = DEFAULT_MP3_BITRATE,
) {
  const suffix =
    format === "mp3" ? `&format=mp3&bitrate=${bitrate}` : "&format=wav";

  return `/api/jobs/${encodeURIComponent(jobId)}/download?format=wav${suffix}`;
}

export function buildJobItemMp3Url(
  jobId,
  itemId,
  bitrate = DEFAULT_MP3_BITRATE,
) {
  return `/api/jobs/${encodeURIComponent(jobId)}/audio/${encodeURIComponent(
    itemId,
  )}?format=mp3&bitrate=${bitrate}`;
}
