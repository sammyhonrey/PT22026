/**
 * Upload validation and file reading.
 *
 * The browser checks files for quick feedback; the server re-validates
 * everything (extension, size, encoding) because client checks can be
 * bypassed.
 */

export const MAX_UPLOAD_FILES = 20;
export const MAX_FILE_BYTES = 1024 * 1024;
export const MAX_TOTAL_BYTES = 5 * 1024 * 1024;

export function isTxtFile(name) {
  return typeof name === "string" && name.toLowerCase().endsWith(".txt");
}

/**
 * @param {FileList|File[]} files
 * @returns {{ok: true, files: File[]} | {ok: false, error: string}}
 */
export function validateFiles(files) {
  const list = Array.from(files || []);

  if (list.length === 0) {
    return { ok: false, error: "Select at least one .txt file." };
  }

  if (list.length > MAX_UPLOAD_FILES) {
    return {
      ok: false,
      error: `Too many files: ${list.length}. The limit is ${MAX_UPLOAD_FILES}.`,
    };
  }

  let total = 0;

  for (const file of list) {
    if (!isTxtFile(file.name)) {
      return {
        ok: false,
        error: `"${file.name}" is not a .txt file.`,
      };
    }

    if (file.size > MAX_FILE_BYTES) {
      return {
        ok: false,
        error: `"${file.name}" is larger than 1 MB.`,
      };
    }

    total += file.size;
  }

  if (total > MAX_TOTAL_BYTES) {
    return {
      ok: false,
      error: "The selected files are larger than 5 MB in total.",
    };
  }

  return { ok: true, files: list };
}

/** Base64-encode an ArrayBuffer without blowing the call stack. */
export function toBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  const chunk = 0x8000;
  let binary = "";

  for (let index = 0; index < bytes.length; index += chunk) {
    binary += String.fromCharCode(...bytes.subarray(index, index + chunk));
  }

  return btoa(binary);
}

export async function readFilePayload(file) {
  const buffer = await file.arrayBuffer();

  return {
    name: file.name,
    data_base64: toBase64(buffer),
  };
}
