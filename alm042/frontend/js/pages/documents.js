/**
 * Document to Speech: upload a PDF/DOCX, review the extracted text, then
 * send it through the existing Long Text job workflow.
 */

import { ApiError } from "../api.js";
import { extractDocument } from "../advanced-api.js";
import {
  MAX_DOCUMENT_BYTES,
  MAX_DOCUMENT_CHARS,
  describeDocument,
  validateDocumentFile,
  validateDocumentText,
} from "../advanced-state.js";
import { createJobView } from "../components/job-view.js";
import {
  createPresetSelect,
  createPronunciationToggle,
} from "../components/generation-extras.js";
import {
  formatSpeed,
  speedForRequest,
  voiceControlsVisible,
  voiceForRequest,
} from "../engine-ui.js";
import { createJobPoller, submitJob } from "../jobs.js";

export function createDocumentsPage({ capabilities, notifications }) {
  const engines = capabilities.engines || {};
  const kokoro = engines.kokoro || {};
  const voices = kokoro.voices || [];
  const speedRange = kokoro.speed || {};

  const grid = document.createElement("div");
  grid.className = "studio-grid";

  // --- Document panel ---------------------------------------------------
  const panel = document.createElement("section");
  panel.className = "panel";

  const head = document.createElement("div");
  head.className = "panel-head";
  head.innerHTML =
    '<h2 class="panel-title">Document</h2><span class="panel-hint">PDF or DOCX · up to 10 MB</span>';

  const body = document.createElement("div");
  body.className = "panel-body stack";

  const input = document.createElement("input");
  input.type = "file";
  input.accept = ".pdf,.docx";
  input.hidden = true;

  const pickButton = document.createElement("button");
  pickButton.type = "button";
  pickButton.className = "button button-ghost";
  pickButton.textContent = "⬆ Choose PDF or DOCX";

  const info = document.createElement("p");
  info.className = "field-note";
  info.textContent = "No document loaded. Upload one to extract its text.";

  const textarea = document.createElement("textarea");
  textarea.className = "textarea document-text";
  textarea.placeholder = "Extracted text appears here and can be edited before generating.";
  textarea.maxLength = MAX_DOCUMENT_CHARS;

  const counter = document.createElement("span");
  counter.className = "counter";
  counter.textContent = `0 / ${MAX_DOCUMENT_CHARS}`;

  const metaRow = document.createElement("div");
  metaRow.className = "textarea-meta";
  metaRow.append(
    Object.assign(document.createElement("span"), {
      textContent: "Scanned PDFs contain images and need OCR (not supported).",
    }),
    counter,
  );

  body.append(pickButton, input, info, textarea, metaRow);
  panel.append(head, body);

  // --- Settings ---------------------------------------------------------
  const settings = document.createElement("section");
  settings.className = "panel";

  const settingsHead = document.createElement("div");
  settingsHead.className = "panel-head";
  settingsHead.innerHTML = '<h2 class="panel-title">Speech settings</h2>';

  const settingsBody = document.createElement("div");
  settingsBody.className = "panel-body stack";

  const engineSelect = document.createElement("select");
  engineSelect.className = "select";

  for (const [id, engine] of Object.entries(engines)) {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = engine.name || id;
    engineSelect.append(option);
  }

  if (capabilities.default && engines[capabilities.default]) {
    engineSelect.value = capabilities.default;
  }

  const voiceSelect = document.createElement("select");
  voiceSelect.className = "select";

  for (const voice of voices) {
    const option = document.createElement("option");
    option.value = voice.id;
    option.textContent = voice.name;
    voiceSelect.append(option);
  }

  const defaultVoice = voices.find((voice) => voice.default);

  if (defaultVoice) {
    voiceSelect.value = defaultVoice.id;
  }

  const speedField = document.createElement("div");
  speedField.className = "field";

  const speedValue = document.createElement("span");
  speedValue.className = "badge badge-accent";
  speedValue.textContent = formatSpeed(speedRange.default ?? 1);

  const speedLabelRow = document.createElement("div");
  speedLabelRow.className = "field-label-row";
  speedLabelRow.append(
    Object.assign(document.createElement("label"), {
      className: "field-label",
      textContent: "Speed",
    }),
    speedValue,
  );

  const speedInput = document.createElement("input");
  speedInput.type = "range";
  speedInput.className = "range";
  speedInput.min = speedRange.min ?? 0.5;
  speedInput.max = speedRange.max ?? 2;
  speedInput.step = "0.05";
  speedInput.value = speedRange.default ?? 1;

  speedField.append(speedLabelRow, speedInput);

  const engineField = document.createElement("div");
  engineField.className = "field";
  engineField.append(
    Object.assign(document.createElement("label"), {
      className: "field-label",
      textContent: "Engine",
    }),
    engineSelect,
  );

  const voiceField = document.createElement("div");
  voiceField.className = "field";
  voiceField.append(
    Object.assign(document.createElement("label"), {
      className: "field-label",
      textContent: "Voice",
    }),
    voiceSelect,
  );

  const presetField = createPresetSelect({
    capabilities,
    notifications,
    onApply: ({ engine, voice, speed }) => {
      engineSelect.value = engine;
      syncVisibility();

      if (voice) {
        voiceSelect.value = voice;
      }

      if (typeof speed === "number") {
        speedInput.value = speed;
        speedValue.textContent = formatSpeed(speed);
      }
    },
  });

  const pronunciation = createPronunciationToggle();

  const generate = document.createElement("button");
  generate.type = "button";
  generate.className = "button button-primary button-block";
  generate.textContent = "Generate from document";

  settingsBody.append(
    engineField,
    voiceField,
    speedField,
    presetField.element,
    pronunciation.element,
    generate,
  );
  settings.append(settingsHead, settingsBody);

  const jobView = createJobView({ title: "Document progress" });
  const side = document.createElement("div");
  side.className = "stack";
  side.append(settings, jobView.element);

  grid.append(panel, side);

  let poller = null;
  let currentDocument = null;

  function syncVisibility() {
    const engine = engineSelect.value;
    voiceField.hidden = !voiceControlsVisible(engine);
    speedField.hidden = !engineSupportsSpeed(engine);
  }

  function engineSupportsSpeed(engine) {
    return Boolean(engines[engine]?.speed?.supported);
  }

  function updateCounter() {
    counter.textContent = `${textarea.value.length} / ${MAX_DOCUMENT_CHARS}`;
    counter.classList.toggle("is-over", textarea.value.length > MAX_DOCUMENT_CHARS);
  }

  textarea.addEventListener("input", updateCounter);
  speedInput.addEventListener("input", () => {
    speedValue.textContent = formatSpeed(speedInput.value);
  });
  engineSelect.addEventListener("change", syncVisibility);

  pickButton.addEventListener("click", () => input.click());

  input.addEventListener("change", async () => {
    const file = input.files?.[0];

    if (!file) {
      return;
    }

    const check = validateDocumentFile(file);

    if (!check.ok) {
      notifications.error(check.error);
      input.value = "";
      return;
    }

    pickButton.disabled = true;
    pickButton.textContent = "Extracting…";

    try {
      currentDocument = await extractDocument(file);
      textarea.value = currentDocument.text;
      updateCounter();
      info.textContent = describeDocument(currentDocument);
      notifications.success(
        `Extracted ${currentDocument.characters} characters.`,
      );
    } catch (error) {
      notifications.error(
        error instanceof ApiError ? error.message : "Extraction failed.",
      );
    } finally {
      pickButton.disabled = false;
      pickButton.textContent = "⬆ Choose PDF or DOCX";
      input.value = "";
    }
  });

  generate.addEventListener("click", async () => {
    const validation = validateDocumentText(textarea.value);

    if (!validation.ok) {
      notifications.error(validation.error);
      return;
    }

    const engine = engineSelect.value;
    const payload = {
      type: "long_text",
      engine,
      pronunciation: pronunciation.enabled,
      items: [
        {
          name: currentDocument?.filename || "Document",
          text: validation.text,
        },
      ],
    };

    if (voiceControlsVisible(engine)) {
      payload.voice = voiceForRequest(engine, voiceSelect.value);
      const speed = speedForRequest(engine, speedInput.value);

      if (speed !== null) {
        payload.speed = speed;
      }
    }

    generate.disabled = true;
    generate.textContent = "Queuing…";
    jobView.clear();

    try {
      const job = await submitJob(payload);
      jobView.render(job);
      notifications.info("Document queued for speech generation.");

      poller = createJobPoller({
        jobId: job.id,
        onUpdate: (updated) => {
          jobView.render(updated);

          if (updated.status === "completed") {
            notifications.success("Document audio is ready and saved in the library.");
            generate.disabled = false;
            generate.textContent = "Generate from document";
          } else if (updated.status === "failed") {
            notifications.error(updated.error || "Document generation failed.");
            generate.disabled = false;
            generate.textContent = "Generate from document";
          }
        },
        onError: (error) => {
          notifications.error(error.message);
          generate.disabled = false;
          generate.textContent = "Generate from document";
        },
      });

      poller.start();
    } catch (error) {
      notifications.error(
        error instanceof ApiError ? error.message : "Could not start the job.",
      );
      generate.disabled = false;
      generate.textContent = "Generate from document";
    }
  });

  syncVisibility();
  updateCounter();

  return {
    element: grid,
    limits: { maxBytes: MAX_DOCUMENT_BYTES, maxChars: MAX_DOCUMENT_CHARS },

    deactivate() {
      if (poller) {
        poller.stop();
        poller = null;
      }
    },
  };
}
