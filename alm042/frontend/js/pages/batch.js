/**
 * Batch Studio: several text entries, optional .txt uploads, one job.
 *
 * Engine, voice and speed come from `/api/engines`; nothing is hardcoded.
 */

import { ApiError } from "../api.js";
import {
  MAX_ENTRIES,
  MAX_ENTRY_CHARS,
  addEntry,
  createEntry,
  removeEntry,
  updateEntry,
  validateEntries,
} from "../batch-state.js";
import { createJobView } from "../components/job-view.js";
import {
  createPresetSelect,
  createPronunciationToggle,
} from "../components/generation-extras.js";
import {
  formatSpeed,
  speedControlVisible,
  voiceControlsVisible,
} from "../engine-ui.js";
import { createJobPoller, submitJob, uploadTextFiles } from "../jobs.js";
import {
  MAX_UPLOAD_FILES,
  readFilePayload,
  validateFiles,
} from "../upload.js";

export function createBatchPage({ capabilities, notifications }) {
  const engines = capabilities.engines || {};
  const kokoro = engines.kokoro || {};
  const voices = kokoro.voices || [];
  const speedRange = kokoro.speed || {};

  let entries = [createEntry()];
  let poller = null;
  let busy = false;

  const layout = document.createElement("div");
  layout.className = "batch-layout";

  // --- Entries ---------------------------------------------------------
  const entriesPanel = document.createElement("section");
  entriesPanel.className = "panel";

  const entriesHead = document.createElement("div");
  entriesHead.className = "panel-head";

  const entriesTitle = document.createElement("h2");
  entriesTitle.className = "panel-title";
  entriesTitle.textContent = "Entries";

  const entriesHint = document.createElement("span");
  entriesHint.className = "panel-hint";
  entriesHint.textContent = `up to ${MAX_ENTRIES} entries · ${MAX_ENTRY_CHARS} characters each`;

  entriesHead.append(entriesTitle, entriesHint);

  const entriesBody = document.createElement("div");
  entriesBody.className = "panel-body stack";

  const uploadInput = document.createElement("input");
  uploadInput.type = "file";
  uploadInput.accept = ".txt,text/plain";
  uploadInput.multiple = true;
  uploadInput.hidden = true;

  const toolbar = document.createElement("div");
  toolbar.className = "batch-toolbar";

  const addButton = document.createElement("button");
  addButton.type = "button";
  addButton.className = "button button-ghost";
  addButton.textContent = "+ Add entry";

  const uploadButton = document.createElement("button");
  uploadButton.type = "button";
  uploadButton.className = "button button-ghost";
  uploadButton.textContent = "\u2b06 Upload .txt files";

  toolbar.append(addButton, uploadButton, uploadInput);

  const entriesList = document.createElement("div");
  entriesList.className = "batch-entries";

  entriesBody.append(toolbar, entriesList);
  entriesPanel.append(entriesHead, entriesBody);

  // --- Controls --------------------------------------------------------
  const controlPanel = document.createElement("section");
  controlPanel.className = "panel";

  const controlHead = document.createElement("div");
  controlHead.className = "panel-head";
  controlHead.innerHTML = '<h2 class="panel-title">Settings</h2>';

  const controlBody = document.createElement("div");
  controlBody.className = "panel-body stack";

  const engineSelect = document.createElement("select");
  engineSelect.className = "select";
  engineSelect.id = "batch-engine";

  for (const [id, info] of Object.entries(engines)) {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = info.name || id;
    engineSelect.append(option);
  }

  if (capabilities.default && engines[capabilities.default]) {
    engineSelect.value = capabilities.default;
  }

  const voiceSelect = document.createElement("select");
  voiceSelect.className = "select";
  voiceSelect.id = "batch-voice";

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

  const speedRow = document.createElement("div");
  speedRow.className = "field";

  const speedLabelRow = document.createElement("div");
  speedLabelRow.className = "field-label-row";

  const speedLabel = document.createElement("label");
  speedLabel.className = "field-label";
  speedLabel.setAttribute("for", "batch-speed");
  speedLabel.textContent = "Speed";

  const speedValue = document.createElement("span");
  speedValue.className = "badge badge-accent";
  speedValue.textContent = formatSpeed(speedRange.default ?? 1);

  speedLabelRow.append(speedLabel, speedValue);

  const speedInput = document.createElement("input");
  speedInput.type = "range";
  speedInput.className = "range";
  speedInput.id = "batch-speed";
  speedInput.min = speedRange.min ?? 0.5;
  speedInput.max = speedRange.max ?? 2;
  speedInput.step = "0.05";
  speedInput.value = speedRange.default ?? 1;

  speedRow.append(speedLabelRow, speedInput);

  const engineField = document.createElement("div");
  engineField.className = "field";
  const engineLabel = document.createElement("label");
  engineLabel.className = "field-label";
  engineLabel.textContent = "Engine";
  engineField.append(engineLabel, engineSelect);

  const voiceField = document.createElement("div");
  voiceField.className = "field";
  const voiceLabel = document.createElement("label");
  voiceLabel.className = "field-label";
  voiceLabel.textContent = "Voice";
  voiceField.append(voiceLabel, voiceSelect);

  const presetField = createPresetSelect({
    capabilities,
    notifications,
    onApply: ({ engine, voice, speed }) => {
      engineSelect.value = engine;
      syncControls();

      if (voice) {
        voiceSelect.value = voice;
      }

      if (typeof speed === "number") {
        speedInput.value = speed;
        speedValue.textContent = formatSpeed(speed);
      }
    },
  });

  const pronunciationToggle = createPronunciationToggle();

  const submitButton = document.createElement("button");
  submitButton.type = "button";
  submitButton.className = "button button-primary button-block";
  submitButton.textContent = "Generate batch";

  controlBody.append(
    engineField,
    voiceField,
    speedRow,
    presetField.element,
    pronunciationToggle.element,
    submitButton,
  );
  controlPanel.append(controlHead, controlBody);

  // --- Job view ---------------------------------------------------------
  const jobView = createJobView({ title: "Batch progress" });

  const side = document.createElement("div");
  side.className = "stack";
  side.append(controlPanel, jobView.element);

  layout.append(entriesPanel, side);

  // --- Behaviour --------------------------------------------------------
  function syncControls() {
    const engine = engineSelect.value;
    voiceField.hidden = !voiceControlsVisible(engine);
    speedRow.hidden = !speedControlVisible(engine);
  }

  function setBusy(value) {
    busy = value;
    submitButton.disabled = value;
    addButton.disabled = value;
    uploadButton.disabled = value;
    engineSelect.disabled = value;
    voiceSelect.disabled = value;
    speedInput.disabled = value;
    submitButton.textContent = value ? "Working…" : "Generate batch";
  }

  function renderEntries() {
    entriesList.replaceChildren();

    entries.forEach((entry, index) => {
      const card = document.createElement("div");
      card.className = "batch-entry";

      const head = document.createElement("div");
      head.className = "batch-entry-head";

      const nameInput = document.createElement("input");
      nameInput.className = "input";
      nameInput.type = "text";
      nameInput.placeholder = `Entry ${index + 1} name`;
      nameInput.value = entry.name;
      nameInput.maxLength = 80;

      nameInput.addEventListener("input", () => {
        entries = updateEntry(entries, entry.id, { name: nameInput.value });
      });

      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "icon-button";
      remove.title = "Remove entry";
      remove.setAttribute("aria-label", `Remove entry ${index + 1}`);
      remove.textContent = "\u00d7";
      remove.disabled = entries.length === 1;

      remove.addEventListener("click", () => {
        entries = removeEntry(entries, entry.id);
        renderEntries();
      });

      head.append(nameInput, remove);

      const text = document.createElement("textarea");
      text.className = "textarea batch-entry-text";
      text.placeholder = "Text to synthesise…";
      text.value = entry.text;

      text.addEventListener("input", () => {
        entries = updateEntry(entries, entry.id, { text: text.value });
      });

      card.append(head, text);
      entriesList.append(card);
    });
  }

  function addNewEntry() {
    if (entries.length >= MAX_ENTRIES) {
      notifications.error(`The limit is ${MAX_ENTRIES} entries.`);
      return;
    }

    entries = addEntry(entries, createEntry());
    renderEntries();
  }

  addButton.addEventListener("click", addNewEntry);
  uploadButton.addEventListener("click", () => uploadInput.click());

  uploadInput.addEventListener("change", async () => {
    const check = validateFiles(uploadInput.files);

    if (!check.ok) {
      notifications.error(check.error);
      uploadInput.value = "";
      return;
    }

    try {
      const payloads = await Promise.all(check.files.map(readFilePayload));
      const files = await uploadTextFiles(payloads);

      for (const file of files) {
        if (entries.length >= MAX_ENTRIES) {
          notifications.error(`Only ${MAX_ENTRIES} entries fit in one batch.`);
          break;
        }

        entries = addEntry(
          entries,
          createEntry({ name: file.name, text: file.text }),
        );
      }

      renderEntries();
      notifications.success(
        `Added ${files.length} file${files.length === 1 ? "" : "s"} from upload.`,
      );
    } catch (error) {
      notifications.error(
        error instanceof ApiError ? error.message : "Upload failed.",
      );
    } finally {
      uploadInput.value = "";
    }
  });

  engineSelect.addEventListener("change", syncControls);

  speedInput.addEventListener("input", () => {
    speedValue.textContent = formatSpeed(speedInput.value);
  });

  submitButton.addEventListener("click", async () => {
    if (busy) {
      return;
    }

    const validation = validateEntries(entries);

    if (!validation.ok) {
      notifications.error(validation.error);
      return;
    }

    const engine = engineSelect.value;
    const payload = {
      type: "batch",
      engine,
      pronunciation: pronunciationToggle.enabled,
      items: validation.items,
    };

    if (voiceControlsVisible(engine)) {
      payload.voice = voiceSelect.value;
      payload.speed = Number(speedInput.value);
    }

    setBusy(true);
    jobView.clear();

    try {
      const job = await submitJob(payload);
      jobView.render(job);
      notifications.info(`Job queued: ${job.total} item(s).`);

      poller = createJobPoller({
        jobId: job.id,
        onUpdate: (updated) => {
          jobView.render(updated);

          if (updated.status === "completed") {
            notifications.success(
              `Batch finished: ${updated.completed} completed, ${updated.failed} failed.`,
            );
            setBusy(false);
          } else if (updated.status === "failed") {
            notifications.error(updated.error || "The batch failed.");
            setBusy(false);
          }
        },
        onError: (error) => {
          notifications.error(error.message);
          setBusy(false);
        },
      });

      poller.start();
    } catch (error) {
      notifications.error(
        error instanceof ApiError ? error.message : "Could not start the batch.",
      );
      setBusy(false);
    }
  });

  syncControls();
  renderEntries();

  return {
    element: layout,

    deactivate() {
      if (poller) {
        poller.stop();
        poller = null;
      }
    },
  };
}
