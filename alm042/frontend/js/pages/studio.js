/**
 * Speech Studio page.
 *
 * Two modes share one composer:
 *   - Short text: single request, Enter generates (Shift+Enter = newline).
 *   - Long text: background job with chunked synthesis,
 *     Ctrl/Cmd+Enter submits (Enter = newline).
 *
 * Engines, voices, speed ranges and limits come from `/api/engines`.
 */

import { ApiError, generateSpeech } from "../api.js";
import { createAudioPanel } from "../components/audio-player.js";
import { createJobView } from "../components/job-view.js";
import {
  createPresetSelect,
  createPronunciationToggle,
} from "../components/generation-extras.js";
import { MAX_ENTRY_CHARS } from "../batch-state.js";
import {
  describeSpeed,
  describeVoice,
  formatSpeed,
  speedControlVisible,
  speedForRequest,
  voiceControlsVisible,
  voiceForRequest,
} from "../engine-ui.js";
import { createJobPoller, submitJob } from "../jobs.js";
import { KeyAction, resolveKeyAction, resolveLongTextKey } from "../keyboard.js";
import { createGenerationGate } from "../state.js";
import {
  DEFAULT_MAX_LENGTH,
  countCharacters,
  counterState,
  validateText,
} from "../validation.js";

const EXAMPLES = [
  "Hello, welcome to VoiceForge.",
  "The quick brown fox jumps over the lazy dog.",
  "This is a test of the speech synthesis system.",
];

function fieldLabel(text, targetId) {
  const label = document.createElement("label");
  label.className = "field-label";
  label.setAttribute("for", targetId);
  label.textContent = text;
  return label;
}

/**
 * @param {{capabilities: object, notifications: object}} options
 */
export function createStudioPage({ capabilities, notifications }) {
  const engines = capabilities.engines || {};
  const defaultEngine = capabilities.default;
  const shortLimit = capabilities.limits?.max_text_length || DEFAULT_MAX_LENGTH;

  const kokoro = engines.kokoro || {};
  const kokoroVoices = kokoro.voices || [];
  const speedRange = kokoro.speed || {};

  const voiceLabels = Object.fromEntries(
    kokoroVoices.map((voice) => [voice.id, voice.name]),
  );

  const gate = createGenerationGate();

  let mode = "short";
  let poller = null;

  const wrapper = document.createElement("div");
  wrapper.className = "stack";

  const grid = document.createElement("div");
  grid.className = "studio-grid";

  // --- Composer ---------------------------------------------------------
  const composer = document.createElement("section");
  composer.className = "panel";
  composer.innerHTML = `
    <div class="panel-head">
      <h2 class="panel-title">Compose</h2>
      <span class="panel-hint">English text</span>
    </div>
  `;

  const composerBody = document.createElement("div");
  composerBody.className = "panel-body stack";

  const modeBar = document.createElement("div");
  modeBar.className = "segmented";
  modeBar.setAttribute("role", "group");
  modeBar.setAttribute("aria-label", "Text mode");

  const shortModeButton = document.createElement("button");
  shortModeButton.type = "button";
  shortModeButton.className = "segment is-active";
  shortModeButton.dataset.mode = "short";
  shortModeButton.textContent = "Short text";

  const longModeButton = document.createElement("button");
  longModeButton.type = "button";
  longModeButton.className = "segment";
  longModeButton.dataset.mode = "long";
  longModeButton.textContent = "Long text";

  modeBar.append(shortModeButton, longModeButton);

  const textField = document.createElement("div");
  textField.className = "field";

  const textarea = document.createElement("textarea");
  textarea.className = "textarea";
  textarea.id = "speech-text";
  textarea.placeholder = "Type or paste a sentence, for example: Welcome to VoiceForge.";
  textarea.setAttribute("aria-describedby", "composer-hint");

  textField.append(fieldLabel("Text", "speech-text"), textarea);

  const metaRow = document.createElement("div");
  metaRow.className = "textarea-meta";

  const hint = document.createElement("span");
  hint.id = "composer-hint";

  const counter = document.createElement("span");
  counter.className = "counter";

  metaRow.append(hint, counter);

  // Engine + voice
  const selectors = document.createElement("div");
  selectors.className = "field-row";

  const engineField = document.createElement("div");
  engineField.className = "field";

  const engineSelect = document.createElement("select");
  engineSelect.className = "select";
  engineSelect.id = "engine-select";

  for (const [id, info] of Object.entries(engines)) {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = info.name || id;
    engineSelect.append(option);
  }

  if (defaultEngine && engines[defaultEngine]) {
    engineSelect.value = defaultEngine;
  }

  engineField.append(fieldLabel("Engine", "engine-select"), engineSelect);

  const voiceField = document.createElement("div");
  voiceField.className = "field";

  const voiceSelect = document.createElement("select");
  voiceSelect.className = "select";
  voiceSelect.id = "voice-select";

  for (const voice of kokoroVoices) {
    const option = document.createElement("option");
    option.value = voice.id;
    option.textContent = voice.name;
    voiceSelect.append(option);
  }

  const defaultVoice = kokoroVoices.find((voice) => voice.default);

  if (defaultVoice) {
    voiceSelect.value = defaultVoice.id;
  }

  voiceField.append(fieldLabel("Voice", "voice-select"), voiceSelect);

  selectors.append(engineField, voiceField);

  // Speed (Kokoro only)
  const speedField = document.createElement("div");
  speedField.className = "field";
  speedField.id = "speed-field";

  const speedLabelRow = document.createElement("div");
  speedLabelRow.className = "field-label-row";

  const speedLabel = document.createElement("label");
  speedLabel.className = "field-label";
  speedLabel.setAttribute("for", "speed-range");
  speedLabel.textContent = "Speed";

  const speedValue = document.createElement("span");
  speedValue.className = "badge badge-accent";
  speedValue.id = "speed-value";

  speedLabelRow.append(speedLabel, speedValue);

  const speedInput = document.createElement("input");
  speedInput.type = "range";
  speedInput.className = "range";
  speedInput.id = "speed-range";
  speedInput.min = speedRange.min ?? 0.5;
  speedInput.max = speedRange.max ?? 2;
  speedInput.step = "0.05";
  speedInput.value = speedRange.default ?? 1;

  const speedScale = document.createElement("div");
  speedScale.className = "range-scale";

  const slow = document.createElement("span");
  slow.textContent = `${speedInput.min}\u00d7 slower`;

  const fast = document.createElement("span");
  fast.textContent = `${speedInput.max}\u00d7 faster`;

  speedScale.append(slow, fast);
  speedField.append(speedLabelRow, speedInput, speedScale);

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
        updateSpeedLabel();
      }

      setBusy(gate.isRunning);
    },
  });

  const pronunciationToggle = createPronunciationToggle();

  const speedNote = document.createElement("p");
  speedNote.className = "field-note";
  speedNote.id = "speed-note";
  speedNote.hidden = true;
  speedNote.textContent =
    "Speed is only available for the Kokoro engine. " +
    "The VoiceForge custom model always uses its native rate.";

  // Actions
  const actions = document.createElement("div");
  actions.className = "composer-actions";

  const spinner = document.createElement("span");
  spinner.className = "spinner";
  spinner.hidden = true;

  const generateButton = document.createElement("button");
  generateButton.type = "button";
  generateButton.className = "button button-primary";
  generateButton.id = "generate-button";
  generateButton.append(spinner, document.createTextNode("Generate speech"));

  actions.append(generateButton);

  const examples = document.createElement("div");
  examples.className = "chip-row";

  for (const example of EXAMPLES) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = example;
    chip.addEventListener("click", () => {
      textarea.value = example;
      updateCounter();
      textarea.focus();
    });
    examples.append(chip);
  }

  composerBody.append(
    modeBar,
    textField,
    metaRow,
    selectors,
    speedField,
    speedNote,
    presetField.element,
    pronunciationToggle.element,
    examples,
    actions,
  );
  composer.append(composerBody);

  // --- Output -----------------------------------------------------------
  const audioPanel = createAudioPanel();
  const jobView = createJobView({ title: "Long text progress" });

  const side = document.createElement("div");
  side.className = "stack";
  side.append(audioPanel.element);

  grid.append(composer, side);
  wrapper.append(grid, jobView.element);

  // --- Mode -------------------------------------------------------------
  function currentLimit() {
    return mode === "long" ? MAX_ENTRY_CHARS : shortLimit;
  }

  function updateModeUi() {
    const long = mode === "long";

    textarea.maxLength = currentLimit();
    textarea.placeholder = long
      ? "Paste a long text. It will be split into sentences and synthesised as a background job."
      : "Type or paste a sentence, for example: Welcome to VoiceForge.";

    hint.textContent = long
      ? "Ctrl + Enter to generate \u00b7 Enter for a new line"
      : "Press Enter to generate \u00b7 Shift + Enter for a new line";

    generateButton.lastChild.textContent = long
      ? "Generate long text"
      : "Generate speech";

    shortModeButton.classList.toggle("is-active", !long);
    longModeButton.classList.toggle("is-active", long);

    examples.hidden = long;

    if (!long) {
      jobView.clear();
    }

    updateCounter();
  }

  shortModeButton.addEventListener("click", () => {
    mode = "short";
    updateModeUi();
  });

  longModeButton.addEventListener("click", () => {
    mode = "long";
    updateModeUi();
  });

  // --- Behaviour --------------------------------------------------------
  function syncVisibility() {
    const engineId = engineSelect.value;
    const voiceEnabled = voiceControlsVisible(engineId);
    const speedEnabled = speedControlVisible(engineId);

    voiceField.hidden = !voiceEnabled;
    voiceSelect.disabled = !voiceEnabled;
    speedField.hidden = !speedEnabled;
    speedInput.disabled = !speedEnabled;
    speedNote.hidden = speedEnabled;
  }

  function updateSpeedLabel() {
    speedValue.textContent = formatSpeed(speedInput.value);
  }

  function updateCounter() {
    const count = countCharacters(textarea.value);
    const limit = currentLimit();
    const state = counterState(count, limit);

    counter.textContent = `${count} / ${limit}`;
    counter.classList.toggle("is-warning", state === "warning");
    counter.classList.toggle("is-over", state === "over");
  }

  function setBusy(busy) {
    generateButton.disabled = busy;
    spinner.hidden = !busy;
    engineSelect.disabled = busy;
    voiceSelect.disabled = busy || !voiceControlsVisible(engineSelect.value);
    speedInput.disabled = busy || !speedControlVisible(engineSelect.value);
    textarea.readOnly = busy;
    shortModeButton.disabled = busy;
    longModeButton.disabled = busy;

    if (!busy) {
      generateButton.lastChild.textContent =
        mode === "long" ? "Generate long text" : "Generate speech";
    } else {
      generateButton.lastChild.textContent = "Working\u2026";
    }
  }

  async function startShortGeneration(engine, voice, speed) {
    try {
      const result = await generateSpeech({
        text: textarea.value,
        engine,
        voice,
        speed,
        pronunciation: pronunciationToggle.enabled,
      });

      audioPanel.show(result, {
        engineLabel: engines[result.engine]?.name || result.engine,
        voiceLabel: describeVoice(result.engine, result.voice, voiceLabels),
        speedLabel: describeSpeed(result.engine, result.speed),
      });

      notifications.success(
        `Generated ${result.duration_seconds.toFixed(2)} s of audio.`,
      );
    } catch (error) {
      notifications.error(
        error instanceof ApiError
          ? error.message
          : "Speech generation failed unexpectedly.",
      );
    }
  }

  async function startLongGeneration(engine, voice, speed) {
    const text = textarea.value.trim();

    if (!text) {
      notifications.error("Enter some text before generating speech.");
      return;
    }

    if (text.length > MAX_ENTRY_CHARS) {
      notifications.error(
        `Text is too long (${text.length} characters). The limit is ${MAX_ENTRY_CHARS}.`,
      );
      return;
    }

    const payload = {
      type: "long_text",
      engine,
      pronunciation: pronunciationToggle.enabled,
      items: [{ name: "Long text", text }],
    };

    if (voice) {
      payload.voice = voice;
    }

    if (typeof speed === "number") {
      payload.speed = speed;
    }

    jobView.clear();

    try {
      const job = await submitJob(payload);
      jobView.render(job);
      notifications.info("Long-text job queued.");

      poller = createJobPoller({
        jobId: job.id,
        onUpdate: (updated) => {
          jobView.render(updated);

          if (updated.status === "completed") {
            notifications.success(
              `Long text finished: ${updated.completed} of ${updated.total} items.`,
            );
            setBusy(false);
          } else if (updated.status === "failed") {
            notifications.error(updated.error || "Long-text generation failed.");
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
        error instanceof ApiError ? error.message : "Could not start the job.",
      );
      setBusy(false);
    }
  }

  /**
   * The single generation entry point. The button and the keyboard shortcuts
   * both call this; the gate guarantees one request at a time.
   */
  async function startGeneration() {
    if (!gate.tryStart()) {
      return;
    }

    const engine = engineSelect.value;
    const voice = voiceForRequest(engine, voiceSelect.value);
    const speed = speedForRequest(engine, speedInput.value);

    if (mode === "long") {
      setBusy(true);

      try {
        await startLongGeneration(engine, voice, speed);
      } finally {
        gate.finish();
        // Long-text jobs re-enable the UI when the job finishes; the job
        // view keeps polling in the background.
        if (!poller) {
          setBusy(false);
        }
      }

      return;
    }

    const validation = validateText(textarea.value, shortLimit);

    if (!validation.ok) {
      gate.finish();
      notifications.error(validation.error);
      textarea.focus();
      return;
    }

    setBusy(true);

    try {
      await startShortGeneration(engine, voice, speed);
    } finally {
      gate.finish();
      setBusy(false);
    }
  }

  textarea.addEventListener("input", updateCounter);
  speedInput.addEventListener("input", updateSpeedLabel);

  textarea.addEventListener("keydown", (event) => {
    if (mode === "long") {
      const action = resolveLongTextKey(event, textarea.value);

      if (action === KeyAction.NEWLINE) {
        return; // Enter inserts a newline
      }

      if (action === KeyAction.GENERATE) {
        event.preventDefault();
        startGeneration();
      }

      return;
    }

    const action = resolveKeyAction(event, textarea.value);

    if (action === KeyAction.NEWLINE || event.isComposing) {
      return;
    }

    if (event.key !== "Enter") {
      return;
    }

    if (event.ctrlKey || event.metaKey || event.altKey) {
      return;
    }

    event.preventDefault();
    startGeneration();
  });

  generateButton.addEventListener("click", startGeneration);

  engineSelect.addEventListener("change", () => {
    syncVisibility();
    setBusy(gate.isRunning);
  });

  syncVisibility();
  updateSpeedLabel();
  updateModeUi();

  return {
    element: wrapper,
    startGeneration,

    deactivate() {
      if (poller) {
        poller.stop();
        poller = null;
      }
    },
  };
}
