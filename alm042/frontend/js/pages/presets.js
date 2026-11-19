/**
 * Voice Presets page: create, edit and delete reusable voice settings.
 */

import { ApiError } from "../api.js";
import {
  createPreset,
  deletePreset,
  listPresets,
  updatePreset,
} from "../advanced-api.js";
import {
  describePreset,
  validatePresetDraft,
} from "../advanced-state.js";
import { formatSpeed, speedControlVisible } from "../engine-ui.js";

export function createPresetsPage({ capabilities, notifications }) {
  const engines = capabilities.engines || {};
  const kokoro = engines.kokoro || {};
  const voices = kokoro.voices || [];
  const speedRange = kokoro.speed || {};

  const panel = document.createElement("section");
  panel.className = "panel";

  const head = document.createElement("div");
  head.className = "panel-head";
  head.innerHTML =
    '<h2 class="panel-title">Voice presets</h2><span class="panel-hint">engine · voice · speed</span>';

  const body = document.createElement("div");
  body.className = "panel-body stack";

  // --- Form -------------------------------------------------------------
  const form = document.createElement("form");
  form.className = "preset-form";

  const name = document.createElement("input");
  name.className = "input";
  name.placeholder = "Preset name";
  name.maxLength = 60;

  const engine = document.createElement("select");
  engine.className = "select";

  for (const [id, info] of Object.entries(engines)) {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = info.name || id;
    engine.append(option);
  }

  const voice = document.createElement("select");
  voice.className = "select";

  for (const entry of voices) {
    const option = document.createElement("option");
    option.value = entry.id;
    option.textContent = entry.name;
    voice.append(option);
  }

  const defaultVoice = voices.find((entry) => entry.default);

  if (defaultVoice) {
    voice.value = defaultVoice.id;
  }

  const speedField = document.createElement("div");
  speedField.className = "preset-speed";

  const speed = document.createElement("input");
  speed.type = "range";
  speed.className = "range";
  speed.min = speedRange.min ?? 0.5;
  speed.max = speedRange.max ?? 2;
  speed.step = "0.05";
  speed.value = speedRange.default ?? 1;

  const speedValue = document.createElement("span");
  speedValue.className = "badge badge-accent";
  speedValue.textContent = formatSpeed(speed.value);

  speedField.append(speed, speedValue);

  const description = document.createElement("input");
  description.className = "input";
  description.placeholder = "Description (optional)";
  description.maxLength = 200;

  const submit = document.createElement("button");
  submit.type = "submit";
  submit.className = "button button-primary";
  submit.textContent = "Save preset";

  form.append(name, engine, voice, speedField, description, submit);

  const list = document.createElement("div");
  list.className = "preset-list";

  body.append(form, list);
  panel.append(head, body);

  const wrapper = document.createElement("div");
  wrapper.append(panel);

  let presets = [];
  let editingId = null;

  function draft() {
    return {
      name: name.value,
      engine: engine.value,
      voice: voice.value,
      speed: Number(speed.value),
    };
  }

  function syncForm() {
    const supportsVoice = voiceFieldVisible(engine.value);
    voice.hidden = !supportsVoice;
    speedField.hidden = !speedControlVisible(engine.value);
  }

  function voiceFieldVisible(engineId) {
    return (engines[engineId]?.voices || []).some((entry) => entry.id);
  }

  function renderRow(preset) {
    const row = document.createElement("div");
    row.className = "preset-row";

    const info = document.createElement("div");
    info.className = "preset-info";

    const title = document.createElement("span");
    title.className = "preset-name";
    title.textContent = preset.name;

    const meta = document.createElement("span");
    meta.className = "preset-meta";
    meta.textContent = describePreset(preset, capabilities);

    info.append(title, meta);

    const actions = document.createElement("div");
    actions.className = "library-card-actions";

    const use = document.createElement("button");
    use.type = "button";
    use.className = "button button-ghost button-small";
    use.textContent = "Open Studio";
    use.addEventListener("click", () => {
      const navItem = document.querySelector('.nav-item[data-page="studio"]');

      if (navItem) {
        navItem.click();
      }
    });

    const edit = document.createElement("button");
    edit.type = "button";
    edit.className = "button button-ghost button-small";
    edit.textContent = "Edit";
    edit.addEventListener("click", () => {
      editingId = preset.id;
      name.value = preset.name;
      engine.value = preset.engine;
      syncForm();

      if (preset.voice) {
        voice.value = preset.voice;
      }

      if (typeof preset.speed === "number") {
        speed.value = preset.speed;
        speedValue.textContent = formatSpeed(preset.speed);
      }

      description.value = preset.description || "";
      submit.textContent = "Update preset";
      window.scrollTo({ top: 0, behavior: "smooth" });
    });

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "button button-ghost button-small is-danger";
    remove.textContent = "Delete";
    remove.addEventListener("click", async () => {
      if (!window.confirm(`Delete the preset "${preset.name}"?`)) {
        return;
      }

      try {
        await deletePreset(preset.id);
        notifications.success("Preset deleted.");
        await load();
      } catch (error) {
        notifications.error(
          error instanceof ApiError ? error.message : "Delete failed.",
        );
      }
    });

    actions.append(use, edit, remove);
    row.append(info, actions);

    return row;
  }

  function render() {
    list.replaceChildren();

    if (presets.length === 0) {
      const empty = document.createElement("div");
      empty.className = "output-empty";
      empty.innerHTML =
        '<p>No presets yet.</p><p class="meta-key">Save a voice you like and reuse it everywhere</p>';
      list.append(empty);
      return;
    }

    list.append(...presets.map(renderRow));
  }

  async function load() {
    try {
      presets = await listPresets();
      render();
    } catch (error) {
      notifications.error(
        error instanceof ApiError ? error.message : "Could not load presets.",
      );
    }
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const validation = validatePresetDraft(draft(), capabilities);

    if (!validation.ok) {
      notifications.error(validation.error);
      return;
    }

    try {
      if (editingId) {
        await updatePreset(editingId, {
          ...validation.preset,
          description: description.value,
        });
        notifications.success("Preset updated.");
        editingId = null;
        submit.textContent = "Save preset";
      } else {
        await createPreset({
          ...validation.preset,
          description: description.value,
        });
        notifications.success("Preset saved.");
      }

      form.reset();
      speed.value = speedRange.default ?? 1;
      speedValue.textContent = formatSpeed(speed.value);
      await load();
    } catch (error) {
      notifications.error(
        error instanceof ApiError ? error.message : "Could not save the preset.",
      );
    }
  });

  engine.addEventListener("change", syncForm);
  speed.addEventListener("input", () => {
    speedValue.textContent = formatSpeed(speed.value);
  });

  syncForm();
  load();

  return { element: wrapper, refresh: load };
}
