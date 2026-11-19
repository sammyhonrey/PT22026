/**
 * Shared controls used by Speech Studio, Batch Studio and Documents:
 * a voice-preset selector and a pronunciation toggle.
 */

import { listPresets } from "../advanced-api.js";
import { applyPresetToControls, describePreset } from "../advanced-state.js";

export function createPresetSelect({ capabilities, onApply, notifications }) {
  const field = document.createElement("div");
  field.className = "field";

  const label = document.createElement("label");
  label.className = "field-label";
  label.textContent = "Voice preset";

  const select = document.createElement("select");
  select.className = "select";
  select.id = "preset-select";

  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "No preset";
  select.append(placeholder);

  const hint = document.createElement("span");
  hint.className = "field-note";
  hint.hidden = true;

  field.append(label, select, hint);

  let presets = [];

  async function refresh() {
    try {
      presets = await listPresets();
    } catch {
      presets = [];
    }

    select.replaceChildren(placeholder);

    for (const preset of presets) {
      const option = document.createElement("option");
      option.value = preset.id;
      option.textContent = preset.name;
      select.append(option);
    }

    hint.hidden = presets.length > 0;
    hint.textContent = "No presets saved yet. Create one on the Voice Presets page.";
  }

  select.addEventListener("change", () => {
    const preset = presets.find((entry) => entry.id === select.value);

    if (!preset) {
      return;
    }

    const values = applyPresetToControls(preset, capabilities);

    if (!values) {
      notifications?.error("This preset can no longer be applied.");
      return;
    }

    onApply(values, preset);
    notifications?.success(`Preset applied: ${describePreset(preset, capabilities)}`);
  });

  refresh();

  return { element: field, refresh };
}

export function createPronunciationToggle({ labelText } = {}) {
  const wrapper = document.createElement("label");
  wrapper.className = "checkbox-row";

  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = true;
  input.id = "pronunciation-toggle";

  const text = document.createElement("span");
  text.textContent =
    labelText || "Use pronunciation dictionary";

  wrapper.append(input, text);

  return {
    element: wrapper,
    get enabled() {
      return input.checked;
    },
  };
}
