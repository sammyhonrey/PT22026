/**
 * Engine-related UI rules, kept pure so they can be tested without a DOM.
 */

export const ENGINE_KOKORO = "kokoro";
export const ENGINE_VOICEFORGE = "voiceforge";

/** Only the Kokoro engine has selectable voices. */
export function voiceControlsVisible(engineId) {
  return engineId === ENGINE_KOKORO;
}

/** The voice is only sent to the backend for Kokoro. */
export function voiceForRequest(engineId, voiceId) {
  return engineId === ENGINE_KOKORO ? voiceId || null : null;
}

/** Only the Kokoro engine supports speed control. */
export function speedControlVisible(engineId) {
  return engineId === ENGINE_KOKORO;
}

/** The speed is only sent to the backend for Kokoro. */
export function speedForRequest(engineId, speed) {
  if (engineId !== ENGINE_KOKORO) {
    return null;
  }

  // An empty slider value must not become 0 through Number("").
  if (speed === null || speed === undefined || speed === "") {
    return null;
  }

  const value = Number(speed);

  return Number.isFinite(value) ? value : null;
}

/** "1.00×" style label for the slider and the metadata. */
export function formatSpeed(speed) {
  const value = Number(speed);

  if (!Number.isFinite(value)) {
    return "1.00×";
  }

  return `${value.toFixed(2)}×`;
}

/** Human label for the voice shown in the output metadata. */
export function describeVoice(engineId, voiceId, voiceLabels = {}) {
  if (engineId !== ENGINE_KOKORO) {
    return "not applicable";
  }

  return voiceLabels[voiceId] || voiceId || "default";
}

/** Human label for the speed shown in the output metadata. */
export function describeSpeed(engineId, speed) {
  if (engineId !== ENGINE_KOKORO) {
    return "not applicable";
  }

  return formatSpeed(speed);
}
