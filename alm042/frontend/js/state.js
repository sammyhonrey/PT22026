/**
 * Minimal observable state container and the generation guard.
 *
 * No framework is used; pages subscribe to the store and re-render the small
 * parts they own.
 */

/**
 * @param {Record<string, unknown>} initialState
 */
export function createStore(initialState = {}) {
  let state = { ...initialState };
  const listeners = new Set();

  return {
    get() {
      return state;
    },

    set(patch) {
      state = { ...state, ...patch };

      for (const listener of listeners) {
        listener(state);
      }
    },

    subscribe(listener) {
      listeners.add(listener);
      listener(state);

      return () => listeners.delete(listener);
    },
  };
}

/**
 * Prevents duplicate generation requests.
 *
 * Both the Enter key and the button call the same `startGeneration()`, and
 * that function asks this gate first. If a request is already in flight the
 * gate returns false and nothing else happens.
 */
export function createGenerationGate() {
  let running = false;

  return {
    get isRunning() {
      return running;
    },

    tryStart() {
      if (running) {
        return false;
      }

      running = true;
      return true;
    },

    finish() {
      running = false;
    },
  };
}
