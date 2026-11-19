type SessionListener = (accessToken: string | null) => void;

let accessToken: string | null = null;
let csrfToken: string | null = null;
const listeners = new Set<SessionListener>();

function notify() {
  listeners.forEach((listener) => listener(accessToken));
}

/**
 * Access and CSRF tokens stay in memory. Persistent renewal is handled only by
 * the backend's HttpOnly refresh cookie.
 */
export const authSession = {
  clear() {
    if (accessToken === null && csrfToken === null) return;
    accessToken = null;
    csrfToken = null;
    notify();
  },

  getCsrfToken() {
    return csrfToken;
  },

  getToken() {
    return accessToken;
  },

  setCsrfToken(token: string | null) {
    csrfToken = token;
  },

  setSession(token: string, csrf: string) {
    accessToken = token;
    csrfToken = csrf;
    notify();
  },

  setToken(token: string) {
    accessToken = token;
    notify();
  },

  subscribe(listener: SessionListener) {
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  },
};
