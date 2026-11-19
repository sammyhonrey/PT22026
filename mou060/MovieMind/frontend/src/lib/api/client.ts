import axios, { type InternalAxiosRequestConfig } from "axios";

import { authSession } from "@/lib/auth/session";
import { toApiClientError } from "@/lib/api/errors";
import type { TokenResponse } from "@/types/api";

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim() || "http://localhost:8000";
export const apiBaseUrl = `${configuredBaseUrl.replace(/\/$/, "")}/api/v1`;

/**
 * The only shared HTTP client. Feature modules own their endpoint functions;
 * this module owns base URL, bearer auth, timeout, and common failures.
 */
export const apiClient = axios.create({
  baseURL: apiBaseUrl,
  headers: { Accept: "application/json", "Content-Type": "application/json" },
  timeout: 12_000,
  withCredentials: true,
});

type RetriableRequestConfig = InternalAxiosRequestConfig & { _authRetry?: boolean };

let refreshPromise: Promise<string | null> | null = null;

function isAuthRequest(url: string | undefined) {
  return Boolean(url && /\/auth\/(login|register|csrf|refresh|logout)(?:[?#]|$)/.test(url));
}

async function fetchCsrfToken() {
  const { data } = await apiClient.get<{ csrf_token: string | null }>("/auth/csrf");
  authSession.setCsrfToken(data.csrf_token);
  return data.csrf_token;
}

async function performRefresh(): Promise<string | null> {
  const csrfToken = await fetchCsrfToken();
  if (!csrfToken) return null;
  const { data } = await apiClient.post<TokenResponse>("/auth/refresh");
  authSession.setSession(data.access_token, data.csrf_token);
  return data.access_token;
}

export function refreshAccessToken() {
  if (!refreshPromise) {
    refreshPromise = performRefresh().finally(() => {
      refreshPromise = null;
    });
  }
  return refreshPromise;
}

apiClient.interceptors.request.use((config) => {
  if (!isAuthRequest(config.url)) {
    const token = authSession.getToken();
    if (token) config.headers.Authorization = `Bearer ${token}`;
  }
  if (/\/auth\/(refresh|logout)(?:[?#]|$)/.test(config.url ?? "")) {
    const csrfToken = authSession.getCsrfToken();
    if (csrfToken) config.headers["X-CSRF-Token"] = csrfToken;
  }
  return config;
});

apiClient.interceptors.response.use(
  (response) => response,
  async (error: unknown) => {
    const normalized = toApiClientError(error);
    if (!axios.isAxiosError(error)) return Promise.reject(normalized);

    const config = error.config as RetriableRequestConfig | undefined;
    if (
      normalized.status !== 401 ||
      !config ||
      config._authRetry ||
      isAuthRequest(config.url)
    ) {
      if (normalized.status === 401 && config?._authRetry) authSession.clear();
      return Promise.reject(normalized);
    }

    config._authRetry = true;
    try {
      const token = await refreshAccessToken();
      if (!token) {
        authSession.clear();
        return Promise.reject(normalized);
      }
      config.headers.Authorization = `Bearer ${token}`;
      return await apiClient(config);
    } catch (refreshError) {
      const refreshStatus =
        typeof refreshError === "object" && refreshError !== null && "status" in refreshError
          ? refreshError.status
          : null;
      if (refreshStatus === 401 || refreshStatus === 403) authSession.clear();
      return Promise.reject(refreshError);
    }
  },
);
