import { AxiosError, isAxiosError } from "axios";

import type { ApiErrorResponse } from "@/types/api";

export type ApiClientError = {
  code: string | null;
  message: string;
  retryAfterSeconds: number | null;
  status: number | null;
};

function isApiErrorResponse(value: unknown): value is ApiErrorResponse {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Record<string, unknown>;
  return typeof candidate.detail === "string" && typeof candidate.code === "string";
}

function safeMessage(status: number | null, code: string | null) {
  if (code === "rate_limited" || status === 429) return "Too many requests. Please wait a moment and try again.";
  if (code === "unauthorized" || status === 401) return "Your session has expired. Please sign in again.";
  if (code === "forbidden" || status === 403) return "You do not have permission to do that.";
  if (code === "not_found" || status === 404) return "We could not find what you were looking for.";
  if (code === "conflict" || status === 409) return "That information is already in use. Please choose another value.";
  if (code === "validation_error" || status === 422) return "Please check the information you entered and try again.";
  if (code === "database_unavailable" || code === "model_unavailable" || status === 503) return "MovieMind is temporarily unavailable. Please try again shortly.";
  if (status !== null && status >= 500) return "Something went wrong on our side. Please try again shortly.";
  return "We could not reach MovieMind. Check your connection and try again.";
}

function retryAfterSeconds(error: AxiosError) {
  const value = error.response?.headers?.["retry-after"];
  const parsed = typeof value === "string" ? Number.parseInt(value, 10) : Number.NaN;
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : null;
}

/** Convert transport failures to a safe error object for UI components. */
export function toApiClientError(error: unknown): ApiClientError {
  if (isAxiosError(error)) {
    const status = error.response?.status ?? null;
    const payload = error.response?.data;
    const code = isApiErrorResponse(payload) ? payload.code : null;
    const rawMessage = error.message.trim();

    const message =
      status === null && rawMessage && !/^(network error|failed to fetch)$/i.test(rawMessage)
        ? rawMessage
        : safeMessage(status, code);

    return {
      code,
      message,
      retryAfterSeconds: retryAfterSeconds(error),
      status,
    };
  }

  if (error instanceof Error) {
    const message = error.message.trim();
    return {
      code: null,
      message: message || safeMessage(null, null),
      retryAfterSeconds: null,
      status: null,
    };
  }

  return { code: null, message: safeMessage(null, null), retryAfterSeconds: null, status: null };
}

export function isApiClientError(error: unknown): error is ApiClientError {
  if (typeof error !== "object" || error === null) return false;
  const candidate = error as Record<string, unknown>;
  return typeof candidate.message === "string" && "status" in candidate && "code" in candidate;
}
