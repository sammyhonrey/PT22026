import { AxiosError, AxiosHeaders } from "axios";
import { describe, expect, it } from "vitest";

import { isApiClientError, toApiClientError } from "@/lib/api/errors";

function axiosError(status: number, data: unknown, headers: Record<string, string> = {}) {
  return new AxiosError("Request failed", "ERR_BAD_RESPONSE", undefined, undefined, {
    config: { headers: new AxiosHeaders() },
    data,
    headers,
    status,
    statusText: "Error",
  });
}

describe("toApiClientError", () => {
  it.each([
    [401, "unauthorized", "Your session has expired. Please sign in again."],
    [403, "forbidden", "You do not have permission to do that."],
    [404, "not_found", "We could not find what you were looking for."],
    [409, "conflict", "That information is already in use. Please choose another value."],
    [422, "validation_error", "Please check the information you entered and try again."],
    [503, "database_unavailable", "MovieMind is temporarily unavailable. Please try again shortly."],
    [500, "internal_error", "Something went wrong on our side. Please try again shortly."],
  ])("maps %i %s to a safe message", (status, code, message) => {
    const error = toApiClientError(axiosError(status, { code, detail: "internal details" }));

    expect(error).toMatchObject({ code, message, status });
    expect(error.message).not.toContain("internal details");
  });

  it("exposes the retry delay for rate limiting", () => {
    const error = toApiClientError(axiosError(429, { code: "rate_limited", detail: "slow down" }, { "retry-after": "60" }));

    expect(error).toMatchObject({ retryAfterSeconds: 60, status: 429 });
    expect(error.message).toContain("Too many requests");
  });

  it("maps a network failure without a response", () => {
    const error = toApiClientError(new AxiosError("Network Error", "ERR_NETWORK"));

    expect(error).toMatchObject({ code: null, retryAfterSeconds: null, status: null });
    expect(error.message).toBe("We could not reach MovieMind. Check your connection and try again.");
  });

  it("preserves a specific Axios transport message instead of hiding it behind the generic fallback", () => {
    const error = toApiClientError(new AxiosError("Request timed out while contacting the API", "ECONNABORTED"));

    expect(error).toMatchObject({ code: null, retryAfterSeconds: null, status: null });
    expect(error.message).toBe("Request timed out while contacting the API");
  });

  it("preserves a non-Axios error message instead of hiding it behind a network fallback", () => {
    const error = toApiClientError(new Error("raw"));

    expect(error).toMatchObject({ code: null, message: "raw", status: null });
    expect(isApiClientError(error)).toBe(true);
    expect(isApiClientError(new Error("raw"))).toBe(false);
  });
});
