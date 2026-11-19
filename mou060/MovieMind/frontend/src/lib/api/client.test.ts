import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";

import { fetchMovies } from "@/features/discovery/api";
import { apiBaseUrl, apiClient } from "@/lib/api/client";
import { authSession } from "@/lib/auth/session";
import { server } from "@/test/msw/server";

describe("apiClient", () => {
  it("attaches the in-memory bearer token to requests", async () => {
    authSession.setToken("test-token");
    let authorization: string | null = null;
    server.use(
      http.get(`${apiBaseUrl}/movies`, ({ request }) => {
        authorization = request.headers.get("authorization");
        return HttpResponse.json({ items: [], limit: 20, offset: 0, total: 0 });
      }),
    );

    await fetchMovies({ limit: 20, offset: 0 });

    expect(authorization).toBe("Bearer test-token");
  });

  it("clears the in-memory session after a 401 response", async () => {
    authSession.setToken("expired-token");
    server.use(
      http.get(`${apiBaseUrl}/movies`, () =>
        HttpResponse.json({ code: "unauthorized", detail: "Token expired" }, { status: 401 }),
      ),
    );

    await expect(fetchMovies({ limit: 20, offset: 0 })).rejects.toMatchObject({
      code: "unauthorized",
      status: 401,
    });
    expect(authSession.getToken()).toBeNull();
  });

  it("shares one refresh across concurrent 401 responses and retries both requests", async () => {
    authSession.setSession("expired-token", "old-csrf");
    let refreshCount = 0;
    let csrfHeader: string | null = null;
    server.use(
      http.get(`${apiBaseUrl}/auth/csrf`, () => HttpResponse.json({ csrf_token: "refresh-csrf" })),
      http.post(`${apiBaseUrl}/auth/refresh`, ({ request }) => {
        refreshCount += 1;
        csrfHeader = request.headers.get("x-csrf-token");
        return HttpResponse.json({
          access_token: "new-token",
          csrf_token: "new-csrf",
          expires_in: 1800,
          token_type: "bearer",
        });
      }),
      http.get(`${apiBaseUrl}/movies`, ({ request }) => {
        if (request.headers.get("authorization") === "Bearer expired-token") {
          return HttpResponse.json({ code: "unauthorized", detail: "Expired" }, { status: 401 });
        }
        return HttpResponse.json({ items: [], limit: 20, offset: 0, total: 0 });
      }),
    );

    await Promise.all([apiClient.get("/movies"), apiClient.get("/movies")]);

    expect(refreshCount).toBe(1);
    expect(csrfHeader).toBe("refresh-csrf");
    expect(authSession.getToken()).toBe("new-token");
    expect(apiClient.defaults.withCredentials).toBe(true);
  });

  it("maps a transport failure to a safe retryable error", async () => {
    server.use(http.get(`${apiBaseUrl}/movies`, () => HttpResponse.error()));

    await expect(fetchMovies({ limit: 20, offset: 0 })).rejects.toMatchObject({
      message: "We could not reach MovieMind. Check your connection and try again.",
      status: null,
    });
  });

  it("surfaces the Retry-After delay from a 429 response", async () => {
    server.use(
      http.get(`${apiBaseUrl}/movies`, () =>
        HttpResponse.json(
          { code: "rate_limited", detail: "Slow down" },
          { headers: { "retry-after": "60" }, status: 429 },
        ),
      ),
    );

    await expect(fetchMovies({ limit: 20, offset: 0 })).rejects.toMatchObject({
      retryAfterSeconds: 60,
      status: 429,
    });
  });
});
