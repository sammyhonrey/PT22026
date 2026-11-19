import { render, screen } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { describe, expect, it } from "vitest";

import { createQueryClient, QueryProvider } from "@/app/providers";
import { AuthProvider, useAuth } from "@/features/auth/AuthProvider";
import { RequireAuth } from "@/features/auth/RequireAuth";
import { apiBaseUrl } from "@/lib/api/client";
import { server } from "@/test/msw/server";
import { testUser } from "@/test/fixtures";

function LoginRoute() {
  const location = useLocation();
  const from = (location.state as { from?: { pathname?: string } } | null)?.from?.pathname;
  return <p>Login route from {from ?? "none"}</p>;
}

function PrivateRoute() {
  const { isAuthenticated, user } = useAuth();
  return <p>{isAuthenticated ? `Private: ${user?.email}` : "Private route"}</p>;
}

function renderProtectedRoute() {
  const client = createQueryClient();
  return render(
    <QueryProvider client={client}>
      <AuthProvider>
        <MemoryRouter initialEntries={["/watchlist"]}>
          <Routes>
            <Route element={<LoginRoute />} path="/login" />
            <Route
              element={
                <RequireAuth>
                  <PrivateRoute />
                </RequireAuth>
              }
              path="/watchlist"
            />
          </Routes>
        </MemoryRouter>
      </AuthProvider>
    </QueryProvider>,
  );
}

describe("AuthProvider restoration", () => {
  it("restores a protected route from the refresh cookie before rendering it", async () => {
    let refreshCsrfHeader: string | null = null;
    server.use(
      http.get(`${apiBaseUrl}/auth/csrf`, () => HttpResponse.json({ csrf_token: "cookie-csrf" })),
      http.post(`${apiBaseUrl}/auth/refresh`, ({ request }) => {
        refreshCsrfHeader = request.headers.get("x-csrf-token");
        return HttpResponse.json({
          access_token: "restored-token",
          csrf_token: "rotated-csrf",
          expires_in: 1800,
          token_type: "bearer",
        });
      }),
      http.get(`${apiBaseUrl}/users/me`, () => HttpResponse.json(testUser)),
    );

    renderProtectedRoute();

    expect(await screen.findByText(`Private: ${testUser.email}`)).toBeInTheDocument();
    expect(screen.queryByText(/Login route/)).not.toBeInTheDocument();
    expect(refreshCsrfHeader).toBe("cookie-csrf");
  });

  it("redirects unauthenticated users only after restoration finishes and preserves destination", async () => {
    let refreshCalls = 0;
    server.use(
      http.get(`${apiBaseUrl}/auth/csrf`, () => HttpResponse.json({ csrf_token: null })),
      http.post(`${apiBaseUrl}/auth/refresh`, () => {
        refreshCalls += 1;
        return HttpResponse.json({ code: "unauthorized", detail: "No session" }, { status: 401 });
      }),
    );

    renderProtectedRoute();

    expect(await screen.findByText("Login route from /watchlist")).toBeInTheDocument();
    expect(refreshCalls).toBe(0);
  });
});
