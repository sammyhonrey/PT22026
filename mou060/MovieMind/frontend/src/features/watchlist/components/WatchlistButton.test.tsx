import { act, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useAuth } from "@/features/auth/AuthProvider";
import type { AuthContextValue } from "@/features/auth/AuthProvider";
import { WatchlistButton } from "@/features/watchlist/components/WatchlistButton";
import { apiBaseUrl } from "@/lib/api/client";
import { dune, makeWatchlistItem, pageOf, testUser } from "@/test/fixtures";
import { server } from "@/test/msw/server";
import { renderWithProviders } from "@/test/test-utils";

vi.mock("@/features/auth/AuthProvider", () => ({ useAuth: vi.fn() }));

const mockedUseAuth = vi.mocked(useAuth);

function authValue(isAuthenticated = true): AuthContextValue {
  return {
    isAuthenticated,
    initializationError: null,
    isLoading: false,
    retryInitialization: vi.fn(),
    signIn: vi.fn(async () => testUser),
    signOut: vi.fn(async () => undefined),
    signUp: vi.fn(async () => testUser),
    user: isAuthenticated ? testUser : null,
  };
}

function installStatefulWatchlist(initiallySaved: boolean) {
  let saved = initiallySaved;
  server.use(
    http.get(`${apiBaseUrl}/watchlist`, () =>
      HttpResponse.json(pageOf(saved ? [makeWatchlistItem(dune)] : [], 100)),
    ),
    http.post(`${apiBaseUrl}/watchlist/:movieId`, () => {
      saved = true;
      return HttpResponse.json(makeWatchlistItem(dune), { status: 201 });
    }),
    http.delete(`${apiBaseUrl}/watchlist/:movieId`, () => {
      saved = false;
      return new HttpResponse(null, { status: 204 });
    }),
  );
}

beforeEach(() => mockedUseAuth.mockReset());

describe("WatchlistButton", () => {
  it("links guests to sign in", () => {
    mockedUseAuth.mockReturnValue(authValue(false));
    renderWithProviders(<WatchlistButton movieId={dune.id} />);

    expect(screen.getByRole("link", { name: "Sign in to save" })).toHaveAttribute("href", "/login");
  });

  it("keeps the compact action accessible and named for the movie", () => {
    mockedUseAuth.mockReturnValue(authValue(false));
    renderWithProviders(<WatchlistButton compact movieId={dune.id} movieTitle={dune.title} />);

    expect(screen.getByRole("link", { name: `Sign in to save ${dune.title}` })).toHaveAttribute(
      "href",
      "/login",
    );
  });

  it("adds a movie and reflects the refreshed saved state", async () => {
    const user = userEvent.setup();
    mockedUseAuth.mockReturnValue(authValue());
    installStatefulWatchlist(false);

    renderWithProviders(<WatchlistButton movieId={dune.id} />);
    await user.click(await screen.findByRole("button", { name: "Add to My List" }));

    expect(await screen.findByRole("button", { name: "Saved to My List" })).toBeInTheDocument();
    expect(await screen.findByRole("status")).toHaveTextContent("Added to your list.");
  });

  it("removes a saved movie", async () => {
    const user = userEvent.setup();
    mockedUseAuth.mockReturnValue(authValue());
    installStatefulWatchlist(true);

    renderWithProviders(<WatchlistButton movieId={dune.id} />);
    await user.click(await screen.findByRole("button", { name: "Saved to My List" }));

    expect(await screen.findByRole("button", { name: "Add to My List" })).toBeInTheDocument();
  });

  it("disables the control while a save request is pending", async () => {
    const user = userEvent.setup();
    mockedUseAuth.mockReturnValue(authValue());
    let release!: () => void;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    let saved = false;
    server.use(
      http.get(`${apiBaseUrl}/watchlist`, () => HttpResponse.json(pageOf(saved ? [makeWatchlistItem(dune)] : [], 100))),
      http.post(`${apiBaseUrl}/watchlist/:movieId`, async () => {
        await gate;
        saved = true;
        return HttpResponse.json(makeWatchlistItem(dune), { status: 201 });
      }),
    );

    renderWithProviders(<WatchlistButton movieId={dune.id} />);
    const button = await screen.findByRole("button", { name: "Add to My List" });
    await user.click(button);

    await waitFor(() => expect(button).toBeDisabled());
    await act(async () => { release(); });
    expect(await screen.findByRole("button", { name: "Saved to My List" })).toBeInTheDocument();
  });

  it("shows a safe error when the save request fails", async () => {
    const user = userEvent.setup();
    mockedUseAuth.mockReturnValue(authValue());
    server.use(
      http.post(`${apiBaseUrl}/watchlist/:movieId`, () =>
        HttpResponse.json({ code: "database_unavailable", detail: "offline" }, { status: 503 }),
      ),
    );

    renderWithProviders(<WatchlistButton movieId={dune.id} />);
    await user.click(await screen.findByRole("button", { name: "Add to My List" }));

    expect(await screen.findByText("MovieMind is temporarily unavailable. Please try again shortly.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add to My List" })).toBeEnabled();
  });
});
