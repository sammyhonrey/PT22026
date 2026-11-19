import { act, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useAuth } from "@/features/auth/AuthProvider";
import type { AuthContextValue } from "@/features/auth/AuthProvider";
import { RatingControl } from "@/features/ratings/components/RatingControl";
import { apiBaseUrl } from "@/lib/api/client";
import { dune, makeRating, testUser } from "@/test/fixtures";
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

beforeEach(() => mockedUseAuth.mockReset());

describe("RatingControl", () => {
  it("prompts guests to sign in instead of showing controls", () => {
    mockedUseAuth.mockReturnValue(authValue(false));
    renderWithProviders(<RatingControl movieId={dune.id} />);

    expect(screen.getByText("Sign in to rate this movie.")).toBeInTheDocument();
    expect(screen.queryByRole("radiogroup")).not.toBeInTheDocument();
  });

  it("displays the rating returned by the authenticated API", async () => {
    mockedUseAuth.mockReturnValue(authValue());
    server.use(http.get(`${apiBaseUrl}/ratings/me/${dune.id}`, () => HttpResponse.json(makeRating(dune, 4))));

    renderWithProviders(<RatingControl movieId={dune.id} />);

    await waitFor(() => expect(screen.getByRole("radio", { name: "Rate 4.0 out of 5" })).toHaveAttribute("aria-checked", "true"));
    expect(screen.getByRole("button", { name: "Remove" })).toBeInTheDocument();
  });

  it("submits a half-step rating through the upsert endpoint", async () => {
    const user = userEvent.setup();
    mockedUseAuth.mockReturnValue(authValue());
    let body: unknown;
    server.use(
      http.post(`${apiBaseUrl}/ratings`, async ({ request }) => {
        body = await request.json();
        return HttpResponse.json(makeRating(dune, 4.5), { status: 201 });
      }),
    );

    renderWithProviders(<RatingControl movieId={dune.id} />);
    await user.click(await screen.findByRole("radio", { name: "Rate 4.5 out of 5" }));

    expect(await screen.findByText("Your rating of 4.5 was saved.")).toBeInTheDocument();
    expect(body).toEqual({ movie_id: dune.id, score: 4.5 });
  });

  it("disables the scale while a rating mutation is pending", async () => {
    const user = userEvent.setup();
    mockedUseAuth.mockReturnValue(authValue());
    let release!: () => void;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    server.use(
      http.post(`${apiBaseUrl}/ratings`, async () => {
        await gate;
        return HttpResponse.json(makeRating(dune, 3.5), { status: 201 });
      }),
    );

    renderWithProviders(<RatingControl movieId={dune.id} />);
    const target = await screen.findByRole("radio", { name: "Rate 3.5 out of 5" });
    await user.click(target);

    await waitFor(() => expect(target).toBeDisabled());
    expect(screen.getByRole("radio", { name: "Rate 5.0 out of 5" })).toBeDisabled();

    await act(async () => { release(); });
    expect(await screen.findByText("Your rating of 3.5 was saved.")).toBeInTheDocument();
  });

  it("shows a safe error and re-enables the scale after a failed rating", async () => {
    const user = userEvent.setup();
    mockedUseAuth.mockReturnValue(authValue());
    server.use(
      http.post(`${apiBaseUrl}/ratings`, () =>
        HttpResponse.json({ code: "internal_error", detail: "boom" }, { status: 500 }),
      ),
    );

    renderWithProviders(<RatingControl movieId={dune.id} />);
    const target = await screen.findByRole("radio", { name: "Rate 4.5 out of 5" });
    await user.click(target);

    expect(await screen.findByText("Something went wrong on our side. Please try again shortly.")).toBeInTheDocument();
    expect(target).not.toBeDisabled();
  });

  it("removes the current rating through the delete endpoint", async () => {
    const user = userEvent.setup();
    mockedUseAuth.mockReturnValue(authValue());
    server.use(http.get(`${apiBaseUrl}/ratings/me/${dune.id}`, () => HttpResponse.json(makeRating(dune, 4))));

    renderWithProviders(<RatingControl movieId={dune.id} />);
    await user.click(await screen.findByRole("button", { name: "Remove" }));

    expect(await screen.findByText("Your rating was removed.")).toBeInTheDocument();
  });
});
