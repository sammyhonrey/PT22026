import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Route, Routes } from "react-router-dom";

import { AppShell } from "@/components/layout/AppShell";
import { useAuth } from "@/features/auth/AuthProvider";
import type { AuthContextValue } from "@/features/auth/AuthProvider";
import { testUser } from "@/test/fixtures";
import { LocationProbe, renderWithProviders } from "@/test/test-utils";

vi.mock("@/features/auth/AuthProvider", () => ({ useAuth: vi.fn() }));

const mockedUseAuth = vi.mocked(useAuth);

function renderShell(value: AuthContextValue) {
  mockedUseAuth.mockReturnValue(value);
  return renderWithProviders(
    <Routes>
      <Route element={<AppShell />} path="/">
        <Route index element={<p>Home content</p>} />
      </Route>
      <Route
        element={
          <>
            <p>Search content</p>
            <LocationProbe />
          </>
        }
        path="/search"
      />
    </Routes>,
  );
}

function authValue(isAuthenticated: boolean, signOut = vi.fn(async () => undefined)): AuthContextValue {
  return {
    isAuthenticated,
    initializationError: null,
    isLoading: false,
    retryInitialization: vi.fn(),
    signIn: vi.fn(async () => testUser),
    signOut,
    signUp: vi.fn(async () => testUser),
    user: isAuthenticated ? testUser : null,
  };
}

beforeEach(() => mockedUseAuth.mockReset());

describe("AppShell", () => {
  it("shows guest account actions and hides personal ones", () => {
    renderShell(authValue(false));

    expect(screen.getByRole("link", { name: "Log in" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Create account/ })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Sign out" })).not.toBeInTheDocument();
    expect(screen.getByRole("contentinfo", { name: "Credits" })).toHaveTextContent(
      "This product uses the TMDB API but is not endorsed or certified by TMDB.",
    );
    expect(screen.getByRole("img", { name: "The Movie Database" })).toHaveAttribute(
      "src",
      "/tmdb-logo.svg",
    );
  });

  it("shows the signed-in identity and signs out on request", async () => {
    const user = userEvent.setup();
    const signOut = vi.fn();
    renderShell(authValue(true, signOut));

    expect(screen.getByRole("link", { name: "Open profile" })).toHaveTextContent(testUser.display_name!);
    await user.click(screen.getByRole("button", { name: "Sign out" }));
    expect(signOut).toHaveBeenCalledTimes(1);
  });

  it("submits the global search and navigates to the search route", async () => {
    const user = userEvent.setup();
    renderShell(authValue(false));

    const form = screen.getByRole("search");
    await user.type(within(form).getByRole("textbox"), "dune");
    await user.click(within(form).getByRole("button", { name: "Search" }));

    await waitFor(() => expect(screen.getByTestId("location")).toHaveTextContent("/search?q=dune"));
    expect(screen.getByText("Search content")).toBeInTheDocument();
  });
});
